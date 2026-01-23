"""
Reward wrapper for reaching the goal block and exploring the maze like a player would do.

Reward shaping:
- step penalty: small negative reward per step to encourage shorter paths
- goal reward: giant reward when standing on GOAL_BLOCK, episode terminates
- exploration: small reward for stepping onto new solid blocks (only for small mazes)
- moving into a wall: small penalty (only when there was meaningful move intent)
- no progress: penalty when there was no meaningful progress in BFS distance over a rolling window
"""
from __future__ import annotations

import math
import random
from collections import deque, defaultdict
from typing import Optional, List, Tuple, Any

import gymnasium as gym
import numpy as np

from mc_env.action import MinecraftAction
from mc_env.env import MinecraftEnv, BlockTypes, SOLID_BLOCKS
from mc_env.observation import MinecraftObservation


class MazeExploringRewardWrapper(gym.Wrapper[MinecraftObservation, np.ndarray, MinecraftObservation, np.ndarray]):
    def __init__(self, env: MinecraftEnv):
        super().__init__(env)
        self.max_steps = 500

        self.step_penalty = -0.005
        self.goal_reward = 100.0
        self.new_block_reward = 0.002
        self.wall_collision_penalty = -0.005
        self.unnecessary_jump_penalty = -0.005

        # Progress / stuck shaping
        self.goal_distance_weight = 0.07
        self.no_progress_penalty = -0.01
        self.no_progress_window = 10

        # Collision intent gating (avoid punishing tiny nudges / near-zero actions)
        self.collision_intent_threshold = 0.1

        # If BFS distance is temporarily unavailable, apply a small penalty but do NOT jump to 5000
        self.unreachable_penalty = -0.02

        # Optional camera penalty (kept off by default)
        self.stupid_camera_positioning_penalty = -0.0

        self._visited_blocks: set[tuple[int, int, int]] = set()
        self.bfs_distances: List[List[Optional[int]]] = []

        self.previous_distance_to_goal: float = 5000.0
        self._recent_goal_distances: deque[float] = deque(maxlen=self.no_progress_window)

        self.gamma = 0.99
        self.tau = 10.0

        self._episode = 0
        self._steps = 0

        self.maze_size = 1
        self.current_target_maze_size = 5
        self.max_maze_size = 6

        self.success_window = 25
        self.episode_outcomes = defaultdict(
            lambda: deque(maxlen=self.success_window)
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._episode += 1
        self._steps = 0
        self._visited_blocks.clear()
        self.previous_distance_to_goal = 5000.0
        self._recent_goal_distances.clear()

        self._update_target_size()

        self.maze_size = self._generate_random_maze_size()
        self.max_steps = min(int(50 * (self.maze_size ** 2)), 1000)
        if self.maze_size >= 5:
            self.no_progress_penalty = 0.0
        else:
            self.no_progress_penalty = -0.01

        new_options = options.copy() if options is not None else {}
        new_options["mazeSize"] = self.maze_size

        obs: MinecraftObservation
        info: tuple[MinecraftObservation, dict[str, Any]]
        while True:
            obs, info = self.env.reset(seed=seed, options=new_options)
            if obs.standingOn == BlockTypes.START_BLOCK:
                break
            print("Warning: Reset did not place agent at start position, retrying...")

        goal_coordinate = 2 if self.maze_size == 1 else (self.maze_size * 3 - 2)
        self.bfs_distances = self._bfs_distance_field_from_goal(
            maze=obs.maze,
            goal=(goal_coordinate, goal_coordinate),
        )

        start_distance = self.bfs_distances[1][1]
        if start_distance is not None:
            self.tau = max(5.0, float(start_distance))
        else:
            self.tau = 10.0

        print(f"Starting episode {self._episode} with mazeSize={new_options['mazeSize']}")

        # Mark start position as visited (if we're standing on a solid block)
        self._mark_visited_if_solid(obs)

        goal_distance = self._bfs_distance_to_goal(obs)
        if goal_distance is not None:
            self.previous_distance_to_goal = float(goal_distance)
            self._recent_goal_distances.append(float(goal_distance))

        obs.maze_distance = goal_distance
        return obs, info

    def step(self, action: np.ndarray):
        self._steps += 1
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        parsed_action = MinecraftAction.from_vector(action, env=self.env)

        # Terminal: goal reached
        if obs.standingOn == BlockTypes.GOAL_BLOCK:
            reward = self.goal_reward
            terminated = True
            self._record_episode_outcome(success=True)
            print(f"Reached GOAL_BLOCK after {self._steps} steps in episode {self._episode}! Terminating episode.")
            return obs, reward, terminated, truncated, info

        # Truncation: max steps
        if self._steps >= self.max_steps:
            truncated = True
            self._record_episode_outcome(success=False)
            print("Max steps reached, truncating episode.")

        reward = self.step_penalty

        # Goal distance shaping (robust handling of None + symmetric clipping)
        goal_distance = self._bfs_distance_to_goal(obs)
        if goal_distance is None:
            # Do not jump to 5000 (that creates huge noisy deltas). Keep the last valid value.
            reward += self.unreachable_penalty
            goal_distance = self.previous_distance_to_goal
        else:
            previous_distance_normalized = -math.tanh(self.previous_distance_to_goal / self.tau)
            goal_distance_normalized = -math.tanh(goal_distance / self.tau)
            shaping = self.gamma * goal_distance_normalized - previous_distance_normalized
            reward += self.goal_distance_weight * shaping
            self.previous_distance_to_goal = goal_distance

        # Keep a rolling window for "no progress" checks
        self._recent_goal_distances.append(float(goal_distance))

        # Exploration bonus (only for small mazes)
        if self.maze_size <=4 and self._mark_visited_if_solid(obs):
            reward += self.new_block_reward

        # Optional camera penalty (off by default)
        if self.stupid_camera_positioning_penalty != 0.0 and self._stupid_camera_positioning(obs):
            reward += self.stupid_camera_positioning_penalty

        # Wall collision penalty (only when there was meaningful intent)
        if self._wall_collision(obs, parsed_action, intent_threshold=self.collision_intent_threshold):
            reward += self.wall_collision_penalty

        # No progress penalty (window-based; avoids "best-ever" trap)
        # Penalize if we did NOT achieve any improvement vs the best in the last window.
        # This targets oscillation/jitter without punishing necessary small detours too harshly.
        if len(self._recent_goal_distances) == self._recent_goal_distances.maxlen:
            best_recent = min(self._recent_goal_distances)
            if float(goal_distance) > best_recent + 1e-9:  # no new min this step
                reward += self.no_progress_penalty

        # Jump penalty
        if parsed_action.jump > 0.0:
            reward += self.unnecessary_jump_penalty

        obs.maze_distance = float(goal_distance)
        return obs, reward, terminated, truncated, info

    def _mark_visited_if_solid(self, obs: MinecraftObservation) -> bool:
        """Return True if a new solid block position was visited."""
        standing_on = obs.standingOn
        if standing_on not in SOLID_BLOCKS:
            return False

        x = obs.x
        y = obs.y
        z = obs.z
        if x is None or y is None or z is None:
            return False

        bx = int(math.floor(float(x)))
        by = int(math.floor(float(y)))
        bz = int(math.floor(float(z)))
        key = (bx, by, bz)
        if key in self._visited_blocks:
            return False
        self._visited_blocks.add(key)
        return True

    @staticmethod
    def _get_goal_visible_distance(obs: MinecraftObservation) -> Optional[float]:
        """Return the minimum visible GOAL_BLOCK distance or None if not visible."""
        min_dist: Optional[float] = None
        for blk, dist in zip(obs.fovBlocks, obs.fovDistances):
            if blk == BlockTypes.GOAL_BLOCK and dist is not None and float(dist) >= 0:
                d = float(dist)
                if min_dist is None or d < min_dist:
                    min_dist = d
        return min_dist

    def _stupid_camera_positioning(self, obs: MinecraftObservation) -> bool:
        """Return True if the camera is positioned stupidly (e.g., looking straight up or down, or directly against a wall)."""
        if self._get_goal_visible_distance(obs) is not None:
            return False

        pitch = obs.pitch
        if pitch is not None and abs(pitch) > 20.0:
            return True

        # Guard against short fov arrays
        if obs.fovDistances is not None and len(obs.fovDistances) > 1275:
            if obs.fovDistances[1275] is not None and 0 <= float(obs.fovDistances[1275]) < 1.0:
                return True

        return False

    @staticmethod
    def _wall_collision(obs: MinecraftObservation, action: MinecraftAction, intent_threshold: float = 0.2) -> bool:
        """
        Return True if the agent attempted to move meaningfully but displacement was (almost) zero.
        """
        intent = abs(action.moveForward) + abs(action.moveSidewards)
        if intent < intent_threshold:
            return False

        return obs.dx is not None and obs.dz is not None and abs(obs.dx) <= 0.01 and abs(obs.dz) <= 0.01

    @staticmethod
    def _bfs_distance_field_from_goal(
        maze: List[List[bool]],
        goal: Tuple[int, int],
    ) -> List[List[Optional[int]]]:
        """
        Builds a distance field dist[c][r] = shortest steps from (c, r) to goal,
        or None if unreachable or blocked.
        maze[c][r] == True  => wall (blocked)
        maze[c][r] == False => open
        """
        rows = len(maze)
        cols = len(maze[0]) if rows else 0
        gr, gc = goal

        dist: List[List[Optional[int]]] = [[None] * cols for _ in range(rows)]

        if not (0 <= gr < rows and 0 <= gc < cols):
            return dist
        if maze[gr][gc]:
            return dist

        q = deque([(gr, gc)])
        dist[gr][gc] = 0

        while q:
            r, c = q.popleft()
            d = dist[r][c]
            assert d is not None

            if r > 0 and not maze[r - 1][c] and dist[r - 1][c] is None:
                dist[r - 1][c] = d + 1
                q.append((r - 1, c))
            if r + 1 < rows and not maze[r + 1][c] and dist[r + 1][c] is None:
                dist[r + 1][c] = d + 1
                q.append((r + 1, c))
            if c > 0 and not maze[r][c - 1] and dist[r][c - 1] is None:
                dist[r][c - 1] = d + 1
                q.append((r, c - 1))
            if c + 1 < cols and not maze[r][c + 1] and dist[r][c + 1] is None:
                dist[r][c + 1] = d + 1
                q.append((r, c + 1))

        return dist

    def _bfs_distance_to_goal(self, obs: MinecraftObservation) -> Optional[float]:
        """
        Integer BFS distance + within-cell continuous shaping.

        d = base_bfs(cell) + alpha * dist_to_target_center

        where target_center is the center of the neighbor cell that has the smallest
        BFS distance (i.e., the next step on a shortest path).
        """
        x_exact = obs.x
        z_exact = obs.z
        if x_exact is None or z_exact is None:
            return None

        x = int(math.floor(float(x_exact)))
        z = int(math.floor(float(z_exact)))

        rows = len(self.bfs_distances)
        cols = len(self.bfs_distances[0]) if rows else 0
        if not (0 <= x < rows and 0 <= z < cols):
            return None

        base = self.bfs_distances[x][z]
        if base is None:
            return None
        elif base == 0.0:
            return 0.0

        best_n = (x, z)
        best_d = base

        for nx, nz in ((x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)):
            if 0 <= nx < rows and 0 <= nz < cols:
                nd = self.bfs_distances[nx][nz]
                if nd is not None and nd < best_d:
                    best_d = nd
                    best_n = (nx, nz)

        tx, tz = best_n
        target_x = tx + 0.5
        target_z = tz + 0.5

        dx = float(x_exact) - target_x
        dz = float(z_exact) - target_z
        dist_to_target_center = math.sqrt(dx * dx + dz * dz)

        alpha = 1.0
        return float(self.bfs_distances[tx][tz]) + alpha * dist_to_target_center


    def _generate_random_maze_size(self):
        s = self.current_target_maze_size

        choices = [
            (s, 0.4),
            (s - 1, 0.3),
            (s - 2, 0.2),
            (random.randint(1, 3), 0.1),
        ]

        r = random.random()
        acc = 0.0
        for size, prob in choices:
            acc += prob
            if r <= acc:
                return max(1, size)

        return s


    def _record_episode_outcome(self, success: bool):
        size = self.maze_size
        self.episode_outcomes[size].append(success)


    def _success_rate(self, size: int):
        outcomes = self.episode_outcomes[size]
        if len(outcomes) < self.success_window:
            return None
        return sum(outcomes) / len(outcomes)


    def _update_target_size(self):
        rate = self._success_rate(self.current_target_maze_size)
        if rate is None:
            return

        if rate > 0.8:
            self.current_target_maze_size = min(
                self.current_target_maze_size + 1,
                self.max_maze_size
            )
            print(f"↑ Increasing target size to {self.current_target_maze_size}")

        elif rate < 0.2 and self.current_target_maze_size > 1:
            self.current_target_maze_size -= 1
            print(f"↓ Decreasing target size to {self.current_target_maze_size}")
