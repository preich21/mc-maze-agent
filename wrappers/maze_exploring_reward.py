"""
Reward wrapper for reaching the goal block and exploring the maze like a player would do.

Reward shaping:
- step penalty: small negative reward per step to encourage shorter paths
- goal reward: giant reward when standing on GOAL_BLOCK, episode terminates; medium reward for seeing the goal for the first time; small shaping reward for getting closer to the goal when visible
- exploration: small reward for stepping onto new solid blocks
- stupid camera positioning: small penalty
- moving into a wall: small penalty
"""
from __future__ import annotations

import math
from collections import deque
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
        self.new_block_reward = 0.01
        self.stupid_camera_positioning_penalty = -0.0
        self.wall_collision_penalty = -0.005
        self.unnecessary_jump_penalty = -0.005
        self.no_progress_penalty = -0.1
        self.goal_first_seen_bonus = 0.0
        self.goal_distance_weight = 0.05

        self._visited_blocks: set[tuple[int, int, int]] = set()
        self.bfs_distances: List[List[Optional[int]]] = []
        self.previous_distance_to_goal = 5_000
        self.best_distance_to_goal = 5_000
        self.steps_since_best_distance = 0

        self._episode = 0
        self._steps = 0
        self.maze_size = 1
        self._reached_goal_streak = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._episode += 1
        self._steps = 0
        self._visited_blocks.clear()
        self.previous_distance_to_goal = 5_000
        self.best_distance_to_goal = 5_000
        self.steps_since_best_distance = 0

        if self._reached_goal_streak >= 20:
            self.maze_size += 1
            self._reached_goal_streak = 0

        new_options = options.copy() if options is not None else {}
        new_options['mazeSize'] = self.maze_size

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

        print(f"Starting episode {self._episode} with mazeSize={new_options['mazeSize']}")

        # Mark start position as visited (if we're standing on a solid block)
        self._mark_visited_if_solid(obs)

        goal_distance = self._bfs_distance_to_goal(obs)
        self.previous_distance_to_goal = goal_distance if goal_distance is not None else 5_000
        obs.maze_distance = self.previous_distance_to_goal
        return obs, info

    def step(self, action: np.ndarray):
        self._steps += 1
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        parsed_action = MinecraftAction.from_vector(action, env=self.env)

        # Check terminal conditions
        if obs.standingOn == BlockTypes.GOAL_BLOCK:
            reward = self.goal_reward
            terminated = True
            self._reached_goal_streak += 1
            print(f"Reached GOAL_BLOCK after {self._steps} steps in episode {self._episode}! Terminating episode.")
            return obs, reward, terminated, truncated, info
        if self._steps >= self.max_steps: # don't return since the last step's reward should still be given i think
            truncated = True
            self._reached_goal_streak = 0
            print("Max steps reached, truncating episode.")

        # Default per-step penalty
        reward = self.step_penalty

        # Goal distance reward
        goal_distance = self._bfs_distance_to_goal(obs)
        goal_distance = goal_distance if goal_distance is not None else 5_000
        goal_distance_delta = (self.previous_distance_to_goal - goal_distance)
        if goal_distance_delta > 2.0: # the agent cant possibly cover a distance >2 in one step
            goal_distance_delta = 0.0
        reward += self.goal_distance_weight * goal_distance_delta
        self.previous_distance_to_goal = goal_distance

        # Exploration bonus: reward stepping onto new solid blocks, but only for small mazes
        if self._mark_visited_if_solid(obs) and self.maze_size < 4:
            reward += self.new_block_reward

        # Penalty for putting the camera straight against a wall or looking too far up/down
        if self._stupid_camera_positioning(obs):
            reward += self.stupid_camera_positioning_penalty

        # Penalty for trying to move into a wall
        if self._wall_collision(obs, parsed_action):
            reward += self.wall_collision_penalty

        # Penalty for no progress (i.e. no actual movement) over multiple steps
        if goal_distance < self.best_distance_to_goal:
            self.best_distance_to_goal = goal_distance
            self.steps_since_best_distance = 0
        elif self.steps_since_best_distance >= 10:
            reward += self.no_progress_penalty
            self.steps_since_best_distance += 1
        else:
            self.steps_since_best_distance += 1

        # Penalty for unnecessary jumping (unnecessary being always since the maze is flat)
        if parsed_action.jump > 0.0:
            reward += self.unnecessary_jump_penalty

        obs.maze_distance = goal_distance
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
        if abs(pitch) > 20.0:
            return True

        if obs.fovDistances[1275] < 1.0:
            return True

        return False

    @staticmethod
    def _wall_collision(obs: MinecraftObservation, action: MinecraftAction) -> bool:
        """
        Return True if the agent attempted to move into a wall.
        """
        if action.moveForward == 0.0 and action.moveSidewards == 0.0:
            return False

        return abs(obs.dx) <= 0.01 and abs(obs.dz) <= 0.01


    @staticmethod
    def _bfs_distance_field_from_goal(
            maze: List[List[bool]],
            goal: Tuple[int, int],
    ) -> List[List[Optional[int]]]:
        """
        Builds a distance field dist[r][c] = shortest steps from (r,c) to goal,
        or None if unreachable or blocked.
        maze[r][c] == True  => wall (blocked)
        maze[r][c] == False => open
        """
        rows = len(maze)
        cols = len(maze[0]) if rows else 0
        gr, gc = goal

        dist: List[List[Optional[int]]] = [[None] * cols for _ in range(rows)]

        # Basic bounds / validity checks
        if not (0 <= gr < rows and 0 <= gc < cols):
            return dist
        if maze[gr][gc]:  # goal is inside a wall
            return dist

        q = deque([(gr, gc)])
        dist[gr][gc] = 0

        while q:
            r, c = q.popleft()
            d = dist[r][c]
            assert d is not None

            # 4-neighbors
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
        Option 1: Integer BFS distance + within-cell continuous shaping.

        We compute:
          d = base_bfs(cell) + alpha * dist_to_target_center

        where target_center is the center of the neighbor cell that has the smallest
        BFS distance (i.e., the next step on a shortest path). This yields a smooth
        value that decreases as the agent moves toward the next optimal cell.
        """
        x_exact = obs.x
        z_exact = obs.z
        if x_exact is None or z_exact is None:
            return None

        # Use floor to map continuous position to grid cell
        x = int(math.floor(float(x_exact)))
        z = int(math.floor(float(z_exact)))

        rows = len(self.bfs_distances)
        cols = len(self.bfs_distances[0]) if rows else 0
        if not (0 <= x < rows and 0 <= z < cols):
            return None

        base = self.bfs_distances[x][z]
        if base is None:
            return None

        # Find the best (downhill) neighbor: smallest BFS distance among 4-neighbors
        best_n = (x, z)
        best_d = base

        # 4-neighborhood
        for nx, nz in ((x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)):
            if 0 <= nx < rows and 0 <= nz < cols:
                nd = self.bfs_distances[nx][nz]
                if nd is not None and nd < best_d:
                    best_d = nd
                    best_n = (nx, nz)

        # Continuous component: distance from current position to center of best neighbor cell.
        # Cell center is (cell_x + 0.5, cell_z + 0.5)
        tx, tz = best_n
        target_x = tx + 0.5
        target_z = tz + 0.5

        dx = float(x_exact) - target_x
        dz = float(z_exact) - target_z
        dist_to_target_center = math.sqrt(dx * dx + dz * dz)

        # Weight of the continuous term. Keep small so BFS dominates.
        # With a 1x1 cell, dist is in [0, ~0.707].
        alpha = 0.25  # tweak 0.1..0.5 if you want more/less density

        # Note: we use base BFS of current cell, not of neighbor. The continuous term
        # drives motion toward the next optimal cell; once you cross into it, base drops.
        return float(base) + alpha * dist_to_target_center
