"""Reward wrapper for reaching the goal block.

Reward shaping:
- step penalty: small negative reward per step to encourage shorter paths
- goal reward: +1.0 when standing on GOAL_BLOCK, episode terminates
- survival: 0 otherwise
"""
from __future__ import annotations

import math
from typing import Optional

import gymnasium as gym

from env_mc import MinecraftEnv, MinecraftObservation, MinecraftAction, BlockTypes, SOLID_BLOCKS

class SimpleGoalRewardWrapper(gym.Wrapper[MinecraftObservation, MinecraftAction, MinecraftObservation, MinecraftAction]):
    def __init__(self, env: MinecraftEnv):
        super().__init__(env)
        self.step_penalty = -0.001
        self.goal_reward = 5.0
        self.death_penalty = -0.5
        self.new_block_reward = 0.2
        self.max_steps = 500

        # Goal visibility shaping
        self.goal_first_seen_bonus = 0.5
        self.goal_distance_weight = 0.02
        self.goal_seen: bool = False
        self.last_goal_distance: Optional[float] = None

        self._steps = 0
        self._visited_blocks: set[tuple[int, int, int]] = set()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._steps = 0
        self._visited_blocks.clear()
        self.goal_seen = False
        self.last_goal_distance = None
        obs, info = self.env.reset(seed=seed, options=options)
        # Mark start position as visited (if we're standing on a solid block)
        self._mark_visited_if_solid(obs)
        return obs, info

    def step(self, action: MinecraftAction):
        self._steps += 1
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        # Check terminal conditions - no step penalty needed here
        if obs.died:
            reward = self.death_penalty
            terminated = True
            return obs, reward, terminated, truncated, info
        elif obs.standingOn == BlockTypes.GOAL_BLOCK:
            reward = self.goal_reward
            terminated = True
            return obs, reward, terminated, truncated, info

        # Default per-step reward
        reward = self.step_penalty

        if self._steps >= self.max_steps:
            truncated = True
            return obs, reward, terminated, truncated, info

        # Goal visibility shaping
        goal_dist = SimpleGoalRewardWrapper._get_goal_visible_distance(obs)
        if goal_dist is not None:
            if not self.goal_seen:
                reward += self.goal_first_seen_bonus
                self.goal_seen = True
            if self.last_goal_distance is not None:
                # Positive if distance decreased, negative if increased
                delta = self.last_goal_distance - goal_dist
                # Clamp to avoid huge jumps from measurement noise
                delta = max(-1.0, min(1.0, delta))
                reward += self.goal_distance_weight * delta
            self.last_goal_distance = goal_dist

        # Exploration bonus: reward stepping onto new solid blocks.
        if self._mark_visited_if_solid(obs):
            reward += self.new_block_reward

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
