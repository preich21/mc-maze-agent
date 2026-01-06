"""Reward wrapper for reaching the goal block.

Reward shaping:
- step penalty: small negative reward per step to encourage shorter paths
- goal reward: +1.0 when standing on GOAL_BLOCK, episode terminates
- survival: 0 otherwise
"""
from __future__ import annotations

import math
import gymnasium as gym

from env_mc import MinecraftEnv
from types import MinecraftObservation, MinecraftAction

GOAL_BLOCK = "GOAL_BLOCK"
START_BLOCK = "START_BLOCK"
SOLID_BLOCKS = {"BLOCK", START_BLOCK, GOAL_BLOCK}


class SimpleGoalRewardWrapper(gym.Wrapper[MinecraftObservation, MinecraftAction, MinecraftObservation, MinecraftAction]):
    def __init__(self, env: MinecraftEnv):
        super().__init__(env)
        self.step_penalty = -0.001
        self.goal_reward = 1.0
        self.death_penalty = -1.0
        self.new_block_reward = 0.01
        self.max_steps = 500
        self._steps = 0
        self._visited_blocks: set[tuple[int, int, int]] = set()

    def reset(self, **kwargs):  # noqa: ANN003
        self._steps = 0
        self._visited_blocks.clear()
        obs, info = self.env.reset(**kwargs)
        # Mark start position as visited (if we're standing on a solid block)
        self._mark_visited_if_solid(obs)
        return obs, info

    def step(self, action: MinecraftAction):
        self._steps += 1
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        standing_on = obs.standingOn
        died = bool(obs.died)

        if died:
            reward = self.death_penalty
            terminated = True
            return obs, reward, terminated, truncated, info
        elif standing_on == GOAL_BLOCK:
            reward = self.goal_reward
            terminated = True
            return obs, reward, terminated, truncated, info

        reward = self.step_penalty
        if self._steps >= self.max_steps:
            truncated = True
            return obs, reward, terminated, truncated, info


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
