from __future__ import annotations

import math
from typing import Optional, cast

import gymnasium as gym
import numpy as np

from mc_env.action import MinecraftAction
from mc_env.env import MinecraftEnv, BlockTypes, SOLID_BLOCKS
from mc_env.observation import MinecraftObservation

class SimpleGoalRewardWrapper(gym.Wrapper[MinecraftObservation, np.ndarray, MinecraftObservation, np.ndarray]):
    def __init__(self, env: MinecraftEnv):
        super().__init__(env)
        self.step_penalty = {
            "only_forward": -0.001,
            "partly_forward": -0.0015,
            "default": -0.002,
            "backward": -0.0025,
        }
        self.goal_reward = 80.0
        self.death_penalty = -80.0
        self.fov_empty_penalty = -0.005

        self.new_block_reward = 0.02
        # self.goal_reward = 40.0
        # self.death_penalty = -40.0
        #
        # self.pitch_range_center = 20.0
        # self.pitch_range_width = 30.0
        # self.pitch_range_reward = 0.0002
        #
        # self.new_block_reward = 0.01
        self.max_steps = 500

        self.goal_first_seen_bonus = 0.5
        self.goal_seen_bonus = 0.0005
        self.goal_distance_weight = 0.01

        # self.goal_not_visible_penalty = -0.001

        self.goal_seen: bool = False

        self._steps = 0
        self._visited_blocks: set[tuple[int, int, int]] = set()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._steps = 0
        self._visited_blocks.clear()
        self.goal_seen = False

        obs, info = self.env.reset(seed=seed, options=options)
        self._mark_visited_if_solid(obs)
        return obs, info

    def step(self, action: np.ndarray):
        self._steps += 1
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        parsed_action = MinecraftAction.from_vector(action)
        info = dict(info)

        if obs.died:
            return obs, self.death_penalty, True, truncated, info
        if obs.standingOn == BlockTypes.GOAL_BLOCK:
            return obs, self.goal_reward, True, truncated, info

        reward = self._get_step_penalty(parsed_action)
        info["shaping/step_penalty"] = float(reward)

        # pitch_rew = self._reward_pitch_range(obs)
        # info["shaping/pitch_rew"] = float(pitch_rew)
        # reward +=pitch_rew

        if self._steps >= self.max_steps:
            return obs, reward, terminated, True, info

        goal_dist = SimpleGoalRewardWrapper._get_goal_visible_distance(obs)
        if goal_dist is not None:
            if not self.goal_seen:
                reward += float(self.goal_first_seen_bonus)
                self.goal_seen = True
            else:
                reward += float(self.goal_seen_bonus)

        # --- Distance to goal shaping ---
        env = cast(MinecraftEnv, self.env)
        dist_to_goal = math.sqrt(
            (obs.x - env.active_start_point.goalX) ** 2 +
            (obs.y - env.active_start_point.goalY) ** 2 +
            (obs.z - env.active_start_point.goalZ) ** 2
        )
        # Penalize distance (closer = higher reward)
        distance_shaping = -self.goal_distance_weight * dist_to_goal
        reward += distance_shaping
        info["shaping/distance_to_goal"] = dist_to_goal
        info["shaping/distance_shaping"] = distance_shaping

        visible_blocks = len([b for b in obs.fovBlocks if b != BlockTypes.AIR])
        fov_empty_penalty = 0
        if visible_blocks == 0:
            fov_empty_penalty = self.fov_empty_penalty  # "STOP standing still blind!"
        info["shaping/fov_empty_penalty"] = float(fov_empty_penalty)
        reward += fov_empty_penalty

        if self._mark_visited_if_solid(obs):
            reward += float(self.new_block_reward)

        return obs, reward, terminated, truncated, info

    def _get_step_penalty(self, action: MinecraftAction) -> float:
        if action.moveForward and not action.moveLeft and not action.moveRight:
            return float(self.step_penalty["only_forward"])
        elif action.moveForward:
            return float(self.step_penalty["partly_forward"])
        elif action.moveBackward:
            return float(self.step_penalty["backward"])
        else:
            return float(self.step_penalty["default"])

    # def _reward_pitch_range(self, obs: MinecraftObservation) -> float:
    #     # Smooth Gaussian peak @center °
    #     pitch_reward = self.pitch_range_reward * np.exp(
    #         -((float(obs.pitch) - self.pitch_range_center) ** 2) / (2 * self.pitch_range_width ** 2)
    #     )
    #     return float(pitch_reward)

    def _mark_visited_if_solid(self, obs: MinecraftObservation) -> bool:
        if obs.standingOn not in SOLID_BLOCKS:
            return False
        pos = self._block_pos(obs)
        if pos is None:
            return False
        if pos in self._visited_blocks:
            return False
        self._visited_blocks.add(pos)
        return True

    @staticmethod
    def _block_pos(obs: MinecraftObservation) -> Optional[tuple[int, int, int]]:
        x = obs.x
        y = obs.y
        z = obs.z
        if x is None or y is None or z is None:
            return None
        return int(math.floor(float(x))), int(math.floor(float(y))), int(math.floor(float(z)))

    @staticmethod
    def _get_goal_visible_distance(obs: MinecraftObservation) -> Optional[float]:
        min_dist: Optional[float] = None
        for blk, dist in zip(obs.fovBlocks, obs.fovDistances):
            if int(blk) == int(BlockTypes.GOAL_BLOCK) and dist is not None and float(dist) >= 0:
                d = float(dist)
                if min_dist is None or d < min_dist:
                    min_dist = d
        return min_dist
