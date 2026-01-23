from __future__ import annotations

import math
from typing import Optional, cast

import gymnasium as gym
import numpy as np

from mc_env.action import MinecraftAction
from mc_env.env import MinecraftEnv, BlockTypes, SOLID_BLOCKS, FOV_HEIGHT, FOV_WIDTH
from mc_env.observation import MinecraftObservation

class SimpleGoalRewardWrapper(gym.Wrapper[MinecraftObservation, np.ndarray, MinecraftObservation, np.ndarray]):
    def __init__(self, env: MinecraftEnv):
        super().__init__(env)
        self.mc_env = env
        self.step_penalty = {
            "survival_reward": 0.003,
        #     "only_forward": -0.001,
        #     "partly_forward": -0.0015,
        #     "default": -0.002,
        #     "backward": -0.0025,
        }
        self.goal_reward = 20.0
        self.death_penalty = -20.0
        self.fov_empty_penalty = -0.005
        self.fov_empty_do_nothing_bonus = 0.001
        self.unsafe_forward_penalty = -0.05

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
        self.goal_distance_weight = 0.2
        self.prev_dist_to_goal = None

        # self.goal_not_visible_penalty = -0.001

        self.goal_seen: bool = False

        self._steps = 0
        self._visited_blocks: set[tuple[int, int, int]] = set()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._steps = 0
        self._visited_blocks.clear()
        self.goal_seen = False

        obs, info = self.env.reset(seed=seed, options=options)
        start_point = self.mc_env.active_start_point
        self.prev_dist_to_goal = self._distance_between((start_point.x, start_point.y, start_point.z), (start_point.goalX, start_point.goalY, start_point.goalZ))
        self._mark_visited_if_solid(obs)
        return obs, info

    def step(self, action: np.ndarray):
        self._steps += 1
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        parsed_action = MinecraftAction.from_vector(action)
        info = dict(info)
        active_start_point = self.mc_env.active_start_point

        if obs.died:
            return obs, self.death_penalty, True, truncated, info
        if obs.standingOn == BlockTypes.GOAL_BLOCK:
            return obs, self.goal_reward, True, truncated, info

        reward = self.step_penalty["survival_reward"]
        # reward = self._get_step_penalty(parsed_action)
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
        dist_to_goal = self._distance_between((obs.x, obs.y, obs.z),
                                              (active_start_point.goalX, active_start_point.goalY, active_start_point.goalZ))
        dist_to_goal_delta = self.prev_dist_to_goal - dist_to_goal
        distance_shaping = self.goal_distance_weight * dist_to_goal_delta
        reward += distance_shaping
        info["shaping/distance_to_goal_delta"] = dist_to_goal_delta
        info["shaping/distance_shaping"] = distance_shaping
        self.prev_dist_to_goal = dist_to_goal

        visible_blocks = len([b for b in obs.fovBlocks if b != BlockTypes.AIR])
        fov_empty_rew = 0
        if visible_blocks == 0:
            if parsed_action.moveForward or parsed_action.moveBackward or parsed_action.moveRight or parsed_action.moveLeft:
                fov_empty_rew = self.fov_empty_penalty
            else:
                fov_empty_rew = self.fov_empty_do_nothing_bonus
        info["shaping/fov_empty_rew"] = float(fov_empty_rew)
        reward += fov_empty_rew

        unsafe_forward = 0
        if not self._has_ground_ahead(obs) and parsed_action.moveForward:
            unsafe_forward = self.unsafe_forward_penalty
        info["shaping/unsafe_forward"] = unsafe_forward
        reward += unsafe_forward

        if self._mark_visited_if_solid(obs):
            reward += float(self.new_block_reward)

        return obs, reward, terminated, truncated, info

    # def _get_step_penalty(self, action: MinecraftAction) -> float:
    #     if action.moveForward and not action.moveLeft and not action.moveRight:
    #         return float(self.step_penalty["only_forward"])
    #     elif action.moveForward:
    #         return float(self.step_penalty["partly_forward"])
    #     elif action.moveBackward:
    #         return float(self.step_penalty["backward"])
    #     else:
    #         return float(self.step_penalty["default"])

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

    @staticmethod
    def _distance_between(block1: tuple[float, float, float], block2: tuple[float, float, float]) -> float:
        return math.sqrt(
            (block1[0] - block2[0]) ** 2 +
            (block1[1] - block2[1]) ** 2 +
            (block1[2] - block2[2]) ** 2
        )

    @staticmethod
    def _has_ground_ahead(obs, max_dist=4):
        rows = range(int(FOV_HEIGHT * 0.6), FOV_HEIGHT)
        cols = range(FOV_WIDTH // 3, (FOV_WIDTH // 3) * 2)

        for r in rows:
            for c in cols:
                idx = r * FOV_WIDTH + c
                block = obs.fovBlocks[idx]
                dist = obs.fovDistances[idx]
                if block in SOLID_BLOCKS and dist is not None and dist < max_dist:
                    return True
        return False
