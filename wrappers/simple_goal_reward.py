from __future__ import annotations

import math
from typing import Optional

import gymnasium as gym
import numpy as np

from mc_env.env import MinecraftEnv, BlockTypes, SOLID_BLOCKS, FOV_HEIGHT, FOV_WIDTH
from mc_env.observation import MinecraftObservation


class SimpleGoalRewardWrapper(gym.Wrapper[MinecraftObservation, np.ndarray, MinecraftObservation, np.ndarray]):
    def __init__(self, env: MinecraftEnv):
        super().__init__(env)
        self.step_penalty = -0.002
        self.goal_reward = 10.0
        self.death_penalty = -5.0

        self.new_block_reward = 0.02
        self.max_steps = 500

        self.goal_first_seen_bonus = 1.0
        self.goal_distance_weight = 0.2

        # self.air_step_penalty = -0.05
        # self.no_progress_penalty = -0.01
        self.goal_not_visible_penalty = -0.001
        #
        # # ---- Vision/safety shaping ----
        # self.bottom_air_frac_threshold = 0.90
        # self.bottom_all_air_penalty = -0.03  # stronger than before (was -0.01)
        #
        # self.look_ahead_solid_frac_threshold = 0.10
        # self.look_ahead_ground_reward = 0.01
        #
        # # ---- explicit pitch band shaping (encourage looking down) ----
        # self.pitch_target_lo = 10.0
        # self.pitch_target_hi = 70.0
        # self.pitch_in_band_reward = 0.01      # small but consistent
        # self.pitch_out_of_band_penalty = -0.005  # gentle pressure to enter band
        #
        # # ---- Movement shaping ----
        # self.forward_step_penalty_reduction = 0.002
        # self.forward_disp_clip = 0.25

        self.goal_seen: bool = False
        self.last_goal_distance: Optional[float] = None

        self._steps = 0
        self._visited_blocks: set[tuple[int, int, int]] = set()
        # self._last_block_pos: Optional[tuple[int, int, int]] = None
        # self._last_xz: Optional[tuple[float, float]] = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._steps = 0
        self._visited_blocks.clear()
        self.goal_seen = False
        self.last_goal_distance = None
        # self._last_block_pos = None
        # self._last_xz = None

        obs, info = self.env.reset(seed=seed, options=options)
        self._mark_visited_if_solid(obs)
        # self._last_block_pos = self._block_pos(obs)
        # self._last_xz = (float(obs.x), float(obs.z))
        return obs, info

    def step(self, action: np.ndarray):
        self._steps += 1
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        if obs.died:
            return obs, self.death_penalty, True, truncated, info
        if obs.standingOn == BlockTypes.GOAL_BLOCK:
            return obs, self.goal_reward, True, truncated, info

        reward = float(self.step_penalty)

        if self._steps >= self.max_steps:
            return obs, reward, terminated, True, info

        # grounded = obs.standingOn in SOLID_BLOCKS

        # if not grounded:
        #     reward += float(self.air_step_penalty)

        # r_fp = self._reward_for_forward_progress(obs, grounded=grounded)
        # reward += r_fp
        #
        # r_bottom = self._reward_for_ground_visibility(obs)
        # reward += r_bottom
        #
        # r_ahead = self._reward_for_looking_ahead(obs)
        # reward += r_ahead
        #
        # r_pitch = self._reward_for_pitch_band(obs)
        # reward += r_pitch
        #
        # cur_pos = self._block_pos(obs)
        # if self._last_block_pos is not None and cur_pos == self._last_block_pos:
        #     reward += float(self.no_progress_penalty)
        # self._last_block_pos = cur_pos

        goal_dist = SimpleGoalRewardWrapper._get_goal_visible_distance(obs)
        if goal_dist is not None:
            if not self.goal_seen:
                reward += float(self.goal_first_seen_bonus)
                self.goal_seen = True

            if self.last_goal_distance is not None:
                delta = self.last_goal_distance - goal_dist
                delta = max(-1.0, min(1.0, delta))
                reward += float(self.goal_distance_weight) * float(delta)

            self.last_goal_distance = goal_dist
        else:
            reward += float(self.goal_not_visible_penalty)

        if self._mark_visited_if_solid(obs):
            reward += float(self.new_block_reward)

        # Expose shaping components for TensorBoard/debug callback (optional)
        # info = dict(info)
        # info["shaping/forward_progress"] = float(r_fp)
        # info["shaping/bottom_visibility"] = float(r_bottom)
        # info["shaping/look_ahead"] = float(r_ahead)
        # info["shaping/pitch_band_reward"] = float(r_pitch)
        # info["shaping/pitch_deg"] = float(obs.pitch)

        return obs, reward, terminated, truncated, info

    # def _reward_for_pitch_band(self, obs: MinecraftObservation) -> float:
    #     pitch = float(obs.pitch)
    #     if self.pitch_target_lo <= pitch <= self.pitch_target_hi:
    #         return float(self.pitch_in_band_reward)
    #     return float(self.pitch_out_of_band_penalty)
    #
    # def _reward_for_forward_progress(self, obs: MinecraftObservation, *, grounded: bool) -> float:
    #     if not grounded:
    #         self._last_xz = (float(obs.x), float(obs.z))
    #         return 0.0
    #     if self._last_xz is None:
    #         self._last_xz = (float(obs.x), float(obs.z))
    #         return 0.0
    #
    #     last_x, last_z = self._last_xz
    #     dx = float(obs.x) - float(last_x)
    #     dz = float(obs.z) - float(last_z)
    #
    #     yaw_rad = math.radians(float(obs.yaw))
    #     fwd_x = math.sin(yaw_rad)
    #     fwd_z = math.cos(yaw_rad)
    #
    #     forward_disp = dx * fwd_x + dz * fwd_z
    #     forward_disp = float(np.clip(forward_disp, -self.forward_disp_clip, self.forward_disp_clip))
    #
    #     self._last_xz = (float(obs.x), float(obs.z))
    #
    #     if forward_disp <= 0.0:
    #         return 0.0
    #     frac = forward_disp / float(self.forward_disp_clip)
    #     return float(self.forward_step_penalty_reduction) * float(frac)
    #
    # def _reward_for_ground_visibility(self, obs: MinecraftObservation) -> float:
    #     air_frac = self._region_air_fraction(obs, y0_frac=0.75, y1_frac=1.00, x0_frac=0.00, x1_frac=1.00)
    #     if air_frac >= float(self.bottom_air_frac_threshold):
    #         return float(self.bottom_all_air_penalty)
    #     return 0.0
    #
    # def _reward_for_looking_ahead(self, obs: MinecraftObservation) -> float:
    #     air_frac = self._region_air_fraction(obs, y0_frac=0.60, y1_frac=0.85, x0_frac=0.30, x1_frac=0.70)
    #     solid_frac = 1.0 - air_frac
    #     if solid_frac >= float(self.look_ahead_solid_frac_threshold):
    #         return float(self.look_ahead_ground_reward)
    #     return 0.0
    #
    # @staticmethod
    # def _region_air_fraction(
    #     obs: MinecraftObservation,
    #     *,
    #     y0_frac: float,
    #     y1_frac: float,
    #     x0_frac: float,
    #     x1_frac: float,
    # ) -> float:
    #     blocks = np.asarray(obs.fovBlocks, dtype=np.int32)
    #     if blocks.size != FOV_HEIGHT * FOV_WIDTH:
    #         raise RuntimeError("FOV size mismatch in _region_air_fraction")
    #     grid = blocks.reshape(FOV_HEIGHT, FOV_WIDTH)
    #
    #     y0 = int(np.clip(math.floor(FOV_HEIGHT * y0_frac), 0, FOV_HEIGHT))
    #     y1 = int(np.clip(math.ceil(FOV_HEIGHT * y1_frac), 0, FOV_HEIGHT))
    #     x0 = int(np.clip(math.floor(FOV_WIDTH * x0_frac), 0, FOV_WIDTH))
    #     x1 = int(np.clip(math.ceil(FOV_WIDTH * x1_frac), 0, FOV_WIDTH))
    #
    #     if y1 <= y0 or x1 <= x0:
    #         return 1.0
    #
    #     region = grid[y0:y1, x0:x1]
    #     air = int(BlockTypes.AIR)
    #     return float(np.mean(region == air))

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
