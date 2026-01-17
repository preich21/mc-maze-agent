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
        self.goal_reward = 100.0
        self.death_penalty = -10.0

        # Exploration bonus reduced so it does not dominate learning direction
        self.new_block_reward = 0.02
        self.max_steps = 500

        self.goal_first_seen_bonus = 1.0
        self.goal_distance_weight = 0.2

        # Additional shaping
        self.air_step_penalty = -0.05  # per-step penalty when not on a solid block
        self.no_progress_penalty = -0.01  # per-step penalty if (bx,by,bz) unchanged
        self.goal_not_visible_penalty = -0.001  # optional tiny nudge to search/turn

        # --- Vision/safety shaping (human-like): "look where I'm going" ---
        # Penalize if the bottom of the view contains almost no ground information.
        self.bottom_air_frac_threshold = 0.90
        self.bottom_all_air_penalty = -0.01

        # Reward if the agent directs its view towards the ground ahead.
        # (Lower-middle region contains some non-AIR blocks.)
        self.look_ahead_solid_frac_threshold = 0.10
        self.look_ahead_ground_reward = 0.01

        # --- Movement shaping: reduce step penalty when moving forward while grounded ---
        # This does NOT directly reward selecting a "forward" action; it rewards
        # actual positive displacement in the facing direction while standing on a solid block.
        self.forward_step_penalty_reduction = 0.002  # max reduction per step
        self.forward_disp_clip = 0.25  # meters per step counted toward "forward"

        self.goal_seen: bool = False
        self.last_goal_distance: Optional[float] = None

        self._steps = 0
        self._visited_blocks: set[tuple[int, int, int]] = set()
        self._last_block_pos: Optional[tuple[int, int, int]] = None
        self._last_xz: Optional[tuple[float, float]] = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._steps = 0
        self._visited_blocks.clear()
        self.goal_seen = False
        self.last_goal_distance = None
        self._last_block_pos = None
        self._last_xz = None

        obs, info = self.env.reset(seed=seed, options=options)
        self._mark_visited_if_solid(obs)
        self._last_block_pos = self._block_pos(obs)
        self._last_xz = (float(obs.x), float(obs.z))
        return obs, info

    def step(self, action: np.ndarray):
        self._steps += 1
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        # Terminal conditions
        if obs.died:
            return obs, self.death_penalty, True, truncated, info
        if obs.standingOn == BlockTypes.GOAL_BLOCK:
            return obs, self.goal_reward, True, truncated, info

        reward = float(self.step_penalty)

        # Truncation
        if self._steps >= self.max_steps:
            return obs, reward, terminated, True, info

        grounded = obs.standingOn in SOLID_BLOCKS

        # ---- Additional shaping: discourage unsafe "air" steps ----
        if not grounded:
            reward += float(self.air_step_penalty)

        # ---- Additional shaping: encourage forward progress (displacement-based) ----
        reward += self._reward_for_forward_progress(obs, grounded=grounded)

        # ---- Additional shaping: encourage keeping relevant ground in view ----
        # These shape camera behavior without hard-coding a specific pitch.
        reward += self._reward_for_ground_visibility(obs)
        reward += self._reward_for_looking_ahead(obs)

        # ---- Additional shaping: discourage no movement / loops ----
        cur_pos = self._block_pos(obs)
        if self._last_block_pos is not None and cur_pos == self._last_block_pos:
            reward += float(self.no_progress_penalty)
        self._last_block_pos = cur_pos

        # ---- Goal visibility shaping ----
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
            # Optional tiny pressure to look for the goal
            reward += float(self.goal_not_visible_penalty)

        # ---- Exploration bonus: stepping onto new solid blocks ----
        if self._mark_visited_if_solid(obs):
            reward += float(self.new_block_reward)

        return obs, reward, terminated, truncated, info

    def _reward_for_forward_progress(self, obs: MinecraftObservation, *, grounded: bool) -> float:
        """Reduce the step penalty if the agent actually moves forward while grounded.

        This helps break left/right/back symmetry without rewarding suicidal "hold forward".
        """
        if not grounded:
            self._last_xz = (float(obs.x), float(obs.z))
            return 0.0

        if self._last_xz is None:
            self._last_xz = (float(obs.x), float(obs.z))
            return 0.0

        last_x, last_z = self._last_xz
        dx = float(obs.x) - float(last_x)
        dz = float(obs.z) - float(last_z)

        # Assumes yaw=0 faces +Z and yaw increases clockwise (MC-like).
        yaw_rad = math.radians(float(obs.yaw))
        fwd_x = math.sin(yaw_rad)
        fwd_z = math.cos(yaw_rad)

        forward_disp = dx * fwd_x + dz * fwd_z
        forward_disp = float(np.clip(forward_disp, -self.forward_disp_clip, self.forward_disp_clip))

        self._last_xz = (float(obs.x), float(obs.z))

        if forward_disp <= 0.0:
            return 0.0

        frac = forward_disp / float(self.forward_disp_clip)
        return float(self.forward_step_penalty_reduction) * float(frac)

    def _reward_for_ground_visibility(self, obs: MinecraftObservation) -> float:
        """Penalize if the bottom part of the view is mostly AIR."""
        air_frac = self._region_air_fraction(obs, y0_frac=0.75, y1_frac=1.00, x0_frac=0.00, x1_frac=1.00)
        if air_frac >= float(self.bottom_air_frac_threshold):
            return float(self.bottom_all_air_penalty)
        return 0.0

    def _reward_for_looking_ahead(self, obs: MinecraftObservation) -> float:
        """Reward if the agent looks at the ground *ahead* (not just straight down)."""
        air_frac = self._region_air_fraction(obs, y0_frac=0.60, y1_frac=0.85, x0_frac=0.30, x1_frac=0.70)
        solid_frac = 1.0 - air_frac
        if solid_frac >= float(self.look_ahead_solid_frac_threshold):
            return float(self.look_ahead_ground_reward)
        return 0.0

    @staticmethod
    def _region_air_fraction(
        obs: MinecraftObservation,
        *,
        y0_frac: float,
        y1_frac: float,
        x0_frac: float,
        x1_frac: float,
    ) -> float:
        """Return AIR fraction in a rectangular region of the FOV grid.

        Fractions are expressed in [0,1] over height/width; (0,0) is top-left.
        """
        blocks = np.asarray(obs.fovBlocks, dtype=np.int32)
        if blocks.size != FOV_HEIGHT * FOV_WIDTH:
            raise RuntimeError("FOV size mismatch in _region_air_fraction")

        grid = blocks.reshape(FOV_HEIGHT, FOV_WIDTH)

        y0 = int(np.clip(math.floor(FOV_HEIGHT * y0_frac), 0, FOV_HEIGHT))
        y1 = int(np.clip(math.ceil(FOV_HEIGHT * y1_frac), 0, FOV_HEIGHT))
        x0 = int(np.clip(math.floor(FOV_WIDTH * x0_frac), 0, FOV_WIDTH))
        x1 = int(np.clip(math.ceil(FOV_WIDTH * x1_frac), 0, FOV_WIDTH))

        if y1 <= y0 or x1 <= x0:
            return 1.0  # empty region => treat as all AIR (conservative)

        region = grid[y0:y1, x0:x1]
        air = int(BlockTypes.AIR)
        return float(np.mean(region == air))

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
