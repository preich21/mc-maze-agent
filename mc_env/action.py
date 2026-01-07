from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

if TYPE_CHECKING:
    from mc_env.env import MinecraftEnv


@dataclass
class MinecraftActionVector:
    """Flat vector used by RL agents (5 dims):
    [moveForward, moveSidewards, jump, yawDelta, pitchDelta]
    - moveForward: -1..1 (negative = back)
    - moveSidewards: -1..1 (negative = left)
    - jump: 0 or 1 (>=0.5 treated as True)
    - yawDelta: degrees per tick (will be applied for env.step_ticks ticks)
    - pitchDelta: degrees per tick (clamped to [-90,90] in MC)
    """

    vec: np.ndarray

    @staticmethod
    def get_space(env: "MinecraftEnv") -> gym.spaces.Space:
        # Allow full per-tick rotation up to configured maxima.
        yaw_max_per_tick = float(env.yaw_delta_max_deg / env.step_ticks)
        pitch_max_per_tick = float(env.pitch_delta_max_deg / env.step_ticks)
        low = np.array([-1.0, -1.0, 0.0, -yaw_max_per_tick, -pitch_max_per_tick], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, yaw_max_per_tick, pitch_max_per_tick], dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    # @staticmethod
    # def from_array(arr: np.ndarray) -> "MinecraftActionVector":
    #     return MinecraftActionVector(vec=np.asarray(arr, dtype=np.float32).reshape(-1))

    def as_array(self) -> np.ndarray:
        return np.asarray(self.vec, dtype=np.float32).reshape(-1)


@dataclass
class MinecraftAction:
    episode: int
    step: int
    applyForTicks: int  # number of ticks to apply this action for == env.step_ticks
    moveForward: float  # -1.0 back, 0.0 none, 1.0 forward
    moveSidewards: float  # -1.0 left, 0.0 none, 1.0 right
    jump: bool  # true to jump, false otherwise
    yawDelta: float  # horizontal rotation per tick
    pitchDelta: float  # vertical rotation per tick

    @staticmethod
    def from_vector(vector: MinecraftActionVector, env: "MinecraftEnv") -> "MinecraftAction":
        arr = vector.as_array()
        if arr.shape[0] != 5:
            raise ValueError(f"Expected action vector length 5, got {arr.shape}")

        yaw_max = float(env.yaw_delta_max_deg / env.step_ticks)
        pitch_max = float(env.pitch_delta_max_deg / env.step_ticks)

        move_forward = float(np.clip(arr[0], -1.0, 1.0))
        move_sidewards = float(np.clip(arr[1], -1.0, 1.0))
        jump = bool(arr[2] >= 0.5)
        yaw_delta = float(np.clip(arr[3], -yaw_max, yaw_max))
        pitch_delta = float(np.clip(arr[4], -pitch_max, pitch_max))

        return MinecraftAction(
            episode=env.episode,
            step=env.step_idx + 1,
            applyForTicks=env.step_ticks,
            moveForward=move_forward,
            moveSidewards=move_sidewards,
            jump=jump,
            yawDelta=yaw_delta,
            pitchDelta=pitch_delta,
        )

    # def to_vector(self) -> MinecraftActionVector:
    #     return MinecraftActionVector(
    #         vec=np.array(
    #             [
    #                 self.moveForward,
    #                 self.moveSidewards,
    #                 1.0 if self.jump else 0.0,
    #                 self.yawDelta,
    #                 self.pitchDelta,
    #             ],
    #             dtype=np.float32,
    #         )
    #     )
