from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np

from ws.messages import OutgoingMessage

if TYPE_CHECKING:
    from mc_env.env import MinecraftEnv


@dataclass
class MinecraftAction(OutgoingMessage):
    episode: int
    step: int
    applyForTicks: int  # number of ticks to apply this action for == env.step_ticks
    moveForward: float  # -1.0 back, 0.0 none, 1.0 forward
    moveSidewards: float  # -1.0 left, 0.0 none, 1.0 right
    jump: bool  # true to jump, false otherwise
    yawDelta: float  # horizontal rotation per tick
    pitchDelta: float  # vertical rotation per tick

    def to_message(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = "ACTION_REQUEST"
        return payload

    @staticmethod
    def from_vector(vector: np.ndarray, env: "MinecraftEnv") -> "MinecraftAction":
        """
        Expect normalized agent actions in [-1, 1]:
          [moveForward, moveSidewards, jump, yawNorm, pitchNorm]
        where yawNorm/pitchNorm are scaled to per-tick degrees internally.
        """
        if vector.shape[0] != 5:
            raise ValueError(f"Expected action vector length 5, got {vector.shape}")

        # Normalize inputs defensively
        v = np.asarray(vector, dtype=np.float32)
        v = np.clip(v, -1.0, 1.0)

        # Movement stays in [-1, 1]
        move_forward = float(v[0])
        move_sidewards = float(v[1])

        # Jump: threshold the continuous output
        jump = bool(v[2] >= 0.5)

        # Scale normalized yaw/pitch -> degrees per tick
        yaw_max_per_tick = float(env.yaw_delta_max_deg / env.step_ticks)
        pitch_max_per_tick = float(env.pitch_delta_max_deg / env.step_ticks)

        yaw_delta = float(v[3] * yaw_max_per_tick)
        pitch_delta = float(v[4] * pitch_max_per_tick)

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

    @staticmethod
    def get_space(env: "MinecraftEnv") -> gym.spaces.Space:
        """
        Normalized continuous action space for SAC/TD3 (5 dims), all in [-1, 1].
        yaw/pitch are scaled in from_vector() using env yaw/pitch maxima.
        """
        low = np.array([-1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)
