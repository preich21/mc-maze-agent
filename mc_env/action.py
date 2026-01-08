from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING, Any, Dict

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
        if vector.shape[0] != 5:
            raise ValueError(f"Expected action vector length 5, got {vector.shape}")

        yaw_max = float(env.yaw_delta_max_deg / env.step_ticks)
        pitch_max = float(env.pitch_delta_max_deg / env.step_ticks)

        move_forward = float(vector[0] - 1.0)  # Convert 0,1,2 to -1.0,0.0,1.0
        move_sidewards = float(vector[1] - 1.0) # Convert 0,1,2 to -1.0,0.0,1.0
        jump = bool(vector[2])
        yaw_delta = float(np.clip((vector[3] - 10) / 10 * yaw_max, -yaw_max, yaw_max))
        pitch_delta = float(np.clip((vector[4] - 10) / 10 * pitch_max, -pitch_max, pitch_max))

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
        """Flat vector used by RL agents (5 dims):
            [moveForward, moveSidewards, jump, yawDelta, pitchDelta]
            - moveForward: -1..1 (negative = back)
            - moveSidewards: -1..1 (negative = left)
            - jump: 0 or 1 (>=0.5 treated as True)
            - yawDelta: degrees per tick (will be applied for env.step_ticks ticks)
            - pitchDelta: degrees per tick (clamped to [-90,90] in MC)
            """
        # Allow full per-tick rotation up to configured maxima.
        # yaw_max_per_tick = float(env.yaw_delta_max_deg / env.step_ticks)
        # pitch_max_per_tick = float(env.pitch_delta_max_deg / env.step_ticks)
        # low = np.array([-1.0, -1.0, 0.0, -yaw_max_per_tick, -pitch_max_per_tick], dtype=np.float32)
        # high = np.array([1.0, 1.0, 1.0, yaw_max_per_tick, pitch_max_per_tick], dtype=np.float32)
        # return gym.spaces.Box(low=low, high=high, dtype=np.float32)
        return gym.spaces.MultiDiscrete([
            3,  # moveForward: 0=back, 1=none, 2=forward
            3,  # moveSidewards: 0=left, 1=none, 2=right
            2,  # jump: 0=false, 1=true
            21,  # yawDelta
            21   # pitchDelta
        ])

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
