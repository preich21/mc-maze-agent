from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING, Any, Dict

import gymnasium as gym
import numpy as np

from ws.messages import OutgoingMessage

if TYPE_CHECKING:
    from mc_env.env import MinecraftEnv

FPS = 19

YAW_DELTA_MAX_DEG: float = 250.0 / FPS
PITCH_DELTA_MAX_DEG: float = 200.0 / FPS

@dataclass
class MinecraftAction(OutgoingMessage):
    moveForward: bool
    moveBackward: bool
    moveLeft: bool
    moveRight: bool
    jump: bool  # true to jump, false otherwise
    yawDelta: float  # horizontal rotation per tick
    pitchDelta: float  # vertical rotation per tick

    def to_message(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = "ACTION_REQUEST"
        return payload

    def to_vector(self) -> np.ndarray:
        """Reverse of from_vector(): MinecraftAction → 5D vector."""

        # moveForward/Backward → continuous [-1,1]
        if self.moveForward and not self.moveBackward:
            move_fwd = 1.0
        elif self.moveBackward and not self.moveForward:
            move_fwd = -1.0
        else:  # neither or both
            move_fwd = 0.0

        # moveLeft/Right → continuous [-1,1]
        if self.moveRight and not self.moveLeft:
            move_side = 1.0
        elif self.moveLeft and not self.moveRight:
            move_side = -1.0
        else:
            move_side = 0.0

        # jump → 0/1
        jump_prob = 1.0 if self.jump else 0.0

        # normalize deltas back to [-1,1]
        yaw_norm = float(self.yawDelta) / YAW_DELTA_MAX_DEG
        pitch_norm = float(self.pitchDelta) / PITCH_DELTA_MAX_DEG

        return np.array([
            move_fwd,
            move_side,
            jump_prob,
            np.clip(yaw_norm, -1.0, 1.0),
            np.clip(pitch_norm, -1.0, 1.0)
        ], dtype=np.float32)


    @staticmethod
    def from_vector(vector: np.ndarray) -> "MinecraftAction":
        if vector.shape[0] != 5:
            raise ValueError(f"Expected action vector length 5, got {vector.shape}")

        move_backward, move_forward = MinecraftAction.discrete_move_value_from_continuous(vector[0])
        move_left, move_right = MinecraftAction.discrete_move_value_from_continuous(vector[1])
        jump = bool(vector[2] >= 0.5)
        yaw_delta = float(vector[3]) * YAW_DELTA_MAX_DEG
        pitch_delta = float(vector[4]) * PITCH_DELTA_MAX_DEG

        return MinecraftAction(
            moveForward=move_forward,
            moveBackward=move_backward,
            moveLeft=move_left,
            moveRight=move_right,
            jump=jump,
            yawDelta=yaw_delta,
            pitchDelta=pitch_delta,
        )

    @staticmethod
    def discrete_move_value_from_continuous(value: float) -> tuple[bool, bool]:
        value = np.clip(float(value), -1.0, 1.0)
        bool1 = False
        bool2 = False
        if value < -0.1:
            bool1 = True
        elif value > 0.1:
            bool2 = True
        return bool1, bool2

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
        low = np.array([-1.0, -1.0, 0.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)
