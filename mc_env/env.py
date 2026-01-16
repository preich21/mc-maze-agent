import enum
import logging
from typing import Tuple

import gymnasium as gym
import numpy as np

from mc_env.action import MinecraftAction
from mc_env.observation import MinecraftObservation
from mc_env.reset import ResetRequest
from mc_env.start_points import StartPoint
from ws.bridge import MinecraftWsBridge
from ws.messages import IncomingMessageType, HelloMessage

LOGGER = logging.getLogger(__name__)

class BlockTypes(enum.IntEnum):
    """Matches Java enum: AIR(0), START_BLOCK(2), BLOCK(1), GOAL_BLOCK(3)."""

    AIR = 0
    BLOCK = 1
    START_BLOCK = 2
    GOAL_BLOCK = 3

SOLID_BLOCKS = {BlockTypes.BLOCK, BlockTypes.START_BLOCK, BlockTypes.GOAL_BLOCK}

FOV_RAYS = 50 * 50

class MinecraftEnv(gym.Env[MinecraftObservation, np.ndarray]):
    metadata = {"render_modes": []}

    start_points: list[StartPoint] | None = None

    def __init__(self, uri: str = 'ws://127.0.0.1:8081',
                 step_ticks: int = 2,
                 yaw_delta_max_deg: float = 180.0,
                 pitch_delta_max_deg: float = 90.0,
                 curriculum_steps: int = None):
        super().__init__()

        self._ws = MinecraftWsBridge(uri, self.on_hello)

        self.step_ticks = step_ticks
        self.yaw_delta_max_deg = yaw_delta_max_deg
        self.pitch_delta_max_deg = pitch_delta_max_deg
        self.curriculum_steps = curriculum_steps
        if self.curriculum_steps is None:
            print("Curriculum steps not given, reset will use different start points with same probability")

        self.episode = 0
        self.step_idx = 0
        self.total_steps = 0

        # real obs space is defined in wrapper
        self.observation_space = gym.spaces.Space()
        self.action_space = MinecraftAction.get_space(self)

    def on_hello(self, message: HelloMessage):
        self.start_points = message.start_points

    def reset(self, seed=None, options=None) -> Tuple[MinecraftObservation, dict]:
        super().reset(seed=seed)
        self.episode += 1
        self.step_idx = 0

        start_point = self._choose_start_point()
        t = 1.0
        if self.curriculum_steps is not None:
            t = min(1.0, float(self.total_steps) / float(max(1, self.curriculum_steps)))
        self._randomize_pitch(start_point, t)
        self._randomize_yaw(start_point, t)
        request = ResetRequest(episode=self.episode, start_point=start_point, seed=seed, options=options)

        obs = self._ws.send(request, IncomingMessageType.STATE_AFTER_RESET)
        info = {"start_point": start_point}
        return obs, info

    def _choose_start_point(self) -> StartPoint | None:
        if not self.start_points:
            raise RuntimeError("start_points not initialized yet (no hello received)")
        if len(self.start_points) == 0:
            return None

        if self.curriculum_steps is not None:
            # Expect StartPoint to carry a numeric weight. Fallback to 1\.0 if missing.
            w0 = np.asarray([float(getattr(sp, "weight", 1.0)) for sp in self.start_points], dtype=np.float64)
            if w0.ndim != 1 or w0.size != len(self.start_points) or np.any(w0 <= 0):
                raise ValueError("all startPoint weights must be > 0")

            # Anneal from weighted (easy biased) -> uniform over time.
            uniform = np.ones_like(w0)
            t = min(1.0, float(self.total_steps) / float(max(1, self.curriculum_steps)))
            w = (1.0 - t) * w0 + t * uniform
            p = w / float(w.sum())

            start_point_id = int(self.np_random.choice(len(self.start_points), p=p))

        else:
            start_point_id = int(self.np_random.choice(len(self.start_points)))

        start_point = self.start_points[start_point_id]


        return start_point

    def _randomize_pitch(self, start_point: StartPoint, t: float) -> None:
        # Start easy: narrow pitch, then widen.
        easy_lo, easy_hi = -15.0, 15.0
        hard_lo, hard_hi = -90.0, 90.0

        pitch_lo = (1.0 - t) * easy_lo + t * hard_lo
        pitch_hi = (1.0 - t) * easy_hi + t * hard_hi

        pitch_delta = float(self.np_random.uniform(pitch_lo, pitch_hi))
        start_point.pitch = ((start_point.pitch + pitch_delta) % 180.0) - 90.0

    def _randomize_yaw(self, start_point: StartPoint, t: float) -> None:
        # Start easy: narrow yaw, then widen.
        easy_lo, easy_hi = -15.0, 15.0
        hard_lo, hard_hi = -180.0, 180.0

        yaw_lo = (1.0 - t) * easy_lo + t * hard_lo
        yaw_hi = (1.0 - t) * easy_hi + t * hard_hi

        yaw_delta = float(self.np_random.uniform(yaw_lo, yaw_hi))
        start_point.yaw = ((start_point.yaw + yaw_delta) % 360.0) - 180.0

    def step(self, action: np.ndarray) -> Tuple[MinecraftObservation, float, bool, bool, dict]:
        self.step_idx += 1
        self.total_steps += 1

        parsed_action = MinecraftAction.from_vector(action, env=self)

        obs = self._ws.send(parsed_action, IncomingMessageType.STATE_AFTER_ACTION)

        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def close(self):
        self._ws.close()
        super().close()
