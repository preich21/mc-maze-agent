import math

import gymnasium as gym
import numpy as np

from mc_env.env import FOV_RAYS, BlockTypes
from mc_env.observation import MinecraftObservation


class ObservationVectorizer(gym.ObservationWrapper[np.ndarray, np.ndarray, MinecraftObservation]):
    """Convert mod obs into a fixed-size float vector with sane scaling for off-policy RL (SAC/TD3)."""

    # KEEP CONSISTENT WITH MOD
    RAY_MAX = 50.0
    BFS_MAX = 5000.0

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # Layout sizes
        n_blocks = len(BlockTypes)
        n_surround = 8  # surrounding blocks

        low = np.array(
            [-1.0] * 3  # dx, dy, dz (normalized)
            + [-1.0, -1.0, -1.0, -1.0]  # cos/sin yaw/pitch
            + [0.0] * n_blocks  # standing on one-hot
            + [0.0] * (n_surround * n_blocks)  # surrounding blocks one-hot
            + [0.0] * FOV_RAYS  # fov distances normalized
            + [0.0] * (FOV_RAYS * n_blocks)  # fov block types one-hot
            + [0.0],  # bfs distance normalized
            dtype=np.float32,
        )
        high = np.array(
            [1.0] * 3  # dx, dy, dz (normalized)
            + [1.0, 1.0, 1.0, 1.0]  # cos/sin yaw/pitch
            + [1.0] * n_blocks  # standing on one-hot
            + [1.0] * (n_surround * n_blocks)  # surrounding blocks one-hot
            + [1.0] * FOV_RAYS  # fov distances normalized
            + [1.0] * (FOV_RAYS * n_blocks)  # fov block types one-hot
            + [1.0],  # bfs distance normalized
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self._n_blocks = n_blocks
        self._n_surround = n_surround

        # Pre-allocate identity for one-hot (tiny performance win)
        self._eye4 = np.eye(self._n_blocks, dtype=np.float32)

    @staticmethod
    def _safe_index(idx: int, size: int) -> int:
        return idx if 0 <= idx < size else 0

    @classmethod
    def _norm_bfs(cls, bfs: float) -> np.float32:
        # Clamp then log-normalize to [0,1]
        bfs = float(np.clip(bfs, 0.0, cls.BFS_MAX))
        return np.float32(np.log1p(bfs) / np.log1p(cls.BFS_MAX))

    @classmethod
    def _norm_ray_dist(cls, d: np.ndarray) -> np.ndarray:
        # Replace sentinel -1 with "max distance", then scale to [0,1]
        d = d.astype(np.float32, copy=False)
        d = np.where(d < 0.0, cls.RAY_MAX, d)
        d = np.clip(d, 0.0, cls.RAY_MAX)
        return d / np.float32(cls.RAY_MAX)

    def observation(self, observation: MinecraftObservation):
        dx = np.float32(observation.dx)
        dy = np.float32(observation.dy)
        dz = np.float32(observation.dz)

        dxyz = np.clip(np.array([dx, dy, dz], dtype=np.float32) / np.float32(0.5), -1.0, 1.0)

        # sin/cos in [-1,1]
        yaw = np.float32(observation.yaw)
        pitch = np.float32(observation.pitch)
        yaw_cos = np.float32(math.cos(yaw))
        yaw_sin = np.float32(math.sin(yaw))
        pitch_cos = np.float32(math.cos(pitch))
        pitch_sin = np.float32(math.sin(pitch))

        # standing block one-hot
        standing_vec = np.zeros(self._n_blocks, dtype=np.float32)
        standing_idx = self._safe_index(int(observation.standingOn), self._n_blocks)
        standing_vec[standing_idx] = 1.0

        # surrounding blocks one-hot (expect 8 neighbors; pad/trim to be stable)
        surrounding = list(observation.surroundingBlocks) if observation.surroundingBlocks is not None else []
        if len(surrounding) < self._n_surround:
            surrounding = surrounding + [0] * (self._n_surround - len(surrounding))
        else:
            surrounding = surrounding[: self._n_surround]

        surrounding_blocks_vec = np.zeros(self._n_blocks * self._n_surround, dtype=np.float32)
        for i, b in enumerate(surrounding):
            b_idx = self._safe_index(int(b), self._n_blocks)
            surrounding_blocks_vec[i * self._n_blocks + b_idx] = 1.0

        # FOV distances (normalize to [0,1], handle -1 sentinel)
        fov_dist = np.asarray(observation.fovDistances, dtype=np.float32)[:FOV_RAYS]
        if fov_dist.shape[0] < FOV_RAYS:
            fov_dist = np.pad(fov_dist, (0, FOV_RAYS - fov_dist.shape[0]), constant_values=-1.0)
        fov_dist = self._norm_ray_dist(fov_dist)

        # FOV block types one-hot (0..3)
        fov_blocks = np.asarray([int(b) for b in (observation.fovBlocks or [])], dtype=np.int32)[:FOV_RAYS]
        if fov_blocks.shape[0] < FOV_RAYS:
            fov_blocks = np.pad(fov_blocks, (0, FOV_RAYS - fov_blocks.shape[0]), constant_values=0)
        fov_blocks = np.clip(fov_blocks, 0, self._n_blocks - 1)
        fov_blocks_oh = self._eye4[fov_blocks].reshape(-1).astype(np.float32, copy=False)

        # BFS distance (log-normalize to [0,1])
        bfs_raw = observation.maze_distance if observation.maze_distance is not None else self.BFS_MAX
        bfs = np.array([self._norm_bfs(bfs_raw)], dtype=np.float32)

        vec = np.concatenate(
            [
                dxyz,
                np.array([yaw_cos, yaw_sin, pitch_cos, pitch_sin], dtype=np.float32),
                standing_vec,
                surrounding_blocks_vec,
                fov_dist.astype(np.float32, copy=False),
                fov_blocks_oh,
                bfs,
            ],
            dtype=np.float32,
        )
        return vec
