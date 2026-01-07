import gymnasium as gym
import numpy as np

from mc_env.env import FOV_RAYS, BlockTypes
from mc_env.observation import MinecraftObservation


class ObservationVectorizer(gym.ObservationWrapper[np.ndarray, np.ndarray, MinecraftObservation]):
    """Convert dict obs from the mod into a fixed-size float vector."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # [x,y,z,yaw,pitch] + standing_one_hot(4) + fov_dist(25) + fov_type_one_hot(25*4)
        vec_len = 3 + 2 + len(BlockTypes) + FOV_RAYS + (FOV_RAYS * 4)
        low = np.full(vec_len, -np.inf, dtype=np.float32)
        high = np.full(vec_len, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation: MinecraftObservation):  # noqa: ANN001
        # observation is a MinecraftObservation dataclass
        x = np.float32(observation.x)
        y = np.float32(observation.y)
        z = np.float32(observation.z)
        yaw = np.float32(observation.yaw)
        pitch = np.float32(observation.pitch)

        standing_vec = np.zeros(len(BlockTypes), dtype=np.float32)
        standing_idx = int(observation.standingOn)
        if standing_idx < 0 or standing_idx >= len(BlockTypes):
            standing_idx = 0
        standing_vec[standing_idx] = 1.0

        fov_dist = np.asarray(observation.fovDistances, dtype=np.float16)[:FOV_RAYS]
        if fov_dist.shape[0] < FOV_RAYS:
            fov_dist = np.pad(fov_dist, (0, FOV_RAYS - fov_dist.shape[0]), constant_values=-1.0)

        fov_blocks = np.asarray([int(b) for b in observation.fovBlocks], dtype=np.uint8)[:FOV_RAYS]
        if fov_blocks.shape[0] < FOV_RAYS:
            fov_blocks = np.pad(fov_blocks, (0, FOV_RAYS - fov_blocks.shape[0]), constant_values=0)
        fov_blocks = np.clip(fov_blocks, 0, 3)
        fov_blocks_oh = np.eye(4, dtype=np.float32)[fov_blocks].reshape(-1)

        vec = np.concatenate([
            np.array([x, y, z, yaw, pitch], dtype=np.float32),
            standing_vec,
            fov_dist.astype(np.float32),
            fov_blocks_oh.astype(np.float32),
        ]).astype(np.float32)
        return vec