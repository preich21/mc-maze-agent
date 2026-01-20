import math

import gymnasium as gym
import numpy as np

from mc_env.env import FOV_RAYS, BlockTypes
from mc_env.observation import MinecraftObservation


class ObservationVectorizer(gym.ObservationWrapper[np.ndarray, np.ndarray, MinecraftObservation]):
    """Convert dict obs from the mod into a fixed-size float vector."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        low = np.array(
            [-0.5] * 3 # dx, dy, dz
            + [-1, -1] # cos(yaw), sin(yaw)
            + [0] * len(BlockTypes) # standing on one-hot
            + [0] * (8 * len(BlockTypes)) # surrounding blocks one-hot
            + [-1.0] * FOV_RAYS # fov distances
            + [0] * (FOV_RAYS * 4) # fov block types one-hot
            + [0], # bfs distance
            dtype=np.float32,
        )
        high = np.array(
            [0.5] * 3 # dx, dy, dz
            + [1, 1] # cos(yaw), sin(yaw)
            + [1] * len(BlockTypes) # standing on one-hot
            + [1] * (8 * len(BlockTypes)) # surrounding blocks one-hot
            + [50.0] * FOV_RAYS # fov distances
            + [1] * (FOV_RAYS * 4) # fov block types one-hot
            + [5_000.0], # bfs distance
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation: MinecraftObservation):  # noqa: ANN001
        # observation is a MinecraftObservation dataclass
        dx = np.float32(observation.dx)
        dy = np.float32(observation.dy)
        dz = np.float32(observation.dz)
        yaw_cos = math.cos(np.float32(observation.yaw))
        yaw_sin = math.sin(np.float32(observation.yaw))

        standing_vec = np.zeros(len(BlockTypes), dtype=np.float32)
        standing_idx = int(observation.standingOn)
        if standing_idx < 0 or standing_idx >= len(BlockTypes):
            standing_idx = 0
        standing_vec[standing_idx] = 1.0

        surrounding_blocks_vec = np.zeros(len(BlockTypes) * len(observation.surroundingBlocks), dtype=np.float32)
        for i, b in enumerate(observation.surroundingBlocks):
            b_idx = int(b)
            if b_idx < 0 or b_idx >= len(BlockTypes):
                b_idx = 0
            surrounding_blocks_vec[i * len(BlockTypes) + b_idx] = 1.0

        fov_dist = np.asarray(observation.fovDistances, dtype=np.float16)[:FOV_RAYS]
        if fov_dist.shape[0] < FOV_RAYS:
            fov_dist = np.pad(fov_dist, (0, FOV_RAYS - fov_dist.shape[0]), constant_values=50.0)

        fov_blocks = np.asarray([int(b) for b in observation.fovBlocks], dtype=np.uint8)[:FOV_RAYS]
        if fov_blocks.shape[0] < FOV_RAYS:
            fov_blocks = np.pad(fov_blocks, (0, FOV_RAYS - fov_blocks.shape[0]), constant_values=0)
        fov_blocks = np.clip(fov_blocks, 0, 3)
        fov_blocks_oh = np.eye(4, dtype=np.float32)[fov_blocks].reshape(-1)

        vec = np.concatenate([
            np.array([dx, dy, dz, yaw_cos, yaw_sin], dtype=np.float32),
            standing_vec,
            surrounding_blocks_vec,
            fov_dist.astype(np.float32),
            fov_blocks_oh.astype(np.float32),
            np.array([observation.maze_distance if observation.maze_distance is not None else 5000.0], dtype=np.float32),
        ]).astype(np.float32)
        return vec