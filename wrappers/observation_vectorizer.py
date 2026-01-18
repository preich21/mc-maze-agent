import logging

import gymnasium as gym
import numpy as np

from mc_env.env import FOV_RAYS, BlockTypes, FOV_HEIGHT, FOV_WIDTH
from mc_env.observation import MinecraftObservation

POS_MIN, POS_MAX = -1000.0, 1000.0

LOGGER = logging.getLogger(__name__)

class ObservationVectorizer(gym.ObservationWrapper):
    """Convert MinecraftObservation into a Dict obs for MultiInputPolicy.

    Keys:
    - image: (C, H, W) float32, where C = len(BlockTypes) + 1
        - block one-hot: len(BlockTypes)
        - distance: 1
    - state: (D,) float32
        - x, y, z normalized
        - yaw, pitch normalized
        - standing-on one-hot: len(BlockTypes)

    This keeps the CNN focused on spatial structure and avoids wasting conv capacity on
    constant broadcast planes.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self._n_block = len(BlockTypes)

        # Image branch: block one-hot + distance
        img_channels = self._n_block + 1
        image_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(img_channels, FOV_HEIGHT, FOV_WIDTH),
            dtype=np.float32,
        )

        # State branch: [x,y,z,yaw,pitch] + standing one-hot
        state_dim = 8 + 5 + self._n_block
        state_low = np.full(state_dim, -1.1, dtype=np.float32)  # slight margin
        state_high = np.full(state_dim, 1.1, dtype=np.float32)
        state_space = gym.spaces.Box(
            low=state_low,
            high=state_high,
            shape=(state_dim,),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Dict({"image": image_space, "state": state_space})

    def observation(self, observation: MinecraftObservation):
        fov_dist = np.asarray(observation.fovDistances, dtype=np.float32)
        fov_blocks = np.asarray([int(b) for b in observation.fovBlocks], dtype=np.int32)

        if len(fov_dist) != FOV_RAYS or len(fov_blocks) != FOV_RAYS:
            raise RuntimeError("FOV rays length mismatch in ObservationVectorizer")

        # ----- image branch -----
        # distance normalized to [0,1]
        dist_norm = np.clip(fov_dist, 0.0, 50.0) / 50.0
        dist_grid = dist_norm.reshape(FOV_HEIGHT, FOV_WIDTH)

        # block one-hot: (n_block, H, W)
        blk = np.clip(fov_blocks, 0, self._n_block - 1)
        block_oh = np.eye(self._n_block, dtype=np.float32)[blk]  # (RAYS, n_block)
        block_oh = block_oh.reshape(FOV_HEIGHT, FOV_WIDTH, self._n_block)
        block_oh = np.transpose(block_oh, (2, 0, 1))

        image = np.concatenate([block_oh, dist_grid[None, :, :]], axis=0).astype(np.float32)

        # ----- state branch -----
        x_norm = np.clip((float(observation.x) - POS_MIN) / (POS_MAX - POS_MIN), 0.0, 1.0)
        z_norm = np.clip((float(observation.z) - POS_MIN) / (POS_MAX - POS_MIN), 0.0, 1.0)
        y_norm = np.clip(float(observation.y) / 10.0, 0.0, 1.0)

        yaw_rad = np.deg2rad(float(observation.yaw))  # degrees → radians
        yaw_sin = np.sin(yaw_rad)  # [-1,1]
        yaw_cos = np.cos(yaw_rad)  # [-1,1]
        pitch_norm = np.clip(float(observation.pitch) / 90.0, -1.0, 1.0)

        standing = np.zeros(self._n_block, dtype=np.float32)
        standing_idx = int(observation.standingOn)
        if 0 <= standing_idx < self._n_block:
            standing[standing_idx] = 1.0

        died = float(observation.died)

        if observation.actionStartedTick is not None:
            action_age = float(observation.tick - observation.actionStartedTick) / 20.0  # ticks → sec
        else:
            action_age = 0.0
        if action_age > 1.0:
            LOGGER.warn("Action age > 1.0 sec: %f", action_age)
        action_age_norm = np.clip(action_age, 0.0, 1.0)

        if observation.activeActionRequest is not None:
            prev_action_vec = observation.activeActionRequest.to_vector()
        else:
            prev_action_vec = np.zeros(5, dtype=np.float32)

        state = np.concatenate(
            [
                np.array([x_norm, y_norm, z_norm, yaw_sin, yaw_cos, pitch_norm, died, action_age_norm], dtype=np.float32),
                prev_action_vec,
                standing,
            ],
            axis=0,
        ).astype(np.float32)

        return {"image": image, "state": state}
