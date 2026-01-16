import gymnasium as gym
import numpy as np

from mc_env.env import FOV_RAYS, BlockTypes, FOV_HEIGHT, FOV_WIDTH
from mc_env.observation import MinecraftObservation

POS_MIN, POS_MAX = -1000.0, 1000.0

class ObservationVectorizer(gym.ObservationWrapper):
    """
    Convert MinecraftObservation into CNN-friendly tensor (C, H, W=50x50),
    embedding all previous features as constant channels for CnnPolicy.

    Channels:
    0: normalized block type (rays)
    1: normalized distance (rays)
    2: normalized yaw
    3: normalized pitch
    4: normalized x
    5: normalized y
    6: normalized z
    7+: standing-on one-hot channels (broadcast over 50x50 grid)
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        n_channels = 2 + 5 + len(BlockTypes)  # rays + yaw/pitch/xyz + standing one-hot
        low = np.zeros((n_channels, FOV_HEIGHT, FOV_WIDTH), dtype=np.float32)
        high = np.ones((n_channels, FOV_HEIGHT, FOV_WIDTH), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def observation(self, observation: MinecraftObservation):  # noqa: ANN001
        fov_dist = np.asarray(observation.fovDistances, dtype=np.float32)
        fov_blocks = np.asarray([int(b) for b in observation.fovBlocks], dtype=np.int32)

        if len(fov_dist) != FOV_RAYS or len(fov_blocks) != FOV_RAYS:
            raise RuntimeError("FOV rays length mismatch in ObservationVectorizer")

        # ----- normalize rays -----
        dist_norm = np.clip(fov_dist, 0.0, 20.0) / 20.0  # [0,1]
        max_block_id = max(1, len(BlockTypes) - 1)
        blocks_norm = np.clip(fov_blocks, 0, max_block_id) / float(max_block_id)

        # ----- position / rotation -----
        # x,z: [-1000,1000] -> [0,1]
        x_norm = np.clip((float(observation.x) - POS_MIN) / (POS_MAX - POS_MIN), 0.0, 1.0)
        z_norm = np.clip((float(observation.z) - POS_MIN) / (POS_MAX - POS_MIN), 0.0, 1.0)
        # y: [0,10] -> [0,1]
        y_norm = np.clip(float(observation.y) / 10.0, 0.0, 1.0)

        # yaw/pitch: [-180,180]/[−90,90] -> [0,1]
        yaw_norm = np.clip(float(observation.yaw) / 180.0, -1.0, 1.0)
        yaw_norm = (yaw_norm + 1.0) / 2.0
        pitch_norm = np.clip(float(observation.pitch) / 90.0, -1.0, 1.0)
        pitch_norm = (pitch_norm + 1.0) / 2.0

        # ----- standing-on one-hot -----
        standing_vec = np.zeros(len(BlockTypes), dtype=np.float32)
        standing_idx = int(observation.standingOn)
        if 0 <= standing_idx < len(BlockTypes):
            standing_vec[standing_idx] = 1.0

        # ----- create constant channels (50,50) -----
        yaw_channel = np.full((FOV_HEIGHT, FOV_WIDTH), yaw_norm, dtype=np.float32)
        pitch_channel = np.full((FOV_HEIGHT, FOV_WIDTH), pitch_norm, dtype=np.float32)
        x_channel = np.full((FOV_HEIGHT, FOV_WIDTH), x_norm, dtype=np.float32)
        y_channel = np.full((FOV_HEIGHT, FOV_WIDTH), y_norm, dtype=np.float32)
        z_channel = np.full((FOV_HEIGHT, FOV_WIDTH), z_norm, dtype=np.float32)

        standing_channels = [
            np.full((FOV_HEIGHT, FOV_WIDTH), val, dtype=np.float32)
            for val in standing_vec
        ]

        # ----- reshape ray channels to 2D -----
        dist_grid = dist_norm.reshape(FOV_HEIGHT, FOV_WIDTH)
        block_grid = blocks_norm.reshape(FOV_HEIGHT, FOV_WIDTH)

        # ----- stack all: (11, 50, 50) -----
        grid = np.stack(
            [block_grid, dist_grid, yaw_channel, pitch_channel,
             x_channel, y_channel, z_channel, *standing_channels],
            axis=0,
        ).astype(np.float32)

        return grid
