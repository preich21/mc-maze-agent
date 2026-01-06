"""PPO training entrypoint for the Minecraft WS environment.

- Uses ReachGoalRewardWrapper for task-specific reward/termination.
- Converts dict observations into a fixed float vector suitable for PPO.
- Auto-selects MPS on Apple Silicon, else CUDA/CPU.
- Configuration lives in variables below (no CLI flags).
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from env_mc import MinecraftEnv
from wrappers.simple_goal_reward import SimpleGoalRewardWrapper
from wrappers.action_flatten import ActionFlattenWrapper

# --------- Config ---------
URI = "ws://127.0.0.1:8081"
TOTAL_STEPS = 20_000
STEP_TICKS = 2
MAX_STEPS = 500
STEP_PENALTY = -0.001
GOAL_REWARD = 1.0
DEATH_PENALTY = -1.0
LOGDIR = "runs/ppo_minecraft"
LEARNING_RATE = 3e-4
BATCH_SIZE = 64
SEED = 42

# Observation vector layout: [x, y, z, yaw, pitch, standing_one_hot(4), fov_distances[25]]
FOV_RAYS = 25
STANDING_MAP: Dict[str, int] = {
    "AIR": 0,
    "BLOCK": 1,
    "START_BLOCK": 2,
    "GOAL_BLOCK": 3,
}


class ObservationVectorizer(gym.ObservationWrapper):
    """Convert dict obs from the mod into a fixed-size float vector."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # [x,y,z,yaw,pitch] + standing_one_hot(4) + fov_dist(25) + fov_type_one_hot(25*4)
        vec_len = 3 + 2 + len(STANDING_MAP) + FOV_RAYS + (FOV_RAYS * 4)
        low = np.full(vec_len, -np.inf, dtype=np.float32)
        high = np.full(vec_len, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):  # noqa: ANN001
        if not isinstance(observation, dict):
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        x = np.float32(observation.get("x", 0.0))
        y = np.float32(observation.get("y", 0.0))
        z = np.float32(observation.get("z", 0.0))
        yaw = np.float32(observation.get("yaw", 0.0))
        pitch = np.float32(observation.get("pitch", 0.0))

        standing = observation.get("standingOn", "AIR")
        standing_vec = np.zeros(len(STANDING_MAP), dtype=np.float32)
        standing_idx = STANDING_MAP.get(standing, 0)
        standing_vec[standing_idx] = 1.0

        fov_dist = np.asarray(observation.get("fovDistances", []), dtype=np.float16)[:FOV_RAYS]
        if fov_dist.shape[0] < FOV_RAYS:
            fov_dist = np.pad(fov_dist, (0, FOV_RAYS - fov_dist.shape[0]), constant_values=-1.0)

        fov_blocks = np.asarray(observation.get("fovBlocks", []), dtype=np.uint8)[:FOV_RAYS]
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


def select_device() -> str:
    # if torch.backends.mps.is_available():
    #     return "mps"
    # if torch.cuda.is_available():
    #     return "cuda"
    return "cpu"


def make_env() -> Callable[[], gym.Env]:
    def _init():
        env = MinecraftEnv(uri=URI, step_ticks=STEP_TICKS)
        env = SimpleGoalRewardWrapper(env)
        env.step_penalty = STEP_PENALTY
        env.goal_reward = GOAL_REWARD
        env.death_penalty = DEATH_PENALTY
        env.max_steps = MAX_STEPS
        env = ObservationVectorizer(env)
        env = ActionFlattenWrapper(env)
        return env

    return _init


def main() -> None:
    os.makedirs(LOGDIR, exist_ok=True)
    device = select_device()

    vec_env = DummyVecEnv([make_env()])
    vec_env.seed(SEED)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        device=device,
        verbose=1,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        tensorboard_log=LOGDIR,
        seed=SEED,
    )

    logger = configure(LOGDIR, ["stdout", "tensorboard"])
    model.set_logger(logger)

    model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True)
    model.save(os.path.join(LOGDIR, "ppo_minecraft_goal"))
    vec_env.close()


if __name__ == "__main__":
    main()
