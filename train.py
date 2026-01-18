"""PPO training entrypoint for the Minecraft WS environment.

- Uses task-specific reward/termination wrapper.
- Converts observations into a Dict suitable for MultiInputPolicy (CNN+MLP).
- Configuration lives in variables below; training length can be selected via --time.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from callbacks.debug_tensorboard import DebugMetricsTensorboardCallback
from mc_env.env import MinecraftEnv
from wrappers.debug_observation import DebugMinecraftObsWrapper
from wrappers.observation_vectorizer import ObservationVectorizer
from wrappers.simple_goal_reward import SimpleGoalRewardWrapper
from wrappers.maze_exploring_reward import MazeExploringRewardWrapper


# --------- Static config ---------
URI = "ws://127.0.0.1:8081"
LOGDIR = "models/current_run"
SEED = 42


@dataclass(frozen=True)
class TrainConfig:
    # Environment selection
    env_type: str  # "simple" | "maze"

    # Training length preset
    time_preset: str  # "short" | "mid" | "long"

    # PPO params
    total_steps: int
    n_steps: int
    batch_size: int
    n_epochs: int
    learning_rate: float

    # Env params
    max_steps_per_episode: int
    curriculum_steps: int | None


def set_meta_params(argv: list[str] | None = None) -> TrainConfig:
    """Parse CLI args and return a complete training config.

    Presets:
    - short: fastest possible "does it run" sanity check.
    - mid: short-but-meaningful learning signal.
    - long: more realistic training budget for harder tasks.
    """
    parser = argparse.ArgumentParser(description="Minecraft Agent Training")
    parser.add_argument("--env", choices=["simple", "maze"], required=True)
    parser.add_argument("--time", choices=["short", "mid", "long"], default="mid")
    args = parser.parse_args(argv)

    env_type: str = args.env
    time_preset: str = args.time

    if time_preset == "short":
        # Smoke test: minimal time, just to validate end-to-end pipeline.
        total_steps = 2_048 * 4
        curriculum_steps = max(1, int(total_steps * 0.5))
        n_steps = 256
        batch_size = 64
        n_epochs = 2
        learning_rate = 3e-4

        max_steps_per_episode = 100


    elif time_preset == "mid":
        # Minimal meaningful learning: enough for reward curves to move.
        total_steps = 80_000
        curriculum_steps = max(1, int(total_steps * 0.5))
        n_steps = 1_024
        batch_size = 64
        n_epochs = 5
        learning_rate = 3e-4

        max_steps_per_episode = 250


    elif time_preset == "long":
        # Realistic training budget. Adjust based on your setup speed.
        total_steps = 1_000_000
        curriculum_steps = max(1, int(total_steps * 0.5))
        n_steps = 2_048
        batch_size = 128
        n_epochs = 10
        learning_rate = 3e-4

        max_steps_per_episode = 500

    else:
        raise ValueError(f"Unknown time preset: {time_preset}")

    # PPO constraints: batch_size must divide (n_envs * n_steps). n_envs=1 here.
    if n_steps % batch_size != 0:
        # make it valid by shrinking batch size to the largest divisor <= current
        divisors = [d for d in (64, 32, 16, 8, 4, 2, 1) if n_steps % d == 0]
        batch_size = divisors[0] if divisors else 1

    return TrainConfig(
        env_type=env_type,
        time_preset=time_preset,
        total_steps=total_steps,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        max_steps_per_episode=max_steps_per_episode,
        curriculum_steps=curriculum_steps,
    )


def select_device() -> str:
    # if torch.backends.mps.is_available():
    #     return "mps"
    # if torch.cuda.is_available():
    #     return "cuda"
    return "cpu"


def make_env(cfg: TrainConfig) -> Callable[[], gym.Env]:
    def _init():
        env = MinecraftEnv(uri=URI, curriculum_steps=cfg.curriculum_steps)

        if cfg.env_type == "simple":
            env = SimpleGoalRewardWrapper(env)
        elif cfg.env_type == "maze":
            env = MazeExploringRewardWrapper(env)
        else:
            raise ValueError(f"Unknown env type: {cfg.env_type}")

        # Both reward wrappers expose max_steps as an attribute.
        env.max_steps = cfg.max_steps_per_episode
        env = DebugMinecraftObsWrapper(env)
        env = ObservationVectorizer(env)
        return env

    return _init


def main() -> None:
    cfg = set_meta_params()

    os.makedirs(LOGDIR, exist_ok=True)
    device = select_device()

    vec_env = DummyVecEnv([make_env(cfg)])
    vec_env.seed(SEED)

    model = PPO(
        policy="MultiInputPolicy",
        policy_kwargs={
            "normalize_images": False,
            # Larger initial exploration in continuous action dims
            "log_std_init": -0.5,  # try -0.5 or 0.0 (bigger == noisier)
        },
        env=vec_env,
        device=device,
        verbose=1,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        tensorboard_log=LOGDIR,
        seed=SEED,
        n_steps=cfg.n_steps,
        n_epochs=cfg.n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.10,
        vf_coef=0.5,
    )

    logger = configure(LOGDIR, ["stdout", "tensorboard"])
    model.set_logger(logger)

    callback = CallbackList([DebugMetricsTensorboardCallback()])
    model.learn(total_timesteps=cfg.total_steps, progress_bar=True, callback=callback)
    model.save(os.path.join(LOGDIR, f"ppo_minecraft_{cfg.env_type}_{cfg.time_preset}"))

    vec_env.close()


if __name__ == "__main__":
    main()
