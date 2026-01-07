"""PPO training entrypoint for the Minecraft WS environment.

- Uses ReachGoalRewardWrapper for task-specific reward/termination.
- Converts dict observations into a fixed float vector suitable for PPO.
- Auto-selects MPS on Apple Silicon, else CUDA/CPU.
- Configuration lives in variables below (no CLI flags).
"""
from __future__ import annotations

import os
from typing import Callable

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from mc_env.env import MinecraftEnv
from wrappers.observation_vectorizer import ObservationVectorizer
from wrappers.simple_goal_reward import SimpleGoalRewardWrapper

# --------- Config ---------
URI = "ws://127.0.0.1:8081"
TOTAL_STEPS = 20_000
N_STEPS = 2048
BATCH_SIZE = 64
STEP_TICKS = 2
MAX_STEPS = 500
LOGDIR = "runs/ppo_minecraft"
LEARNING_RATE = 3e-4
SEED = 42

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
        env.max_steps = MAX_STEPS
        env = ObservationVectorizer(env)
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
        n_steps=N_STEPS
    )

    logger = configure(LOGDIR, ["stdout", "tensorboard"])
    model.set_logger(logger)

    model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True)
    model.save(os.path.join(LOGDIR, "ppo_minecraft_goal"))

    # # Quick latency probe: how long does the policy need vs how long does env stepping take?
    # # Helps to tune STEP_TICKS based on real inference latency.
    # benchmark_decision_speed(model, vec_env, steps=200, deterministic=True)

    vec_env.close()


if __name__ == "__main__":
    main()
