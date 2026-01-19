"""
SAC training entrypoint for the Minecraft WS environment.

- Off-policy, more sample-efficient than PPO for continuous control.
- Uses replay buffer; good for slow environments.
- Configuration lives in variables below (no CLI flags).
"""
from __future__ import annotations

import argparse
import os
from typing import Callable

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from mc_env.env import MinecraftEnv
from wrappers.observation_vectorizer import ObservationVectorizer
from wrappers.simple_goal_reward import SimpleGoalRewardWrapper
from wrappers.maze_exploring_reward import MazeExploringRewardWrapper

# --------- Config ---------
URI = "ws://127.0.0.1:8081"
TOTAL_STEPS = 400_000

STEP_TICKS = 2
MAX_STEPS = 500

LOGDIR = "runs/sac_minecraft"
MODEL_NAME = "sac_minecraft_goal"

LEARNING_RATE = 3e-4
SEED = 42

# SAC-specific
BUFFER_SIZE = 200_000
LEARNING_STARTS = 1_000
BATCH_SIZE = 256
TRAIN_FREQ = 1          # train every env step
GRADIENT_STEPS = 1      # one gradient update per train step
TAU = 0.005
GAMMA = 0.99

def select_device() -> str:
    # if torch.backends.mps.is_available():
    #     return "mps"
    # if torch.cuda.is_available():
    #     return "cuda"
    return "cpu"

def make_simple_env() -> Callable[[], gym.Env]:
    def _init():
        env = MinecraftEnv(uri=URI, step_ticks=STEP_TICKS)
        env = SimpleGoalRewardWrapper(env)
        env.max_steps = MAX_STEPS
        env = ObservationVectorizer(env)
        return env
    return _init

def make_maze_env() -> Callable[[], gym.Env]:
    def _init():
        env = MinecraftEnv(uri=URI, step_ticks=STEP_TICKS)
        env = MazeExploringRewardWrapper(env)
        env.max_steps = MAX_STEPS
        env = ObservationVectorizer(env)
        return env
    return _init

def make_env() -> Callable[[], gym.Env]:
    parser = argparse.ArgumentParser(description="Minecraft Agent Training: Choose your environment.")
    parser.add_argument("--env", choices=["simple", "maze"])
    env_type = parser.parse_args().env

    if env_type == "simple":
        return make_simple_env()
    elif env_type == "maze":
        return make_maze_env()
    else:
        raise ValueError("Environment type must be specified with --env")

def main() -> None:
    os.makedirs(LOGDIR, exist_ok=True)
    device = select_device()

    vec_env = DummyVecEnv([make_env()])
    vec_env.seed(SEED)

    logger = configure(LOGDIR, ["stdout", "tensorboard"])

    model_path = os.path.join(LOGDIR, f"{MODEL_NAME}.zip")

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = SAC.load(model_path, env=vec_env, device=device)
        model.set_logger(logger)
        model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True, reset_num_timesteps=False)
    else:
        print("No existing model found, creating a new one.")
        model = SAC(
            policy="MlpPolicy",
            env=vec_env,
            device=device,
            verbose=1,
            learning_rate=LEARNING_RATE,
            buffer_size=BUFFER_SIZE,
            learning_starts=LEARNING_STARTS,
            batch_size=BATCH_SIZE,
            train_freq=TRAIN_FREQ,
            gradient_steps=GRADIENT_STEPS,
            tau=TAU,
            gamma=GAMMA,
            ent_coef="auto",
            tensorboard_log=LOGDIR,
            seed=SEED,
        )
        model.set_logger(logger)
        model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True)

    model.save(os.path.join(LOGDIR, MODEL_NAME))
    vec_env.close()

if __name__ == "__main__":
    main()
