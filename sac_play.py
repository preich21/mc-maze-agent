"""
Run a trained SAC policy in the live Minecraft environment.

Usage:
    python play_sac.py --env maze --model-path runs/sac_minecraft/sac_minecraft_goal.zip
or
    python play_sac.py --env maze --model 1
"""
from __future__ import annotations

import argparse
import time

from stable_baselines3 import SAC

from mc_env.env import MinecraftEnv
from wrappers.observation_vectorizer import ObservationVectorizer
from wrappers.simple_goal_reward import SimpleGoalRewardWrapper
from wrappers.maze_exploring_reward import MazeExploringRewardWrapper

# Keep runtime config local to avoid accidental mismatch.
URI = "ws://127.0.0.1:8081"
STEP_TICKS = 2
MAX_STEPS = 500

MODEL_BASE_PATH = "models/"
N_EPISODES = 50
DETERMINISTIC = True
SLEEP_BETWEEN_STEPS_SEC = 0.0  # set >0 for slower visible playback

ARG_PARSER = argparse.ArgumentParser(description="Minecraft Agent: Choose your environment and model.")
ARG_PARSER.add_argument("--env", choices=["simple", "maze"], required=True)
ARG_PARSER.add_argument("--model", choices=["1", "2"], required=False)
ARG_PARSER.add_argument("--model-path", type=str, required=False)


def make_simple_env():
    env = MinecraftEnv(uri=URI, step_ticks=STEP_TICKS)
    env = SimpleGoalRewardWrapper(env)
    env.max_steps = MAX_STEPS
    env = ObservationVectorizer(env)
    return env


def make_maze_env():
    env = MinecraftEnv(uri=URI, step_ticks=STEP_TICKS)
    env = MazeExploringRewardWrapper(env)
    env.max_steps = MAX_STEPS
    env = ObservationVectorizer(env)
    return env


def make_env():
    args = ARG_PARSER.parse_args()
    if args.env == "maze":
        return make_maze_env()
    if args.env == "simple":
        return make_simple_env()
    raise ValueError("Environment type must be specified with --env")


def load_model():
    args = ARG_PARSER.parse_args()

    if args.model_path is not None:
        return SAC.load(args.model_path)

    if args.model == "1":
        return SAC.load(MODEL_BASE_PATH + "1_shy_model/model.zip")
    if args.model == "2":
        return SAC.load(MODEL_BASE_PATH + "2_simple_world_1_first_working_model/model.zip")

    raise ValueError("Model must be specified with --model or --model-path")


def main() -> None:
    env = make_env()
    try:
        model = load_model()
        successes = 0

        for ep in range(1, N_EPISODES + 1):
            obs, info = env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=DETERMINISTIC)
                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += float(reward)
                steps += 1

                if SLEEP_BETWEEN_STEPS_SEC > 0:
                    time.sleep(SLEEP_BETWEEN_STEPS_SEC)

            if terminated:
                successes += 1

            print(
                f"Episode {ep}: steps={steps} total_reward={total_reward:.3f} "
                f"terminated={terminated} truncated={truncated} | "
                f"successrate so far = {successes}/{ep} ({(successes / ep) * 100:.2f}%)"
            )

        print(f"Success rate: {successes}/{N_EPISODES} ({(successes / N_EPISODES) * 100:.2f}%)")
    finally:
        env.close()


if __name__ == "__main__":
    main()
