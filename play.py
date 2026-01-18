"""Run a trained PPO policy in the live Minecraft environment.

This script loads the most recent model saved by train.py and uses it to
control the agent in Minecraft.

Prereqs:
- Minecraft + your mod running (WebSocket server up)
- A saved model exists at runs/ppo_minecraft/ppo_minecraft_goal.zip

Usage:
    python play.py

Stop with Ctrl+C.
"""
from __future__ import annotations

import argparse
import time

from stable_baselines3 import PPO

from mc_env.env import MinecraftEnv
from wrappers.observation_vectorizer import ObservationVectorizer
from wrappers.simple_goal_reward import SimpleGoalRewardWrapper
from wrappers.maze_exploring_reward import MazeExploringRewardWrapper

# Keep runtime config local to play.py to avoid accidental mismatch.
URI = "ws://127.0.0.1:8081"
STEP_TICKS = 2
MAX_STEPS = 500

MODEL_BASE_PATH = "models/"
N_EPISODES = 20
DETERMINISTIC = True
SLEEP_BETWEEN_STEPS_SEC = 0.0  # set >0 for slower visible playback

ARG_PARSER = argparse.ArgumentParser(description="Minecraft Agent: Choose your environment and model.")
ARG_PARSER.add_argument(
    "--model",
    choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]
)
ARG_PARSER.add_argument(
    "--env",
    choices=["simple", "maze"]
)

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
    env_type = ARG_PARSER.parse_args().env
    if env_type == "maze":
        return make_maze_env()
    elif env_type == "simple":
        return make_simple_env()
    else:
        raise ValueError("Environment type must be specified with --env")


def load_model():
    model = ARG_PARSER.parse_args().model
    if model == "1":
        return PPO.load(MODEL_BASE_PATH + "1_shy_model/model.zip")
    elif model == "2":
        return PPO.load(MODEL_BASE_PATH + "2_simple_world_1_first_working_model/model.zip")
    elif model == "3":
        return PPO.load(MODEL_BASE_PATH + "3_first_multi_start_point_model/model.zip")
    elif model == "4":
        return PPO.load(MODEL_BASE_PATH + "4_long_training/model.zip")
    elif model == "5":
        return PPO.load(MODEL_BASE_PATH + "5_more_pitch_and_yaw/model.zip")
    elif model == "6":
        return PPO.load(MODEL_BASE_PATH + "6_first_cnn/model.zip")
    elif model == "7":
        return PPO.load(MODEL_BASE_PATH + "7_improved_cnn/model.zip")
    elif model == "8":
        return PPO.load(MODEL_BASE_PATH + "8_/model.zip")
    elif model == "9":
        return PPO.load(MODEL_BASE_PATH + "9_/model.zip")
    elif model == "10":
        return PPO.load(MODEL_BASE_PATH + "10_/model.zip")
    elif model == "11":
        return PPO.load(MODEL_BASE_PATH + "11_move_action_space_fixed/model.zip")
    elif model == "12":
        return PPO.load(MODEL_BASE_PATH + "12_same_but_20k_steps_only/model.zip")
    elif model == "13":
        return PPO.load(MODEL_BASE_PATH + "13_20k_reduce_laziness/model.zip")
    else:
        raise ValueError("Model must be specified with --model")


def main() -> None:
    env = make_env()
    try:
        model = load_model()

        for ep in range(1, N_EPISODES + 1):
            obs, info = env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=DETERMINISTIC)
                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += float(reward)
                steps += 1
                done = bool(terminated or truncated)

                if SLEEP_BETWEEN_STEPS_SEC > 0:
                    time.sleep(SLEEP_BETWEEN_STEPS_SEC)

            print(f"Episode {ep}: steps={steps} total_reward={total_reward:.3f} terminated={terminated} truncated={truncated}")
    finally:
        env.close()

if __name__ == "__main__":
    main()
