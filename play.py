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

import time

from stable_baselines3 import PPO

from mc_env.env import MinecraftEnv
from wrappers.observation_vectorizer import ObservationVectorizer
from wrappers.simple_goal_reward import SimpleGoalRewardWrapper

# Keep runtime config local to play.py to avoid accidental mismatch.
URI = "ws://127.0.0.1:8081"
STEP_TICKS = 2
MAX_STEPS = 500

MODEL_PATH = "runs/ppo_minecraft/ppo_minecraft_goal.zip"
N_EPISODES = 5
DETERMINISTIC = True
SLEEP_BETWEEN_STEPS_SEC = 0.0  # set >0 for slower visible playback


def make_env():
    env = MinecraftEnv(uri=URI, step_ticks=STEP_TICKS)
    env = SimpleGoalRewardWrapper(env)
    env.max_steps = MAX_STEPS
    env = ObservationVectorizer(env)
    return env


def main() -> None:
    env = make_env()
    try:
        model = PPO.load(MODEL_PATH)

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
