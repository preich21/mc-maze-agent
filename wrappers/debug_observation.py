import gymnasium as gym
import numpy as np

from mc_env.action import MinecraftAction
from mc_env.observation import MinecraftObservation
from mc_env.env import BlockTypes


class DebugMinecraftObsWrapper(gym.Wrapper[MinecraftObservation, int, MinecraftObservation, int]):
    """Logs learning-signal diagnostics using raw MinecraftObservation (pre-vectorizer).

    Tracks:
    - goal visibility frequency + min visible goal distance
    - deaths / episode count
    - episode return and length
    """

    def __init__(self, env: gym.Env, print_every_episodes: int = 10):
        super().__init__(env)
        self.print_every_episodes = int(print_every_episodes)

        self._global_steps = 0
        self._episode_steps = 0
        self._episode_return = 0.0

        self._episodes = 0
        self._deaths = 0

        self._goal_seen_steps = 0
        self._goal_seen_episodes = 0
        self._last_min_goal_dist = None

        self._pitch_delta_avg = 0.0
        self._pitch_delta_max = 0.0
        self._yaw_delta_avg = 0.0
        self._yaw_delta_max = 0.0
        self.forward_count = 0
        self.backward_count = 0
        self.left_count = 0
        self.right_count = 0


    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)

        self._episode_steps = 0
        self._episode_return = 0.0
        self._last_min_goal_dist = None

        self._pitch_delta_avg = 0.0
        self._pitch_delta_max = 0.0
        self._yaw_delta_avg = 0.0
        self._yaw_delta_max = 0.0

        return obs, info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)

        self._global_steps += 1
        self._episode_steps += 1
        self._episode_return += float(reward)

        parsed_action = MinecraftAction.from_vector(action)

        self._yaw_delta_avg += parsed_action.yawDelta
        self._pitch_delta_avg += parsed_action.pitchDelta
        if parsed_action.yawDelta > self._yaw_delta_max:
            self._yaw_delta_max = parsed_action.yawDelta
        if parsed_action.pitchDelta > self._pitch_delta_max:
            self._pitch_delta_max = parsed_action.pitchDelta

        min_goal_dist = self._get_min_visible_goal_distance(obs)
        if min_goal_dist is not None:
            self._goal_seen_steps += 1
            self._last_min_goal_dist = min_goal_dist

        if parsed_action.moveForward:
            self.forward_count += 1
        if parsed_action.moveBackward:
            self.backward_count += 1
        if parsed_action.moveRight:
            self.right_count += 1
        if parsed_action.moveLeft:
            self.left_count += 1
        info["debug/forward"] = float(self.forward_count / self._global_steps)
        info["debug/backward"] = float(self.backward_count / self._global_steps)
        info["debug/left"] = float(self.left_count / self._global_steps)
        info["debug/right"] = float(self.right_count / self._global_steps)

        done = bool(terminated) or bool(truncated)
        if done:
            self._episodes += 1

            died = bool(getattr(obs, "died", False))
            if died:
                self._deaths += 1

            if min_goal_dist is not None:
                self._goal_seen_episodes += 1

            goal_seen_ep_rate = self._goal_seen_episodes / max(1, self._episodes)
            death_rate = self._deaths / max(1, self._episodes)

            # Attach episode metrics to info so callbacks can log them to TensorBoard
            info["debug/ep_len"] = int(self._episode_steps)
            info["debug/ep_return"] = float(self._episode_return)
            info["debug/goal_seen_steps"] = int(self._goal_seen_steps)
            info["debug/goal_seen_ep_rate"] = float(goal_seen_ep_rate)
            info["debug/death_rate"] = float(death_rate)
            info["debug/last_min_goal_dist"] = (
                float(self._last_min_goal_dist) if self._last_min_goal_dist is not None else float("nan")
            )
            info["debug/died"] = int(died)
            self._yaw_delta_avg = self._yaw_delta_avg / self._episode_steps
            self._pitch_delta_avg = self._pitch_delta_avg / self._episode_steps
            info["debug/yaw_delta_avg"] = self._yaw_delta_avg
            info["debug/pitch_delta_avg"] = self._pitch_delta_avg
            info["debug/yaw_delta_max"] = self._yaw_delta_max
            info["debug/pitch_delta_max"] = self._pitch_delta_max

            if self._episodes % self.print_every_episodes == 0:
                print(
                    "[debug] "
                    f"episodes={self._episodes} "
                    f"steps={self._global_steps} "
                    f"ep_len={self._episode_steps} "
                    f"ep_return={self._episode_return:.3f} "
                    f"goal_seen_steps={self._goal_seen_steps} "
                    f"goal_seen_ep_rate={goal_seen_ep_rate:.2%} "
                    f"death_rate={death_rate:.2%} "
                    f"last_min_goal_dist={self._last_min_goal_dist}"
                    f"yaw_delta_avg={self._yaw_delta_avg:.3f} "
                    f"pitch_delta_avg={self._pitch_delta_avg:.3f} "
                    f"yaw_delta_max={self._yaw_delta_max:.3f} "
                    f"pitch_delta_max={self._pitch_delta_max:.3f} "
                )

        return obs, reward, terminated, truncated, info

    @staticmethod
    def _get_min_visible_goal_distance(obs: MinecraftObservation):
        min_dist = None
        for blk, dist in zip(obs.fovBlocks, obs.fovDistances):
            if int(blk) == int(BlockTypes.GOAL_BLOCK):
                d = float(dist)
                if d >= 0.0:
                    if min_dist is None or d < min_dist:
                        min_dist = d
        return min_dist