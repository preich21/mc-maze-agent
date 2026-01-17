import gymnasium as gym

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

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)

        self._episode_steps = 0
        self._episode_return = 0.0
        self._last_min_goal_dist = None

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._global_steps += 1
        self._episode_steps += 1
        self._episode_return += float(reward)

        min_goal_dist = self._get_min_visible_goal_distance(obs)
        if min_goal_dist is not None:
            self._goal_seen_steps += 1
            self._last_min_goal_dist = min_goal_dist

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
            info = dict(info)
            info["debug/episodes"] = int(self._episodes)
            info["debug/global_steps"] = int(self._global_steps)
            info["debug/ep_len"] = int(self._episode_steps)
            info["debug/ep_return"] = float(self._episode_return)
            info["debug/goal_seen_steps"] = int(self._goal_seen_steps)
            info["debug/goal_seen_ep_rate"] = float(goal_seen_ep_rate)
            info["debug/death_rate"] = float(death_rate)
            info["debug/last_min_goal_dist"] = (
                float(self._last_min_goal_dist) if self._last_min_goal_dist is not None else float("nan")
            )
            info["debug/died"] = int(died)

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