import time

from mc_env.observation import MinecraftObservation

class ObservationsBuffer:
    latest_obs: MinecraftObservation | None
    skipped_obs: int

    def __init__(self, obs_timeout_s: float = 2.0):
        self.clear()
        self.obs_timeout_s = obs_timeout_s

    def add_observation(self, observation: MinecraftObservation):
        self.latest_obs = observation
        self.skipped_obs += 1

    def get_observation(self) -> tuple[MinecraftObservation, int]:
        start_time = time.time()
        while self.latest_obs is None:
            if (time.time() - start_time) > self.obs_timeout_s:
                raise TimeoutError("Timed out waiting for new observation")
            time.sleep(0.001)
        obs = self.latest_obs
        skipped_obs = self.skipped_obs
        self.clear()
        return obs, skipped_obs

    def clear(self):
        self.latest_obs = None
        self.skipped_obs = -1