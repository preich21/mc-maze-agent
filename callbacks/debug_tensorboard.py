from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class DebugMetricsTensorboardCallback(BaseCallback):
    """Logs `info["debug/..."]` scalars to TensorBoard on episode end."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        dones = self.locals.get("dones", None)
        if infos is None or dones is None:
            return True

        for info, done in zip(infos, dones):
            if not bool(done) or not isinstance(info, dict):
                continue

            # Write any debug scalars present
            for k, v in info.items():
                if not isinstance(k, str) or not k.startswith("debug/"):
                    continue
                if v is None:
                    continue
                try:
                    self.logger.record(k, float(v))
                except (TypeError, ValueError):
                    continue

        return True