"""Basic integration test that exercises one reset/step over WebSocket."""
from __future__ import annotations

import unittest

import numpy as np

from env_mc import MinecraftEnv, MinecraftAction, MinecraftObservation


class TestMinecraftWsIntegration(unittest.TestCase):
    def test_single_action_round_trip(self) -> None:
        env = MinecraftEnv()
        step_result = None
        try:
            action = MinecraftAction(
                move=np.zeros(5, dtype=np.int8),
                look=np.zeros(2, dtype=np.float32),
            )
            try:
                step_result = env.step(action)
            except (OSError, ConnectionRefusedError, TimeoutError, RuntimeError) as exc:
                self.skipTest(f"WebSocket server unavailable: {exc}")
        finally:
            env.close()

        obs, reward, terminated, truncated, info = step_result
        self.assertIsInstance(obs, MinecraftObservation)
        self.assertIsInstance(info, dict)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)

    def test_reset_trip(self) -> None:
        env = MinecraftEnv()
        step_result = None
        try:
            try:
                step_result = env.reset()
            except (OSError, ConnectionRefusedError, TimeoutError, RuntimeError) as exc:
                self.skipTest(f"WebSocket server unavailable: {exc}")
        finally:
            env.close()

        obs, info = step_result
        self.assertIsInstance(obs, MinecraftObservation)
        self.assertIsInstance(info, dict)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
