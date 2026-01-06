"""Tiny benchmarking helpers for policy inference and env step latency."""
from __future__ import annotations

import time
from typing import Iterable

import numpy as np


def summarize_ms(name: str, samples_s: Iterable[float]) -> None:
    arr = np.asarray(list(samples_s), dtype=np.float64) * 1000.0
    if arr.size == 0:
        print(f"{name}: no samples")
        return
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    mean = float(arr.mean())
    print(f"{name}: mean={mean:.3f}ms p50={p50:.3f}ms p95={p95:.3f}ms n={arr.size}")


def benchmark_decision_speed(model, vec_env, steps: int = 200, deterministic: bool = True) -> None:  # noqa: ANN001
    """Measure model.predict latency vs env.step latency.

    vec_env is expected to be a VecEnv (DummyVecEnv in this project).
    """
    obs = vec_env.reset()

    infer_times = []
    step_times = []

    for _ in range(steps):
        t0 = time.perf_counter()
        action, _state = model.predict(obs, deterministic=deterministic)
        t1 = time.perf_counter()
        obs, rewards, dones, infos = vec_env.step(action)
        t2 = time.perf_counter()

        infer_times.append(t1 - t0)
        step_times.append(t2 - t1)

        if bool(dones[0]):
            obs = vec_env.reset()

    summarize_ms("policy.predict()", infer_times)
    summarize_ms("env.step()", step_times)

    total = np.asarray(infer_times) + np.asarray(step_times)
    summarize_ms("total per decision", total)

    dps = float(1.0 / np.mean(total))
    print(f"decisions/sec: {dps:.2f}")

