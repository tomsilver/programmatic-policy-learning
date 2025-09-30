"""Demo collection utilities."""

import logging
from typing import Any, Callable

import numpy as np

from programmatic_policy_learning.data.demo_types import Demo, Trajectory

EnvFactory = Callable[[], Any]


def collect_demo(
    env_factory: EnvFactory, expert: Any, max_demo_length: int | float = np.inf
) -> Trajectory:
    """Collect a demonstration trajectory from an environment using an expert
    policy."""

    env = env_factory()
    reset_out = env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, info = reset_out
    else:
        obs = reset_out
        info = {}
        logging.warning("env.reset() returned a single value (old gym API)")

    steps: list[Demo] = []
    t = 0
    expert.reset(obs, info)
    while True:
        action = expert.step()
        steps.append(Demo(obs=obs, act=action))
        step_out = env.step(action)
        # handle gym vs. gymnasium
        if len(step_out) == 4:
            obs, reward, done, _ = step_out
            terminated, truncated = done, False
        else:
            obs, reward, terminated, truncated, _ = step_out
        t += 1
        if terminated or truncated or (t >= max_demo_length):
            if not reward > 0:
                # keep behavior parity with original: warn if didn’t succeed
                print("WARNING: demo did not succeed!")
            break
    return Trajectory(steps=steps)
