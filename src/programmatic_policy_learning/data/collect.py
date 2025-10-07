"""Demo collection utilities."""

from typing import Any, Callable, TypeVar

import numpy as np

from programmatic_policy_learning.data.demo_types import Trajectory

EnvFactory = Callable[[], Any]
ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")


def collect_demo(
    env_factory: EnvFactory, expert: Any, max_demo_length: int | float = np.inf
) -> Trajectory[ObsT, ActT]:
    """Collect a demonstration trajectory from an environment using an expert
    policy."""

    env = env_factory()
    reset_out = env.reset()
    assert (
        isinstance(reset_out, tuple) and len(reset_out) == 2
    ), f"Expected env.reset() to return (obs, info), got {reset_out}"
    obs, info = reset_out

    obs_list: list[ObsT] = []
    act_list: list[ActT] = []

    t = 0
    expert.reset(obs, info)
    while True:
        action = expert.step()
        obs_list.append(obs)
        act_list.append(action)

        step_out = env.step(action)

        # handle gym vs. gymnasium
        if len(step_out) == 4:
            obs, reward, done, info = step_out
            terminated, truncated = done, False
        else:
            obs, reward, terminated, truncated, info = step_out
        t += 1
        expert.update(obs, reward, terminated, info)
        if terminated or truncated or (t >= max_demo_length):
            if not reward > 0:
                # keep behavior parity with original: warn if didn’t succeed
                print("WARNING: demo did not succeed!")
            break

    steps = list(zip(obs_list, act_list))
    return Trajectory(steps=steps)
