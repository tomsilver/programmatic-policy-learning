"""Demo collection utilities."""

from typing import Any, Callable, TypeVar

import numpy as np

from programmatic_policy_learning.approaches.base_approach import BaseApproach
from programmatic_policy_learning.data.demo_types import Trajectory

EnvFactory = Callable[[], Any]
ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")


def collect_demo(
    env_factory: EnvFactory,
    expert: BaseApproach,
    max_demo_length: int | float = np.inf,
    env_num: int = 0,  # pylint: disable=unused-argument
) -> Trajectory[ObsT, ActT]:
    """Collect a demonstration trajectory from an environment using an expert
    policy."""

    env = env_factory(env_num)  # type: ignore
    # (becuase not all the providers have this env_num thing)
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
        # print("Nim board layout for env_num", env_num, ":\n", obs)
        t += 1
        expert.update(obs, reward, terminated, info)
        if terminated or truncated or (t >= max_demo_length):
            if not reward > 0:
                # keep behavior parity with original: warn if didn’t succeed
                print("WARNING: demo did not succeed!")
            break
    steps = list(zip(obs_list, act_list))
    return Trajectory(steps=steps)


def get_demonstrations(
    env_factory: EnvFactory,
    expert: BaseApproach,
    demo_numbers: tuple[int, ...],
    max_demo_length: int | float = np.inf,
) -> tuple[Trajectory, dict[int, Trajectory]]:
    """Collect multiple demonstration trajectories using an expert policy."""
    demonstrations: list[Trajectory] = []
    demo_dict: dict[int, Trajectory] = {}

    for i in demo_numbers:
        traj: Trajectory = collect_demo(
            env_factory,
            expert,
            max_demo_length=max_demo_length,
            env_num=i,
        )
        demonstrations.append(traj)
        demo_dict[i] = traj

    all_steps = [step for traj in demonstrations for step in traj.steps]
    return Trajectory(steps=all_steps), demo_dict
