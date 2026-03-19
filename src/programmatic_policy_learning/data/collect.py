"""Demo collection utilities."""

import logging
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

    # print(type(expert))
    # print(max_demo_length)
    # obs, info = env.reset(seed=0)
    # expert.reset(obs, info)
    # for t in range(1000):
    #     action = expert.step()
    #     print("t", t, "action", action)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     print("reward", reward, "terminated", terminated, "truncated", truncated)
    #     expert.update(obs, reward, terminated or truncated, info)
    #     if terminated or truncated:
    #         print("final reward:", reward)
    #         print("final terminated:", terminated)
    #         print("final truncated:", truncated)
    #         print("final info:", info)
    #         print("final obs:", obs)
    #         print("robot xy:", obs[0], obs[1])
    #         print("target xy:", obs[9], obs[10])

    #         rx, ry, r = obs[0], obs[1], obs[3]
    #         tx, ty, tw, th = obs[9], obs[10], obs[17], obs[18]

    #         print("robot center:", (rx, ry))
    #         print("target rect:", (tx, ty, tw, th))
    #         print("target center:", (tx + tw / 2, ty + th / 2))
    #         print("feasible success center x-range:", (tx + r, tx + tw - r))
    #         print("feasible success center y-range:", (ty + r, ty + th - r))

    #         break
    # input()

    # Support both gymnasium reset(seed=...) and older reset() signatures.
    try:
        reset_out = env.reset(seed=env_num)
    except TypeError:
        reset_out = env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, info = reset_out
    else:
        obs, info = reset_out, {}

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
        expert.update(obs, reward, terminated or truncated, info)
        if terminated or truncated or (t >= max_demo_length):
            print("REWARD WHEN DONE:", reward)
            if not reward > 0:
                # keep behavior parity with original: warn if didn’t succeed
                logging.warning("WARNING: demo did not succeed!")
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
