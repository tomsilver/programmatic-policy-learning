"""Demo collection utilities."""

import logging
from collections import defaultdict
from typing import Any, Callable, Sequence, TypeVar

import numpy as np

from programmatic_policy_learning.approaches.base_approach import BaseApproach
from programmatic_policy_learning.data.demo_types import Trajectory
from programmatic_policy_learning.utils.action_quantization import (
    Motion2DActionQuantizer,
)

EnvFactory = Callable[..., Any]
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
    if hasattr(expert, "set_env"):
        expert.set_env(env)

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
    quantizer: Motion2DActionQuantizer | None = None
    action_space = getattr(env, "action_space", None)
    if (
        action_space is not None
        and hasattr(action_space, "low")
        and hasattr(action_space, "high")
    ):
        try:
            quantizer = Motion2DActionQuantizer.from_bounds(
                action_space.low,
                action_space.high,
                bucket_edges=[-0.05, -0.02, -0.006, 0.0, 0.006, 0.02, 0.05],
            )
        except Exception:  # pylint: disable=broad-exception-caught
            quantizer = None

    t = 0
    expert.reset(obs, info)
    while True:
        action = expert.step()
        print("ACTION:", action)
        if quantizer is not None:
            try:
                print("ACTION BUCKET:", quantizer.quantize(action))
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        print("OBSERVATION:", obs)

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
    logging.info("Collecting demonstrations for environments: %s", demo_numbers)
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
