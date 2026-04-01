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

EnvFactory = Callable[[], Any]
ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")


def _log_motion2d_bucket_purity(
    env_factory: EnvFactory,
    demo_numbers: tuple[int, ...],
    trajectories: list[Trajectory],
    *,
    bucket_counts: int = 3,
    bucket_edges: Sequence[float] | None = None,
    eps: float = 1e-8,
) -> None:
    """Log bucket-purity statistics for collected Motion2D expert actions."""
    if not trajectories or not demo_numbers:
        return

    env = env_factory(int(demo_numbers[0]))  # type: ignore[arg-type]
    action_space = getattr(env, "action_space", None)
    if action_space is None or not hasattr(action_space, "low") or not hasattr(
        action_space, "high"
    ):
        return

    try:
        low = np.asarray(action_space.low, dtype=float).reshape(-1)
        high = np.asarray(action_space.high, dtype=float).reshape(-1)
        if low.size < 2 or high.size < 2:
            return
        quantizer = Motion2DActionQuantizer.from_bounds(
            low[:2],
            high[:2],
            bucket_counts=bucket_counts,
            bucket_edges=bucket_edges,

        )
    except Exception:  # pylint: disable=broad-exception-caught
        return
    finally:
        if hasattr(env, "close"):
            env.close()

    bucket_to_actions: dict[tuple[int, ...], list[np.ndarray]] = defaultdict(list)
    for traj in trajectories:
        for _, action in traj.steps:
            action_arr = np.asarray(action, dtype=float).reshape(-1)
            if action_arr.size < 2:
                continue
            action_xy = action_arr[:2]
            bucket = quantizer.quantize(action_xy)
            bucket_to_actions[bucket].append(action_xy)

    if not bucket_to_actions:
        return

    header = (
        "Motion2D bucket purity stats over collected expert actions "
        f"(bucket_edges={list(bucket_edges) if bucket_edges is not None else None}, "
        f"bucket_counts={bucket_counts})"
    )
    logging.info(header)
    print(header)
    for bucket in sorted(bucket_to_actions):
        arr = np.asarray(bucket_to_actions[bucket], dtype=float)
        dx = arr[:, 0]
        dy = arr[:, 1]
        angles = np.unwrap(np.arctan2(dy, dx))
        ratio = np.abs(dx) / (np.abs(dy) + float(eps))
        near_horizontal = np.abs(dx) >= 2.0 * (np.abs(dy) + float(eps))
        near_vertical = np.abs(dy) >= 2.0 * (np.abs(dx) + float(eps))
        mixed_hv = bool(np.any(near_horizontal) and np.any(near_vertical))
        msg = (
            f"Bucket {bucket}: n={len(arr)} "
            f"var_dx={float(np.var(dx)):.8f} "
            f"var_dy={float(np.var(dy)):.8f} "
            f"var_angle={float(np.var(angles)):.8f} "
            f"var_|dx|/(|dy|+eps)={float(np.var(ratio)):.8f} "
            f"mixed_hv={mixed_hv}"
        )
        logging.info(msg)
        print(msg)



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
    if action_space is not None and hasattr(action_space, "low") and hasattr(
        action_space, "high"
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
    print(demo_numbers)
    for i in demo_numbers:
        traj: Trajectory = collect_demo(
            env_factory,
            expert,
            max_demo_length=max_demo_length,
            env_num=i,
        )
        demonstrations.append(traj)
        demo_dict[i] = traj
    # _log_motion2d_bucket_purity(
    #     env_factory,
    #     demo_numbers,
    #     demonstrations,
    #     bucket_edges=[-0.05, -0.006, 0.0, 0.006, 0.05],
    # )
    all_steps = [step for traj in demonstrations for step in traj.steps]
    return Trajectory(steps=all_steps), demo_dict
