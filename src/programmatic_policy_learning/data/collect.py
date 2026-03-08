"""Demo collection utilities."""

import logging
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

from programmatic_policy_learning.approaches.base_approach import BaseApproach
from programmatic_policy_learning.data.demo_types import Trajectory

EnvFactory = Callable[[], Any]
ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")


# def _maybe_frame(raw: Any) -> np.ndarray | None:
#     """Normalize renderer output into a single frame when possible."""
#     if raw is None:
#         return None
#     if isinstance(raw, list):
#         if not raw:
#             return None
#         raw = raw[-1]
#     frame = np.asarray(raw)
#     if frame.ndim != 3:
#         return None
#     return frame


# def _save_video(frames: list[np.ndarray], path: str) -> None:
#     """Save rollout frames to an mp4 file."""
#     clean = [f[:, :, :3] if f.ndim == 3 and f.shape[2] == 4 else f for f in frames]
#     try:
#         from moviepy import ImageSequenceClip  # type: ignore[import-untyped]
#     except Exception as exc:  # pylint: disable=broad-exception-caught
#         logging.warning("Video saving skipped: moviepy unavailable (%s)", exc)
#         return
#     clip = ImageSequenceClip(clean, fps=20)
#     clip.write_videofile(path, codec="libx264", logger=None)
#     logging.info("Demo video saved: %s (%d frames)", path, len(clean))


def collect_demo(
    env_factory: EnvFactory,
    expert: BaseApproach,
    max_demo_length: int | float = np.inf,
    env_num: int = 0,  # pylint: disable=unused-argument
) -> Trajectory[ObsT, ActT]:
    """Collect a demonstration trajectory from an environment using an expert
    policy."""

    env = env_factory(env_num)  # type: ignore

    # (because not all the providers have this 'env_num' attribute)
    reset_out = env.reset(seed=env_num)
    assert (
        isinstance(reset_out, tuple) and len(reset_out) == 2
    ), f"Expected env.reset() to return (obs, info), got {reset_out}"
    obs, info = reset_out

    obs_list: list[ObsT] = []
    act_list: list[ActT] = []
    # frames: list[np.ndarray] = []

    # first_frame = _maybe_frame(env.render() if hasattr(env, "render") else None)
    # if first_frame is not None:
    #     frames.append(first_frame)

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
        # frame = _maybe_frame(env.render() if hasattr(env, "render") else None)
        # if frame is not None:
        #     frames.append(frame)

        t += 1
        expert.update(obs, reward, terminated, info)
        if terminated or truncated or (t >= max_demo_length):
            print("REWARD WHEN DONE:", reward)
            if not reward > 0:
                # keep behavior parity with original: warn if didn’t succeed
                logging.warning("WARNING: demo did not succeed!")
            break

    steps = list(zip(obs_list, act_list))
    # if frames:
    #     video_dir = Path("videos")
    #     video_dir.mkdir(exist_ok=True)
    #     _save_video(frames, str(video_dir / f"collect_demo_{env_num}.mp4"))

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
