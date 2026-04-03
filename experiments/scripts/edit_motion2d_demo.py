"""Replay a hand-edited Motion2D action sequence and save a GIF.

Edit ``EDITABLE_ACTIONS`` below, then run:

    python experiments/scripts/edit_motion2d_demo.py
    python experiments/scripts/edit_motion2d_demo.py --passages 1 --seed 0

The script rolls out the fixed action list in KinDER Motion2D and writes a GIF
under ``videos/`` so you can verify whether the sequence reaches the target.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Callable, cast

import gymnasium as gym
import kinder
import numpy as np

imageio: Any | None
try:
    import imageio.v2 as imageio  # type: ignore
except Exception:  # pylint: disable=broad-exception-caught
    imageio = None

logging.basicConfig(level=logging.INFO)

_MAX_STEPS = 500

_ACTION_FIELD_NAMES = ["dx", "dy", "dtheta", "darm", "vac"]

# Demo-0 action sequence copied from the current expert rollout.
# Edit this list manually to prune or replace noisy corrective actions.
# EDITABLE_ACTIONS: list[list[float]] = [
#     [0.04880309, 0.00549714, -0.02026846, 0.0, 0.0],
#     [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
#     [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
#     [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
#     [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
#     [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
#     [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
#     [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
#     [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
#     [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
#     [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
#     [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
#     [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
#     [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
#     [0.00424129, 0.04970834, -0.15251943, 0.0, 0.0],
#     [0.00424129, 0.04970834, -0.15251943, 0.0, 0.0],
#     [0.00424129, 0.04970834, -0.15251943, 0.0, 0.0],
#     [0.00424129, 0.04970834, -0.15251943, 0.0, 0.0],
#     [0.02284785, 0.04377603, -0.11203627, 0.0, 0.0],
#     [0.02284785, 0.04377603, -0.11203627, 0.0, 0.0],
#     [0.02284785, 0.04377603, -0.11203627, 0.0, 0.0],
#     [0.02284785, 0.04377603, -0.11203627, 0.0, 0.0],
#     [0.04697493, -0.03212217, -0.15574095, 0.0, 0.0],
#     [0.04697493, -0.03212217, -0.15574095, 0.0, 0.0],
#     [0.04697493, -0.03212217, -0.15574095, 0.0, 0.0],
#     [0.04697493, -0.03212217, -0.15574095, 0.0, 0.0],
#     [0.04697493, -0.03212217, -0.15574095, 0.0, 0.0],
#     [0.04697493, -0.03212217, -0.15574095, 0.0, 0.0],
#     [0.04697493, -0.03212217, -0.15574095, 0.0, 0.0],
#     [0.04697493, -0.03212217, -0.15574095, 0.0, 0.0],
#     [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
#     [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
#     [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
#     [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
#     [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
#     [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
#     [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
#     [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
#     [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
#     [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
#     [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
#     [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
#     [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
#     [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
#     [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
#     [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
#     [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
# ]
EDITABLE_ACTIONS: list[list[float]] = [
    [0.04880309, 0.00549714, -0.02026846, 0.0, 0.0],
    [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
    [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
    [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
    [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
    [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
    [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
    [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
    [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
    [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
    [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
    [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
    [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
    [0.00193347, 0.04022326, -0.19150342, 0.0, 0.0],
    [0.00424129, 0.04970834, -0.15251943, 0.0, 0.0],
    [0.00424129, 0.04970834, -0.15251943, 0.0, 0.0],
    [0.00424129, 0.04970834, -0.15251943, 0.0, 0.0],
    [0.00424129, 0.04970834, -0.15251943, 0.0, 0.0],
    [0.00424129, 0.04970834, -0.15251943, 0.0, 0.0],
    [0.00424129, 0.04970834, -0.15251943, 0.0, 0.0],
    [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
    [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
    [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
    [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
    [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
    [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
    [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
    [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
    [0.04697493, 0.040212217, -0.15574095, 0.0, 0.0],
    [0.04697493, 0.040212217, -0.15574095, 0.0, 0.0],
    [0.04697493, 0.040212217, -0.15574095, 0.0, 0.0],
    [0.04697493, 0.040212217, -0.15574095, 0.0, 0.0],
    [0.04697493, 0.040212217, -0.15574095, 0.0, 0.0],
    [0.04697493, 0.040212217, -0.15574095, 0.0, 0.0],
    [0.04697493, 0.040212217, -0.15574095, 0.0, 0.0],
    [0.0484335, 0, -0.08235537, 0.0, 0.0],
    [0.0484335, 0, -0.08235537, 0.0, 0.0],
    [0.0484335, 0, -0.08235537, 0.0, 0.0],
    [0.0484335, 0, -0.08235537, 0.0, 0.0],
    [0.05, 0, -0.08235537, 0.0, 0.0],
    [0.05, 0, -0.08235537, 0.0, 0.0],
    [0.05, 0, -0.08235537, 0.0, 0.0],
    [0.05, 0, -0.08235537, 0.0, 0.0],
    [0.05, 0, -0.08235537, 0.0, 0.0],
    [0.05, 0, -0.08235537, 0.0, 0.0],
    [0.05, 0, -0.08235537, 0.0, 0.0],
    [0.05, 0, -0.08235537, 0.0, 0.0],
    [0.05, 0, -0.08235537, 0.0, 0.0],
    [0.05, 0, -0.08235537, 0.0, 0.0],
    [0.05, 0, -0.08235537, 0.0, 0.0],
    [0.05, 0, -0.08235537, 0.0, 0.0],
    [0.05, 0, -0.08235537, 0.0, 0.0],
    [0.05, 0, -0.08235537, 0.0, 0.0],
    [0.05, 0, -0.08235537, 0.0, 0.0],
    [0.05, 0, -0.08235537, 0.0, 0.0],
    [0.05, 0, -0.08235537, 0.0, 0.0],
    [0.05, 0, -0.08235537, 0.0, 0.0],
    [0.05, 0, -0.08235537, 0.0, 0.0],
    # [0.04697493, -0.03212217, -0.15574095, 0.0, 0.0],
    # [0.04697493, -0.03212217, -0.15574095, 0.0, 0.0],
    # [0.04697493, -0.03212217, -0.15574095, 0.0, 0.0],
    # [0.04697493, -0.03212217, -0.15574095, 0.0, 0.0],
    # [0.04697493, -0.03212217, -0.15574095, 0.0, 0.0],
    # [0.04697493, -0.03212217, -0.15574095, 0.0, 0.0],
    # [0.04697493, -0.03212217, -0.15574095, 0.0, 0.0],
    # [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
    # [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
    # [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
    # [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
    # [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
    # [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
    # [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
    # [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
    # [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
    # [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
    # [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
    # [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
    # [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
    # [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
    # [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
    # [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
    # [0.0484335, 0.00602929, -0.08235537, 0.0, 0.0],
]


def _close_env(env: Any) -> None:
    close_fn = cast(Callable[[], None] | None, getattr(env, "close", None))
    if close_fn is not None:
        close_fn()


def save_gif(frames: list[np.ndarray], path: str, fps: int = 12) -> None:
    """Save frames as a GIF."""
    if imageio is None:
        raise RuntimeError("Saving GIFs requires imageio.")

    clean: list[np.ndarray] = [
        f[:, :, :3] if f.ndim == 3 and f.shape[2] == 4 else f for f in frames
    ]
    imageio.mimsave(path, cast(Any, clean), fps=fps)
    logging.info("GIF saved to: %s  (%d frames)", path, len(clean))


def make_env(env_id: str, seed: int) -> tuple[gym.Env, np.ndarray]:
    """Create the environment and reset it."""
    kinder.register_all_environments()
    env = kinder.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    return env, obs


def log_action_space(action_space: gym.spaces.Box) -> None:
    """Log action bounds for reference while editing."""
    assert isinstance(action_space, gym.spaces.Box)
    for i in range(action_space.shape[0]):
        name = _ACTION_FIELD_NAMES[i] if i < len(_ACTION_FIELD_NAMES) else f"a[{i}]"
        logging.info(
            "[%d] %s: low=%.6f high=%.6f",
            i,
            name,
            float(action_space.low[i]),
            float(action_space.high[i]),
        )


def run_fixed_actions(
    env: gym.Env,
    obs: np.ndarray,
    actions: list[list[float]],
) -> tuple[list[np.ndarray], float, int, bool]:
    """Replay the given action list and return rollout artifacts."""
    del obs
    frames: list[np.ndarray] = []
    total_reward = 0.0
    terminated = False

    for step, action_list in enumerate(actions[:_MAX_STEPS]):
        raw: np.ndarray | list[np.ndarray] | None = env.render()
        if raw is not None:
            frames.append(np.asarray(raw))

        action = np.asarray(action_list, dtype=np.float32)
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        logging.info(
            "step=%03d action=%s reward=%.4f done=%s truncated=%s",
            step,
            np.array2string(action, precision=6, floatmode="fixed"),
            float(reward),
            terminated,
            truncated,
        )

        if terminated or truncated:
            raw = env.render()
            if raw is not None:
                frames.append(np.asarray(raw))
            return frames, total_reward, step + 1, bool(terminated)

    return frames, total_reward, min(len(actions), _MAX_STEPS), terminated


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Replay a hand-edited Motion2D action sequence."
    )
    parser.add_argument("--passages", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output GIF path. Defaults to videos/manual_motion2d_*.mp4",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run the editable action sequence and save a GIF."""
    env_id = f"kinder/Motion2D-p{args.passages}-v0"
    env, obs = make_env(env_id, args.seed)

    logging.info("Env: %s", env_id)
    logging.info("Initial obs shape: %s", obs.shape)
    assert isinstance(env.action_space, gym.spaces.Box)
    log_action_space(env.action_space)
    logging.info("Number of editable actions: %d", len(EDITABLE_ACTIONS))

    try:
        frames, total_reward, num_steps, reached_target = run_fixed_actions(
            env,
            obs,
            EDITABLE_ACTIONS,
        )
    finally:
        _close_env(env)

    logging.info(
        "Finished replay: steps=%d total_reward=%.4f reached_target=%s",
        num_steps,
        total_reward,
        reached_target,
    )

    out_path = args.out
    if not out_path:
        video_dir = Path("videos")
        video_dir.mkdir(exist_ok=True)
        out_path = str(video_dir / f"manual_motion2d_p{args.passages}_s{args.seed}.mp4")
    if frames:
        save_gif(frames, out_path)
    else:
        logging.warning("No frames were captured; GIF was not written.")


if __name__ == "__main__":
    main(parse_args())
