"""Debug script: Run Motion2D using the rejection-sampling expert (f_1 ∧ f_2).

Records a video of the rollout.

Usage::

    python experiments/scripts/run_motion2d_rollout.py # default p0
    python experiments/scripts/run_motion2d_rollout.py --passages 5
    python experiments/scripts/run_motion2d_rollout.py --passages 3 --seed 0
"""

import argparse
import logging
from pathlib import Path

import gymnasium as gym
import kinder
import numpy as np
from moviepy import ImageSequenceClip  # type: ignore[import-untyped]

from programmatic_policy_learning.approaches.experts.motion2d_experts import (
    Motion2DRejectionSamplingExpert,
    create_motion2d_expert,
)

logging.basicConfig(level=logging.INFO)
_MAX_STEPS = 500


_ACTION_FIELD_NAMES = ["dx", "dy", "dtheta", "darm", "vac"]


def log_obs(obs: np.ndarray) -> None:
    """Log a structured summary of every observation field."""
    logging.info("--- Robot (indices 0-8) ---")
    logging.info("  [0] x=%.4f", obs[0])
    logging.info("  [1] y=%.4f", obs[1])
    logging.info("  [2] theta=%.4f", obs[2])
    logging.info("  [3] base_radius=%.4f", obs[3])
    logging.info("  [4] arm_joint=%.4f", obs[4])
    logging.info("  [5] arm_length=%.4f", obs[5])
    logging.info("  [6] vacuum=%.4f", obs[6])
    logging.info("  [7] gripper_height=%.4f", obs[7])
    logging.info("  [8] gripper_width=%.4f", obs[8])

    logging.info("--- Target (indices 9-18) ---")
    logging.info("  [9]  x=%.4f", obs[9])
    logging.info("  [10] y=%.4f", obs[10])
    logging.info("  [11] theta=%.4f", obs[11])
    logging.info("  [12] static=%.4f", obs[12])
    logging.info("  [13] color_r=%.4f", obs[13])
    logging.info("  [14] color_g=%.4f", obs[14])
    logging.info("  [15] color_b=%.4f", obs[15])
    logging.info("  [16] z_order=%.4f", obs[16])
    logging.info("  [17] width=%.4f", obs[17])
    logging.info("  [18] height=%.4f", obs[18])

    num_obstacles = (len(obs) - 19) // 10
    num_passages = num_obstacles // 2
    if num_obstacles > 0:
        logging.info(
            "--- Obstacles (%d total, %d passages) ---",
            num_obstacles,
            num_passages,
        )

    for i in range(num_obstacles):
        base = 19 + 10 * i
        logging.info(
            "  Obstacle %d (indices %d-%d):",
            i,
            base,
            base + 9,
        )
        logging.info("    [%d] x=%.4f", base, obs[base])
        logging.info("    [%d] y=%.4f", base + 1, obs[base + 1])
        logging.info("    [%d] theta=%.4f", base + 2, obs[base + 2])
        logging.info("    [%d] static=%.4f", base + 3, obs[base + 3])
        logging.info("    [%d] color_r=%.4f", base + 4, obs[base + 4])
        logging.info("    [%d] color_g=%.4f", base + 5, obs[base + 5])
        logging.info("    [%d] color_b=%.4f", base + 6, obs[base + 6])
        logging.info("    [%d] z_order=%.4f", base + 7, obs[base + 7])
        logging.info("    [%d] width=%.4f", base + 8, obs[base + 8])
        logging.info("    [%d] height=%.4f", base + 9, obs[base + 9])

    for i in range(num_passages):
        bot_base = 19 + 20 * i
        top_base = bot_base + 10

        bot_y = obs[bot_base + 1]
        bot_h = obs[bot_base + 9]
        top_y = obs[top_base + 1]
        wall_x = obs[bot_base]

        gap_bottom = bot_y + bot_h
        gap_top = top_y
        passage_y = (gap_bottom + gap_top) / 2

        logging.info(
            "  Passage %d: wall_x=%.4f, gap=[%.4f, %.4f], center=%.4f",
            i,
            wall_x,
            gap_bottom,
            gap_top,
            passage_y,
        )


def log_action_space(action_space: gym.spaces.Box) -> None:
    """Log the name, bounds, and dtype of each action dimension."""
    assert isinstance(action_space, gym.spaces.Box)
    logging.info("--- Action space (%s) ---", action_space)
    for i in range(action_space.shape[0]):
        name = _ACTION_FIELD_NAMES[i] if i < len(_ACTION_FIELD_NAMES) else f"a[{i}]"
        logging.info(
            "  [%d] %s: low=%.4f, high=%.4f",
            i,
            name,
            action_space.low[i],
            action_space.high[i],
        )


def save_video(frames: list[np.ndarray], path: str) -> None:
    """Strip alpha channel if present and save frames as mp4."""
    clean = [f[:, :, :3] if f.ndim == 3 and f.shape[2] == 4 else f for f in frames]
    clip = ImageSequenceClip(clean, fps=20)
    clip.write_videofile(path, codec="libx264", logger=None)
    logging.info("Video saved to: %s  (%d frames)", path, len(clean))


def make_env(
    env_id: str, seed: int
) -> tuple[gym.Env, np.ndarray, Motion2DRejectionSamplingExpert]:
    """Create the environment, reset it, and build the expert."""
    kinder.register_all_environments()
    env = kinder.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    assert isinstance(env.action_space, gym.spaces.Box)
    expert = create_motion2d_expert(env.action_space, seed=seed)
    return env, obs, expert


def run_rollout(
    env: gym.Env,
    obs: np.ndarray,
    expert: Motion2DRejectionSamplingExpert,
) -> tuple[list[np.ndarray], float, int, bool]:
    """Execute the rollout loop and return (frames, reward, steps,
    terminated)."""
    frames: list[np.ndarray] = []
    total_reward = 0.0

    for step in range(_MAX_STEPS):
        raw: np.ndarray | list[np.ndarray] | None = env.render()

        if raw is not None:
            frames.append(np.asarray(raw))

        action = expert(obs)
        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += float(reward)

        if terminated or truncated:
            raw = env.render()
            if raw is not None:
                frames.append(np.asarray(raw))
            return frames, total_reward, step, terminated

    return frames, total_reward, _MAX_STEPS, False


def main(args: argparse.Namespace) -> None:
    """Run the expert rollout and save a video."""
    env_id = f"kinder/Motion2D-p{args.passages}-v0"
    env, obs, expert = make_env(env_id, args.seed)

    logging.info("Env: %s", env_id)
    logging.info("Observation space: %s", env.observation_space)
    logging.info("Obs shape: %s", obs.shape)
    assert isinstance(env.action_space, gym.spaces.Box)
    logging.info("=== Action Space ===")
    log_action_space(env.action_space)
    logging.info("=== Initial Observation ===")
    log_obs(obs)

    frames, total_reward, steps, terminated = run_rollout(env, obs, expert)
    env.close()

    if terminated:
        logging.info("Done at step %d!", steps)
    else:
        logging.warning("Reached max steps (%d) without termination", _MAX_STEPS)
    logging.info("Total reward: %.1f", total_reward)

    if frames:
        video_dir = Path("videos")
        video_dir.mkdir(exist_ok=True)
        save_video(
            frames,
            str(video_dir / f"motion2d_p{args.passages}_rollout.mp4"),
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Motion2D rollout with expert.")
    parser.add_argument(
        "--passages",
        type=int,
        default=0,
        help="Number of wall passages (0-5, default: 0)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
