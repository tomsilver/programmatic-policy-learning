"""Debug script: Run Motion2D using the rejection-sampling expert (f_1 ∧ f_2).

Records a video of the rollout.

Usage::

    python experiments/scripts/run_motion2d_rollout.py                        # default p1
    python experiments/scripts/run_motion2d_rollout.py --passages 5
    python experiments/scripts/run_motion2d_rollout.py --passages 3 --seed 0
"""

import argparse
from pathlib import Path

import kinder
import numpy as np

from programmatic_policy_learning.approaches.experts.motion2d_experts import (
    create_motion2d_expert,
)

_MAX_STEPS = 500


def print_obs(obs: np.ndarray) -> None:
    """Print a structured summary of the observation vector."""
    print(f"Robot:     x={obs[0]:.4f}, y={obs[1]:.4f}, theta={obs[2]:.4f}")
    print(f"           base_radius={obs[3]:.4f}")
    print(f"Target:    x={obs[9]:.4f}, y={obs[10]:.4f}, theta={obs[11]:.4f}")
    print(f"           width={obs[17]:.4f}, height={obs[18]:.4f}")

    num_obstacles = (len(obs) - 19) // 10
    num_passages = num_obstacles // 2

    for i in range(num_passages):
        bot_base = 19 + 20 * i
        top_base = bot_base + 10

        bot_y = obs[bot_base + 1]
        bot_h = obs[bot_base + 9]
        top_y = obs[top_base + 1]
        top_h = obs[top_base + 9]
        wall_x = obs[bot_base]

        gap_bottom = bot_y + bot_h
        gap_top = top_y
        passage_y = (gap_bottom + gap_top) / 2

        print(f"Passage {i}: wall_x={wall_x:.4f}")
        print(f"  Bottom obstacle: y={bot_y:.4f}, height={bot_h:.4f}, "
              f"y-range=[{bot_y:.4f}, {bot_y + bot_h:.4f}]")
        print(f"  Top    obstacle: y={top_y:.4f}, height={top_h:.4f}, "
              f"y-range=[{top_y:.4f}, {top_y + top_h:.4f}]")
        print(f"  Gap: [{gap_bottom:.4f}, {gap_top:.4f}], "
              f"center={passage_y:.4f}")


def save_video(frames: list[np.ndarray], path: str) -> None:
    """Strip alpha channel if present and save frames as mp4."""
    clean = [f[:, :, :3] if f.ndim == 3 and f.shape[2] == 4 else f for f in frames]

    from moviepy import ImageSequenceClip

    clip = ImageSequenceClip(clean, fps=20)
    clip.write_videofile(path, codec="libx264", logger=None)
    print(f"\nVideo saved to: {path}  ({len(clean)} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Motion2D rollout with expert.")
    parser.add_argument("--passages", type=int, default=0,
                        help="Number of wall passages (0-5, default: 1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env_id = f"kinder/Motion2D-p{args.passages}-v0"

    kinder.register_all_environments()

    env = kinder.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset(seed=args.seed)
    expert = create_motion2d_expert(env.action_space, seed=args.seed)

    print(f"Env: {env_id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Obs shape: {obs.shape}\n")
    print("=== Initial Observation ===")
    print_obs(obs)
    print()

    frames: list[np.ndarray] = []
    total_reward = 0.0

    for step in range(_MAX_STEPS):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action = expert(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            print(f"\nDone at step {step}!")
            print(f"Total reward: {total_reward:.1f}")
            print(f"Terminated: {terminated}, Truncated: {truncated}")
            break
    else:
        print(f"\nReached max steps ({_MAX_STEPS}) without termination")
        print(f"Total reward: {total_reward:.1f}")

    env.close()

    if frames:
        video_dir = Path("videos")
        video_dir.mkdir(exist_ok=True)
        save_video(
            frames,
            str(video_dir / f"motion2d_p{args.passages}_rollout.mp4"),
        )


if __name__ == "__main__":
    main()
