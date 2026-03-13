"""Quick test script for the Breakout expert policy (simple hand-coded expert + GIF)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import gymnasium as gym
import imageio.v2 as imageio  # pip install imageio

ExpertPolicy = Callable[[Any], int]


def make_breakout_expert() -> ExpertPolicy:
    """Breakout expert that:
    - Fires at the start to serve
    - Uses frame differencing to estimate ball x-position
    - Moves paddle to stay under ball
    """
    NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3

    steps_since_reset = 0
    prev_gray: np.ndarray | None = None

    def reset() -> None:
        nonlocal steps_since_reset, prev_gray
        steps_since_reset = 0
        prev_gray = None

    def expert(obs: Any) -> int:
        nonlocal steps_since_reset, prev_gray
        steps_since_reset += 1

        if steps_since_reset <= 10:
            if isinstance(obs, np.ndarray) and obs.ndim == 3:
                prev_gray = obs.mean(axis=2)
            return FIRE

        if not isinstance(obs, np.ndarray) or obs.ndim != 3:
            return FIRE

        gray = obs.mean(axis=2)

        paddle_band = gray[185:205, :]
        _, xs_p = np.where(paddle_band > 0)
        paddle_x = float(xs_p.mean()) if xs_p.size > 0 else 80.0

        ball_x: float | None = None
        if prev_gray is not None and prev_gray.shape == gray.shape:
            diff = np.abs(gray - prev_gray)
            diff_roi = diff[30:180, :]
            _, xs_d = np.where(diff_roi > 10)
            if xs_d.size > 0:
                ball_x = float(xs_d.mean())

        prev_gray = gray

        if ball_x is None:
            return NOOP

        if ball_x > paddle_x + 2:
            return RIGHT
        if ball_x < paddle_x - 2:
            return LEFT
        return NOOP

    expert.reset = reset  # type: ignore[attr-defined]
    return expert


def run_breakout_expert(
    env_id: str = "ALE/Breakout-v5",
    num_episodes: int = 5,
    max_steps: int = 50_000,
    *,
    save_gif: bool = True,
    gif_path: str = "breakout_expert_episode1.gif",
    gif_fps: int = 30,
    gif_episode_index: int = 0,
    gif_frame_stride: int = 4,
) -> None:
    try:
        import ale_py
    except Exception as e:
        raise RuntimeError(
            "Failed to import ale_py. This usually means ale-py is not installed or "
            "is incompatible with your gymnasium version."
        ) from e

    gym.register_envs(ale_py)

    if env_id not in gym.registry:
        ale_keys = [k for k in gym.registry.keys() if k.startswith("ALE/")][:10]
        raise RuntimeError(
            f"{env_id} not found in gym registry.\n"
            f"First few ALE envs registered: {ale_keys}\n"
        )

    env = gym.make(env_id, render_mode="rgb_array")

    expert = make_breakout_expert()
    returns: list[float] = []
    frames: list[np.ndarray] = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        expert.reset()  # type: ignore[attr-defined]

        prev_lives = info.get("lives", None)

        total_reward = 0.0
        step = 0
        terminated = False
        truncated = False

        action_counts = np.zeros(getattr(env.action_space, "n", 4), dtype=int)

        while step < max_steps:
            action = int(expert(obs))
            action_counts[action] += 1

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

            lives = info.get("lives", None)

            # Reset expert serve logic on life loss
            if prev_lives is not None and lives is not None and lives < prev_lives:
                expert.reset()  # type: ignore[attr-defined]

            # Episodic-life handling:
            # - If there are still lives, don't end the episode just because a life was lost.
            if lives is not None and lives > 0:
                terminated = False

            # - If lives are truly zero, force termination.
            if lives is not None and lives == 0:
                terminated = True

            prev_lives = lives

            if save_gif and ep == gif_episode_index and (step % gif_frame_stride == 0):
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            step += 1

            if terminated or truncated:
                break

        returns.append(total_reward)
        print(
            f"Episode {ep + 1}: return = {total_reward} | action_counts = {action_counts.tolist()}"
        )

    env.close()

    if save_gif and frames:
        imageio.mimsave(gif_path, frames, fps=gif_fps)
        print(f"\nSaved GIF to {gif_path} ({len(frames)} frames, fps={gif_fps})")

    print("\nSummary:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Mean return: {np.mean(returns):.2f}")
    print(f"  Std return:  {np.std(returns):.2f}")
    print(f"  Min / Max:   {np.min(returns)} / {np.max(returns)}")


if __name__ == "__main__":
    run_breakout_expert(
        env_id="ALE/Breakout-v5",
        num_episodes=10,
        max_steps=50_000,
        save_gif=True,
        gif_path="breakout_expert_episode1.gif",
        gif_fps=30,
        gif_episode_index=0,
        gif_frame_stride=4,
    )
