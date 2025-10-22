"""Tests for pendulum_stupid_approach on Pendulum-v1."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import Env
from numpy.typing import NDArray

from programmatic_policy_learning.approaches.pendulum_stupid_approach import (
    PendulumStupidAlgorithm,
)

Obs = NDArray[np.float32]
Act = NDArray[np.float32]


def test_pendulum_stupid_algorithm() -> None:
    """Runs for 500 steps and writes a short GIF."""
    env: Env[Obs, Act] = gym.make("Pendulum-v1", render_mode="rgb_array")
    frames: list[NDArray[np.uint8]] = []

    approach = PendulumStupidAlgorithm(
        "N/A", env.observation_space, env.action_space, seed=123
    )
    obs, info = env.reset()
    approach.reset(obs, info)

    total_reward = 0.0
    # Run for 500 steps and add to the total reward
    for _ in range(100):
        action = approach.step()
        obs, reward, terminated, truncated, step_info = env.step(action)

        r = float(reward)
        total_reward += r
        approach.update(obs, r, terminated or truncated, step_info)

        frame: Any = env.render()
        if isinstance(frame, np.ndarray):
            frames.append(
                frame.astype(np.uint8, copy=False) if frame.dtype != np.uint8 else frame
            )

        if terminated or truncated:
            obs, info = env.reset()

        avg_reward = total_reward / 500
        assert np.isfinite(avg_reward), "Average reward is not finite."
        assert (
            -20.0 < avg_reward <= 0.0
        ), f"Average reward out of expected range: {avg_reward:.2f}"

    # Uncomment to make a GIF
    # import imageio.v3 as iio
    # if frames:
    #     gif_array: NDArray[np.uint8] = np.stack(frames, axis=0)
    #     iio.imwrite("pendulum.gif", gif_array, duration=0.033, loop=0)
