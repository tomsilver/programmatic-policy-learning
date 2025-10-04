"""Tests for pendulum_stupid_approach on Pendulum-v1."""

from typing import Any, cast

import gymnasium as gym
import imageio
import numpy as np

from programmatic_policy_learning.approaches.pendulum_stupid_approach import (
    PendulumStupidAlgorithm,
)


def test_pendulum_stupid_algorithm() -> None:
    """Runs for 500 steps and writes a short GIF."""
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    approach = PendulumStupidAlgorithm(
        "N/A", env.observation_space, env.action_space, seed=123
    )

    frames: list[np.ndarray] = []
    obs, info = env.reset()
    approach.reset(obs, info)

    total_reward: float = 0.0
    # Run for 500 steps and add to the total reward
    for _ in range(500):
        action = approach.step()
        obs, reward, terminated, _, step_info = env.step(action)

        r = float(reward)
        total_reward += r
        approach.update(obs, r, terminated, step_info)

        frame: np.ndarray | None = cast(np.ndarray | None, env.render())
        if frame is not None:
            frames.append(np.asarray(frame))

    print(f"Average reward: {total_reward / 500.0:.2f}")

    cast(Any, env).close()

    imageio.mimsave(
        uri="pendulum.gif",
        ims=cast(list[Any], frames),
        duration=33,
        loop=0,
    )
