"""Tests for random_actions.py."""

from typing import Any, cast

import gymnasium
import imageio
import numpy as np

from programmatic_policy_learning.approaches.pendulum_stupid_algorithm import (
    PendulumStupidAlgorithm,
)


def test_pendulum_stupid_algorithm():
    """Tests for RandomActionsApproach()."""
    env = gymnasium.make("Pendulum-v1", render_mode="rgb_array")
    approach = PendulumStupidAlgorithm(
        "N/A", env.observation_space, env.action_space, seed=123
    )

    frames: list[np.ndarray] = []
    obs, info = env.reset()
    approach.reset(obs, info)

    total_reward: float = 0.0
    for _ in range(500):
        action = approach.step()
        obs, reward, terminated, _, info = env.step(action)
        total_reward += float(reward)
        approach.update(obs, float(reward), bool(terminated), info)

        frame_obj: np.ndarray | list[np.ndarray] | None = env.render()
        if isinstance(frame_obj, list):
            frames.extend(cast(list[np.ndarray], frame_obj))
        elif isinstance(frame_obj, np.ndarray):
            frames.append(frame_obj)

    print(f"Average reward: {total_reward/500:.2f}")

    env.close()  # type: ignore[no-untyped-call]
    imageio.mimsave("pendulum.gif", cast(list[Any], frames), duration=1.0 / 30.0)
