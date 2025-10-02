"""Tests for random_actions.py."""

import gymnasium
import imageio

from programmatic_policy_learning.approaches.pendulum_stupid_approach import (
    PendulumStupidAlgorithm,
)


def test_pendulum_stupid_algorithm():
    """Tests for RandomActionsApproach()."""
    # Just test that this runs without crashing.
    env = gymnasium.make("Pendulum-v1", render_mode="rgb_array")
    approach = PendulumStupidAlgorithm(
        "N/A", env.observation_space, env.action_space, seed=123
    )
    frames = []
    obs, info = env.reset()
    approach.reset(obs, info)
    total_reward = 0
    for _ in range(500):
        action = approach.step()
        obs, reward, terminated, _, info = env.step(action)
        total_reward += reward
        approach.update(obs, reward, terminated, info)
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    print(f"Average reward: {total_reward/500:.2f}")

    env.close()
    imageio.mimsave("pendulum.gif", frames, duration=33, loop=0)
