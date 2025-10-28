"""Tests for pendulum_experts.py."""

import gymnasium as gym

from programmatic_policy_learning.approaches.experts.pendulum_experts import (
    PendulumParametricPolicy,
)


def test_pendulum_parametric_policy() -> None:
    """Tests for PendulumParametricPolicy()."""
    # Create the environment.
    env = gym.make("Pendulum-v1", render_mode="rgb_array")

    # Create the policy with reasonable parameters.
    policy = PendulumParametricPolicy(
        init_params={"kp": 8.0, "kd": 2.0},
        param_bounds={
            "kp": (0.0, 100.0),
            "kd": (0.0, 100.0),
        },
    )

    # Run the policy and make sure it doesn't crash.
    obs, _ = env.reset(seed=123)
    for _ in range(5):
        act = policy.act(obs)
        obs, _, _, _, _ = env.step(act)
