"""Test file for grid_search_approach."""

import gymnasium as gym

from programmatic_policy_learning.approaches.experts.pendulum_experts import (
    PendulumParametricPolicy,
)
from programmatic_policy_learning.approaches.grid_search_approach import (
    evaluate_policy,
    grid_search_param,
)


def env() -> gym.Env:
    """Return the environment passed into the grid search function."""
    return gym.make("Pendulum-v1", render_mode="rgb_array")


def policy_builder(**params: float) -> PendulumParametricPolicy:
    """Return the policy passed into the grid search function."""
    return PendulumParametricPolicy(
        init_params=dict(params),
        param_bounds={k: (0.0, 100.0) for k in params},
    )


def test_grid_search() -> None:
    """Actually testing and asserting whether or not the grid search function
    is working."""
    steps = 200
    episodes = 4
    seed_base = 123

    baseline_kp = 6.0
    baseline_avg = evaluate_policy(
        p=lambda: policy_builder(kp=baseline_kp, kd=2.0),
        environment_factory=env,
        steps=steps,
        episodes=episodes,
        seed_base=seed_base,
    )

    best_kp, tuned_avg = grid_search_param(
        policy_builder=policy_builder,
        param_name="kp",
        param_bounds=(4.0, 14.0),
        env=env,
        steps=steps,
        episodes=episodes,
        num_points=5,
        kd=2.0,
    )

    assert tuned_avg >= baseline_avg
    assert 4.0 <= best_kp <= 14.0
