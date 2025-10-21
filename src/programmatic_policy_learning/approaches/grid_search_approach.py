"""Generic utilities for evaluating a policy and performing 1D grid search
optimization.

This module provides two core functions:

1. `evaluate_policy`: Runs a given policy across multiple episodes in any
   environment and returns the average total reward.

2. `grid_search_param`: Performs a one-dimensional grid search over a single
   scalar policy parameter (e.g., a gain). For each
   candidate parameter value, it evaluates the resulting policy and records
   its average return also returning the best-performing parameter and its
   corresponding average reward.
"""

from typing import Any, Callable

import numpy as np

from programmatic_policy_learning.approaches.structs import ParametricPolicyBase


def evaluate_policy(
    p: Callable[[], ParametricPolicyBase],
    environment_factory: Callable[[], Any],
    steps: int = 300,
    episodes: int = 5,
    seed_base: int = 0,
) -> float:
    """Run several episodes and return the average total reward (env-
    agnostic)."""
    totals = []
    for i in range(int(episodes)):
        env = environment_factory()
        obs, _ = env.reset(seed=int(seed_base + i))
        policy = p()
        total = 0.0
        for _ in range(int(steps)):
            act = policy.act(obs)
            obs, rew, terminated, truncated, _ = env.step(act)
            total += float(rew)
            if terminated or truncated:
                break
        totals.append(total)
        env.close()
    return float(np.mean(totals))


def grid_search_param(
    policy_builder: Callable[..., ParametricPolicyBase],
    param_name: str,
    param_bounds: tuple[float, float],
    env: Callable[[], Any],
    steps: int = 300,
    episodes: int = 5,
    num_points: int = 9,
    **fixed_params: Any,
) -> tuple[float, float]:
    """Grid-search a single scalar parameter for ANY env + ANY policy."""
    low, high = float(param_bounds[0]), float(param_bounds[1])
    candidates = np.linspace(low, high, int(num_points))

    best_val, best_ret = None, float("-inf")

    for val in candidates:

        def policytemp(v: float = float(val)) -> ParametricPolicyBase:
            """Add in the specific value for this grid search into the param
            details."""
            params = dict(fixed_params)
            params[param_name] = float(v)
            return policy_builder(**params)

        avg_ret = evaluate_policy(
            p=policytemp,
            environment_factory=env,
            steps=steps,
            episodes=episodes,
            seed_base=0,
        )
        print(f"{param_name}={val:.4g} -> avg_return={avg_ret:.4g}")

        if avg_ret > best_ret:
            best_val, best_ret = float(val), float(avg_ret)

    assert best_val is not None, "grid_search_param: no candidates to evaluate"
    best_val_f: float = best_val
    return float(best_val_f), float(best_ret)
