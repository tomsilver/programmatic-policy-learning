"""Generic evaluate + 1D grid search for a single policy parameter."""

from typing import Any, Callable, Tuple

import numpy as np


def evaluate_policy(
    p: Callable[[], Any],
    environment: Callable[[], Any],
    steps: int = 300,
    episodes: int = 5,
    seed_base: int = 0,
) -> float:
    """Run several episodes and return the average total reward (env-
    agnostic)."""
    totals = []
    for i in range(int(episodes)):
        env = environment()
        obs, info = env.reset(seed=int(seed_base + i))
        policy = p()
        total = 0.0
        for _ in range(int(steps)):
            act = policy.act(obs)
            obs, rew, terminated, truncated, info = env.step(act)
            total += float(rew)
            if hasattr(policy, "update"):
                policy.update(obs, float(rew), bool(terminated or truncated), info)
            if terminated or truncated:
                break
        totals.append(total)
        env.close()
    return float(np.mean(totals))


def grid_search_param(
    policy_builder: Callable[..., Any],
    param_name: str,
    param_bounds: Tuple[float, float],
    env: Callable[[], Any],
    steps: int = 300,
    episodes: int = 5,
    num_points: int = 9,
    **fixed_params: Any,
) -> Tuple[float, float]:
    """Grid-search a single scalar parameter for ANY env + ANY policy."""
    low, high = float(param_bounds[0]), float(param_bounds[1])
    candidates = np.linspace(low, high, int(num_points))

    best_val, best_ret = None, float("-inf")

    for val in candidates:

        def policytemp(v: float = float(val)) -> Any:
            """Add in the specific value for this grid search into the param
            details."""
            params = dict(fixed_params)
            params[param_name] = float(v)
            return policy_builder(**params)

        avg_ret = evaluate_policy(
            p=policytemp,
            environment=env,
            steps=steps,
            episodes=episodes,
            seed_base=0,
        )
        print(f"{param_name}={val:.4g} -> avg_return={avg_ret:.4g}")

        if avg_ret > best_ret:
            best_val, best_ret = float(val), float(avg_ret)

    assert best_val is not None, "grid_search_param: no candidates to evaluate"
    print(f"Best {param_name}={best_val:.4g} | avg_return={best_ret:.4g}")
    best_val_f: float = best_val
    return float(best_val_f), float(best_ret)
