"""LLM + 1D grid search.

This program ties together:
  - synthesize_llm_parametric_policy (single prompt → parametric policy)
  - grid_search_param (generic 1D grid search)

And it returns the tuned policy instance (recreated with the best param), the
best parameter value, and the average return at that value.
"""

from __future__ import annotations

from typing import Any, Callable

from gymnasium.spaces import Box
from prpl_llm_utils.models import PretrainedLargeModel

from programmatic_policy_learning.approaches.grid_search_approach import (
    grid_search_param,
)
from programmatic_policy_learning.approaches.llm_single_search_approach import (
    synthesize_llm_parametric_policy,
)
from programmatic_policy_learning.approaches.structs import ParametricPolicyBase


def synthesize_and_grid_search(
    env_factory: Callable[[], Any],
    environment_description: str,
    action_space: Box,
    llm: PretrainedLargeModel,
    example_observation: Any,
    param_name: str,
    param_bounds: tuple[float, float] = (0.0, 20.0),
    num_points: int = 9,
    steps: int = 300,
    episodes: int = 5,
    *,
    param_bounds_all: dict[str, tuple[float, float]],
    fixed_params: dict[str, float] | None = None,
    init_params: dict[str, float] | None = None,
) -> tuple[ParametricPolicyBase, float, float]:
    """Synthesize a parametric policy via one LLM prompt, then 1D grid search.

    Returns:
      tuned_policy, best_param_value, best_avg_return
    """
    mid_init = {k: (lo + hi) / 2.0 for k, (lo, hi) in param_bounds_all.items()}

    if init_params is None:
        init_params = dict(mid_init)
    else:
        for k, v in mid_init.items():
            init_params.setdefault(k, v)

    def _clip(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    for k, (lo, hi) in param_bounds_all.items():
        init_params[k] = _clip(float(init_params[k]), lo, hi)

    fixed_params = dict(fixed_params or {})

    base_policy = synthesize_llm_parametric_policy(
        environment_description=environment_description,
        action_space=action_space,
        llm=llm,
        example_observation=example_observation,
        init_params=init_params,
        param_bounds=param_bounds_all,
    )

    def policy_builder(**params: float) -> ParametricPolicyBase:
        """Used by grid_search_param to create a fresh policy instance."""
        merged = dict(fixed_params)
        merged.update(params)
        policy = base_policy
        policy.set_params(merged)
        return policy

    best_val, best_ret = grid_search_param(
        policy_builder=policy_builder,
        param_name=param_name,
        param_bounds=param_bounds,
        env=env_factory,
        steps=steps,
        episodes=episodes,
        num_points=num_points,
        **fixed_params,
    )
    tuned_params = dict(fixed_params)
    tuned_params[param_name] = float(best_val)
    tuned_policy = policy_builder(**tuned_params)

    return tuned_policy, float(best_val), float(best_ret)
