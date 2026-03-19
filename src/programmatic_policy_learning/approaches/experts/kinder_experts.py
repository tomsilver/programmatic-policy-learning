"""Expert policies for KinDER environments.

Dispatches to the concrete expert constructor based on ``env_name``.
Analogous to :func:`grid_experts.get_grid_expert` for grid environments.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from programmatic_policy_learning.approaches.experts.motion2d_bilevel_experts import (
    create_motion2d_bilevel_expert,
)
from programmatic_policy_learning.approaches.experts.motion2d_experts import (
    create_motion2d_expert,
)

# pylint: disable=line-too-long
from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based.continuous_hint_config import (
    canonicalize_env_name,
)

ExpertPolicy = Any


def create_kinder_expert(
    env_name: str,
    action_space: gym.spaces.Box,
    seed: int = 0,
    *,
    observation_space: Any | None = None,
    num_passages: int = 0,
    expert_kind: str = "bilevel",
    max_abstract_plans: int = 10,
    samples_per_step: int = 3,
    max_skill_horizon: int = 100,
    heuristic_name: str = "hff",
    planning_timeout: float = 60.0,
) -> ExpertPolicy:
    """Build an expert policy for the given KinDER environment.

    Parameters
    ----------
    env_name
        Environment family name (e.g. ``"Motion2D"``).
    action_space
        The continuous Box action space of the environment.
    seed
        RNG seed forwarded to the expert constructor.

    Returns
    -------
    A callable ``(obs) -> action``.
    """
    env_name = canonicalize_env_name(env_name)
    if env_name == "Motion2D":
        if expert_kind == "bilevel":
            if observation_space is None:
                raise ValueError("Motion2D bilevel expert requires observation_space.")
            return create_motion2d_bilevel_expert(
                observation_space,
                action_space,
                seed=seed,
                num_passages=num_passages,
                max_abstract_plans=max_abstract_plans,
                samples_per_step=samples_per_step,
                max_skill_horizon=max_skill_horizon,
                heuristic_name=heuristic_name,
                planning_timeout=planning_timeout,
            )
        if expert_kind != "rejection":
            raise ValueError(
                f"Unknown Motion2D expert kind '{expert_kind}'. "
                "Expected 'rejection' or 'bilevel'."
            )
        return create_motion2d_expert(action_space, seed=seed)
    raise KeyError(
        f"No KinDER expert configured for '{env_name}'. "
        "Add a branch in create_kinder_expert()."
    )
