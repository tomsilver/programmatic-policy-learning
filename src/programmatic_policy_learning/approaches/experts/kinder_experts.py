"""Expert policies for KinDER environments.

Dispatches to the concrete expert constructor based on ``env_name``.
Analogous to :func:`grid_experts.get_grid_expert` for grid environments.
"""

from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym

from programmatic_policy_learning.approaches.experts.motion2d_experts import (
    create_motion2d_expert,
)

# pylint: disable=line-too-long
from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based.continuous_hint_config import (
    canonicalize_env_name,
)

ExpertPolicy = Callable[[Any], Any]


def create_kinder_expert(
    env_name: str,
    action_space: gym.spaces.Box,
    seed: int = 0,
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
        input("INJAAAAAAAA")
        return create_motion2d_expert(action_space, seed=seed)
    raise KeyError(
        f"No KinDER expert configured for '{env_name}'. "
        "Add a branch in create_kinder_expert()."
    )
