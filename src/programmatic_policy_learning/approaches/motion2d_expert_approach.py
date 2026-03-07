"""ExpertApproach subclass for Motion2D that wires the action space."""

from typing import Any

import gymnasium as gym

from programmatic_policy_learning.approaches.expert_approach import ExpertApproach
from programmatic_policy_learning.approaches.experts.motion2d_experts import (
    Act,
    Obs,
    create_motion2d_expert,
)


class Motion2DExpertApproach(ExpertApproach[Obs, Act]):
    """Expert approach for Motion2D using f_1 ∧ f_2 rejection sampling.

    The rejection-sampling expert needs the action space to draw candidate
    actions, so this subclass creates the expert internally where
    ``action_space`` is available.
    """

    def __init__(
        self,
        environment_description: str,
        observation_space: Any,
        action_space: Any,
        seed: int,
    ) -> None:
        assert isinstance(action_space, gym.spaces.Box)
        expert_fn = create_motion2d_expert(action_space, seed)
        super().__init__(
            environment_description, observation_space, action_space, seed, expert_fn
        )
