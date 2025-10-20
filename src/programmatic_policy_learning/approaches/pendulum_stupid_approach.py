"""A simple hardcoded approach for pendulum balancing."""

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from programmatic_policy_learning.approaches.expert_approach import ExpertApproach
from programmatic_policy_learning.approaches.experts.pendulum_experts import (
    create_manual_pendulum_policy,
)

Obs = NDArray[np.float32]
Act = NDArray[np.float32]


class PendulumStupidAlgorithm(ExpertApproach[Obs, Act]):
    """A hardcoded approach that tries to balance the pendulum at the top."""

    def __init__(
        self,
        environment_description: str,
        observation_space: Any,
        action_space: Any,
        seed: int,
    ) -> None:
        assert isinstance(action_space, gym.spaces.Box)
        expert_fn = create_manual_pendulum_policy(action_space)
        super().__init__(
            environment_description, observation_space, action_space, seed, expert_fn
        )
