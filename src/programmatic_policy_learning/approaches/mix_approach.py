"""An approach that mixes ProgrammaticPolicyLearningApproach and
RandomActionsApproach."""

from typing import TypeVar

import numpy as np
from gymnasium.spaces import Space
from prpl_llm_utils.models import PretrainedLargeModel

from programmatic_policy_learning.approaches.base_approach import BaseApproach
from programmatic_policy_learning.approaches.ppl_approach import (
    ProgrammaticPolicyLearningApproach,
)
from programmatic_policy_learning.approaches.random_actions import RandomActionsApproach

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")


class MixApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that mixes ProgrammaticPolicyLearningApproach and
    RandomActionsApproach."""

    def __init__(
        self,
        environment_description: str,
        observation_space: Space[_ObsType],
        action_space: Space[_ActType],
        seed: int,
        llm: PretrainedLargeModel,
    ) -> None:
        super().__init__(environment_description, observation_space, action_space, seed)
        self._ppl_approach = ProgrammaticPolicyLearningApproach(
            environment_description, observation_space, action_space, seed, llm
        )
        self._random_approach = RandomActionsApproach(
            environment_description, observation_space, action_space, seed
        )
        self._rng = np.random.default_rng(seed)

    def reset(self, *args, **kwargs) -> None:
        super().reset(*args, **kwargs)
        self._ppl_approach.reset(*args, **kwargs)
        self._random_approach.reset(*args, **kwargs)

    def _get_action(self) -> _ActType:
        if self._rng.random() < 0.5:
            return self._ppl_approach.step()
        return self._random_approach.step()
