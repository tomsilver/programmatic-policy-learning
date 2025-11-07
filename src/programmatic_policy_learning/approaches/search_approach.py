"""An approach that uses uniform-cost search to find a plan."""

from typing import Any, Callable, Iterator, List, Tuple

import numpy as np
from gymnasium.spaces import Space
from prpl_utils.search import run_astar

from programmatic_policy_learning.approaches.base_approach import (
    BaseApproach,
    _ActType,
    _ObsType,
)

_State = Tuple[int, int]


class SearchApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that uses A* search to find a plan."""

    def __init__(
        self,
        environment_description: str,
        observation_space: Space[_ObsType],
        action_space: Space[_ActType],
        seed: int,
        get_actions: Callable[[], List[_ActType]],
        get_next_state: Callable[[_State, _ActType], _State],
        get_cost: Callable[[], float],
        check_goal: Callable[[_State, Any], bool],
    ) -> None:
        super().__init__(environment_description, observation_space, action_space, seed)
        self._get_actions = get_actions
        self._get_next_state = get_next_state
        self._get_cost = get_cost
        self._check_goal = check_goal
        self._plan: list[_ActType] = []

    def _get_action(self) -> _ActType:
        """Return the next action from the plan."""
        if not self._plan:
            raise ValueError("Plan is empty. Ensure reset() was called.")
        return self._plan.pop(0)

    def reset(self, obs: _ObsType, info: dict) -> None:
        """Reset the approach and generate a new plan."""
        # Fetch the goal directly from the environment
        super().reset(obs, info)
        goal = info["goal"]  # Assuming `self.env` is the environment instance
        self._plan = self._generate_plan(np.array(obs), goal)

    def _generate_plan(self, start: np.ndarray, goal: np.ndarray) -> list[_ActType]:
        """Generate a plan using A* search."""

        def get_successors(
            state: Tuple[int, int],
        ) -> Iterator[Tuple[_ActType, Tuple[int, int], float]]:
            """Generate successors for the current state."""
            for action in self._get_actions():
                next_state = self._get_next_state(state, action)
                if not np.array_equal(next_state, state):  # Valid transition
                    cost = self._get_cost()
                    yield (action, next_state, cost)

        _, plan = run_astar(
            initial_state=tuple(start),
            check_goal=lambda state: self._check_goal(state, goal),
            get_successors=get_successors,
            heuristic=lambda s: abs(s[0] - goal[0]) + abs(s[1] - goal[1]),
        )
        return plan
