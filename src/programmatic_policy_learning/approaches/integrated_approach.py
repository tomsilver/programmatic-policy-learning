"""An approach that synthesizes a programmatic policy using an LLM with access
to a planner."""

import logging
from typing import Any, Callable, List, TypeVar

from gymnasium.spaces import Space
from prpl_llm_utils.code import (
    FunctionOutputRepromptCheck,
    SyntaxRepromptCheck,
    synthesize_python_function_with_llm,
)
from prpl_llm_utils.models import PretrainedLargeModel
from prpl_llm_utils.structs import Query

from programmatic_policy_learning.approaches.base_approach import BaseApproach

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")
_State = tuple[int, int]


# Planner documentation to be included in the LLM prompt
PLANNER_DOC = '''
    from prpl_utils.search import run_astar, SearchMetrics

    def run_astar(
        initial_state,
        check_goal,
        get_successors,
        heuristic,
        max_expansions: int = 10000000,
        max_evals: int = 10000000,
        timeout: float = 10000000,
        lazy_expansion: bool = False,
    ) -> tuple[list, list, SearchMetrics]:
        """A* search algorithm.

    Args:
        initial_state: The starting state (must be hashable, e.g., a tuple).
        check_goal: A function that takes a state and returns True if it is a goal state.
        get_successors: A function that takes a state and yields (action, next_state, cost) tuples.
        heuristic: A function that takes a state and returns an estimated cost to the goal.
        max_expansions: Maximum number of node expansions before stopping.
        max_evals: Maximum number of priority function evaluations before stopping.
        timeout: Maximum time in seconds before stopping.
        lazy_expansion: If True, immediately explore a better child without finishing siblings.

    Returns:
        A tuple of (state_sequence, action_sequence, search_metrics).
        - state_sequence: List of states from initial to goal.
        - action_sequence: List of actions taken to reach the goal.
        - search_metrics: A SearchMetrics object with num_evals and num_expansions.

    Example usage:
        def get_successors(state):
            for action in get_actions():
                next_state = get_next_state(state, action)
                if next_state != state:  # Valid move (not blocked)
                    yield (action, next_state, 1.0)

        def heuristic(state):
            return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

        states, actions, metrics = run_astar(
            initial_state=(0, 0),
            check_goal=lambda s: s == goal,
            get_successors=get_successors,
            heuristic=heuristic,
        )
    """
'''


class IntegratedApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that synthesizes a programmatic policy using an LLM with
    planner access."""

    def __init__(
        self,
        environment_description: str,
        observation_space: Space[_ObsType],
        action_space: Space[_ActType],
        seed: int,
        llm: PretrainedLargeModel,
        get_actions: Callable[[], List[_ActType]],
        get_next_state: Callable[[_State, _ActType], _State],
        get_cost: Callable[[], float],
        check_goal: Callable[[_State, Any], bool],
    ) -> None:
        super().__init__(environment_description, observation_space, action_space, seed)
        self._llm = llm
        self._get_actions = get_actions
        self._get_next_state = get_next_state
        self._get_cost = get_cost
        self._check_goal = check_goal
        # Wait to reset so that we have one example of an observation.
        self._policy: Callable[..., _ActType] | None = None

    def reset(self, *args: Any, **kwargs: Any) -> None:
        super().reset(*args, **kwargs)
        assert self._last_observation is not None
        if self._policy is None:
            self._policy = synthesize_policy_with_planner_access(
                self._environment_description,
                self._llm,
                self._last_observation,
                self._action_space,
                self._get_actions,
                self._get_next_state,
                self._get_cost,
                self._check_goal,
            )

    def _get_action(self) -> _ActType:
        assert self._policy is not None, "Call reset() first."
        assert self._last_observation is not None
        return self._policy(
            self._last_observation,
            self._get_actions,
            self._get_next_state,
            self._get_cost,
            self._check_goal,
        )


def synthesize_policy_with_planner_access(
    environment_description: str,
    llm: PretrainedLargeModel,
    example_observation: _ObsType,
    action_space: Space[_ActType],
    get_actions: Callable[[], List[_ActType]],
    get_next_state: Callable[[_State, _ActType], _State],
    get_cost: Callable[[], float],
    check_goal: Callable[[_State, Any], bool],
) -> Callable[..., _ActType]:
    """Use the LLM to synthesize a programmatic policy with access to a
    planner."""

    function_name = "_policy"
    query = Query(f"""Generate a Python function policy of the form

        ```python
        def _policy(obs, get_actions, get_next_state, get_cost, check_goal):
            from prpl_utils.search import run_astar  # import INSIDE the function
            # your code here
        ```

        The policy is called once per timestep and must return a single valid action
        (an integer from the action space). It must NEVER return None.

        ## Arguments

        - `obs`: The current observation, a tuple like `(row, col)`.
        - `get_actions()`: Returns a list of all possible actions (integers).
        - `get_next_state(state, action)`: Returns the resulting state after taking an
        action from a state. If the action is invalid (e.g., blocked by a wall), it
        returns the same state unchanged. States are tuples like `(row, col)`.
        - `get_cost()`: Returns the cost of a single action (a float).
        - `check_goal(state, goal)`: Returns True if the state matches the goal.

        ## A* Planner

        You have access to an A* search planner. IMPORTANT: You must import it inside
        the function body, not at the top of the file:

        ```python
        from prpl_utils.search import run_astar
        ```

        {PLANNER_DOC}

        To build a `get_successors` function for A*, use the provided `get_actions`,
        `get_next_state`, and `get_cost` arguments. Here is a complete example of
        calling A* inside your policy:

        ```python
        def _policy(obs, get_actions, get_next_state, get_cost, check_goal):
            from prpl_utils.search import run_astar

            goal = (14, 14)

            # Use cached plan if available AND we are at the expected state.
            # A plan is a sequence of actions from a specific state. If the
            # agent is not at the expected state (e.g., because reactive control
            # was used before switching to planning), the cached plan is invalid.
            if hasattr(_policy, 'plan') and _policy.plan and _policy.expected_obs == obs:
                action = _policy.plan.pop(0)
                _policy.expected_obs = get_next_state(obs, action)
                return action

            def get_successors(state):
                for action in get_actions():
                    next_state = get_next_state(state, action)
                    if next_state != state:  # Valid move (not blocked by wall)
                        yield (action, next_state, get_cost())

            def heuristic(state):
                return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

            _, actions, metrics = run_astar(
                initial_state=obs,
                check_goal=lambda s: check_goal(s, goal),
                get_successors=get_successors,
                heuristic=heuristic,
            )
            _policy.plan = list(actions)

            # Store search metrics on the function for external access
            if not hasattr(_policy, 'total_num_evals'):
                _policy.total_num_evals = 0
                _policy.total_num_expansions = 0
            _policy.total_num_evals += metrics.num_evals
            _policy.total_num_expansions += metrics.num_expansions

            if _policy.plan:
                action = _policy.plan.pop(0)
                _policy.expected_obs = get_next_state(obs, action)
                return action
            return get_actions()[0]
        ```

        ## Environment Description

        {environment_description}

        ## Action Space

        {action_space}

        ## Example Observation

        {example_observation}

        ## Instructions

        Your policy should be both effective (reach the goal) and efficient (minimize
        unnecessary computation). Consider the following:

        Your policy should be both effective (reach the goal) and efficient (minimize
        unnecessary computation). Consider the following:

        - A reactive policy (simple rules based on observation and goal) is computationally
            cheap and should be preferred when the path to the goal is straightforward, e.g.,
            when there are no obstacles or constraints between the current state and the goal.
        - The A* planner is powerful but expensive. It should only be invoked when the
            situation genuinely requires search, e.g., when there are obstacles, constraints,
            or complex structure that a simple reactive rule cannot handle.
        - A good policy uses reactive control when it can and only falls back to planning
            when it must. Think about what conditions in the observation indicate that
            planning is necessary versus when a simple rule suffices.
        - IMPORTANT: If you cache a plan, always track the expected next observation
            alongside it (see the example above). A cached plan is only valid if the
            current obs matches where the plan expects you to be. If you use reactive
            control for some steps and then switch to a cached plan, the plan will be
            wrong because it was computed from a different state. Either replan from the
            current obs, or only cache a plan when you will follow it without interruption.
        - The function must ALWAYS return a valid action integer. Never return None.
        - When you call run_astar, store cumulative search metrics on the function
            so they can be inspected externally. See the example above for how to
            accumulate `_policy.total_num_evals` and `_policy.total_num_expansions`.

        Return only the function; do not give example usages.
        """)
    # Insert instructions for different approaches here. For example:

    # Pure Planning: Always Plans
    # Use the A* planner with get_next_state for the entire journey. The
    # get_next_state function already handles all walls and borders correctly
    # (returning the same state if a move is blocked), so A* will find a
    # valid path from any starting position through the entrance and maze
    # to the goal. Plan once and cache the result.

    # Hybrid Reactive + Planning: Plan When Necessary (or at least when LLM thinks it's necessary)
    # Your policy should be both effective (reach the goal) and efficient (minimize
    # unnecessary computation). Consider the following:

    # - A reactive policy (simple rules based on observation and goal) is computationally
    #     cheap and should be preferred when the path to the goal is straightforward, e.g.,
    #     when there are no obstacles or constraints between the current state and the goal.
    # - The A* planner is powerful but expensive. It should only be invoked when the
    #     situation genuinely requires search, e.g., when there are obstacles, constraints,
    #     or complex structure that a simple reactive rule cannot handle.
    # - A good policy uses reactive control when it can and only falls back to planning
    #     when it must. Think about what conditions in the observation indicate that
    #     planning is necessary versus when a simple rule suffices.
    # - IMPORTANT: If you cache a plan, always track the expected next observation
    #     alongside it (see the example above). A cached plan is only valid if the
    #     current obs matches where the plan expects you to be. If you use reactive
    #     control for some steps and then switch to a cached plan, the plan will be
    #     wrong because it was computed from a different state. Either replan from the
    #     current obs, or only cache a plan when you will follow it without interruption.
    # - The function must ALWAYS return a valid action integer. Never return None.
    # - When you call run_astar, store cumulative search metrics on the function
    #     so they can be inspected externally. See the example above for how to
    #     accumulate `_policy.total_num_evals` and `_policy.total_num_expansions`.

    # For reprompt validation, call with the actual env functions
    reprompt_checks = [
        SyntaxRepromptCheck(),
        FunctionOutputRepromptCheck(
            function_name,
            [(example_observation, get_actions, get_next_state, get_cost, check_goal)],
            [action_space.contains],
        ),
    ]

    policy = synthesize_python_function_with_llm(
        function_name,
        llm,
        query,
        reprompt_checks=reprompt_checks,
    )

    logging.info("Synthesized new policy with planner access:")
    logging.info(str(policy))

    return policy
