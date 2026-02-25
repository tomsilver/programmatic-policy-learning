"""An approach that uses an LLM to generate a class-based policy with planner
access.

Unlike IntegratedApproach (which uses SynthesizedPythonFunction and runs
each call in an isolated subprocess), this approach generates a full
class that is loaded in-process via exec/compile.  This means the
generated class can maintain state across calls (cached plans,
accumulated search metrics, etc.).
"""

import logging
from typing import Any, Callable, List, TypeVar

from gymnasium.spaces import Space
from prpl_llm_utils.code import SyntaxRepromptCheck, parse_python_code_from_text
from prpl_llm_utils.models import PretrainedLargeModel
from prpl_llm_utils.reprompting import query_with_reprompts
from prpl_llm_utils.structs import Query

from programmatic_policy_learning.approaches.base_approach import BaseApproach

logger = logging.getLogger(__name__)

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")
_State = tuple[int, int]

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


class AgenticIntegratedApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that uses an LLM to generate a class-based policy with
    planner access.

    The LLM generates a `GeneratedPolicy` class with `__init__`, `reset`, and
    `get_action` methods.  The class is loaded in-process so it can maintain
    state across calls (e.g. cached A* plans and search metrics).
    """

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
        self._generated: Any = None

    def reset(self, *args: Any, **kwargs: Any) -> None:
        super().reset(*args, **kwargs)
        assert self._last_observation is not None
        if self._generated is None:
            code_str = _synthesize_policy_class(
                self._environment_description,
                self._llm,
                self._last_observation,
                self._action_space,
            )
            self._generated = _load_generated_policy(
                code_str,
                self._get_actions,
                self._get_next_state,
                self._get_cost,
                self._check_goal,
            )
            self._generated.reset(self._last_observation)

    def _get_action(self) -> _ActType:
        assert self._generated is not None, "Call reset() first."
        assert self._last_observation is not None
        return self._generated.get_action(self._last_observation)


def _load_generated_policy(
    code_str: str,
    get_actions: Callable,
    get_next_state: Callable,
    get_cost: Callable,
    check_goal: Callable,
) -> Any:
    """Load a GeneratedPolicy class from a code string and instantiate it."""
    namespace: dict[str, Any] = {}
    exec(compile(code_str, "<generated_policy>", "exec"), namespace)  # noqa: S102
    cls = namespace["GeneratedPolicy"]
    return cls(
        get_actions=get_actions,
        get_next_state=get_next_state,
        get_cost=get_cost,
        check_goal=check_goal,
    )


def _synthesize_policy_class(
    environment_description: str,
    llm: PretrainedLargeModel,
    example_observation: _ObsType,
    action_space: Space[_ActType],
) -> str:
    """Use the LLM to synthesize a GeneratedPolicy class and return the code
    string."""

    query = Query(
        f"""Generate a Python file containing a class `GeneratedPolicy` with the
        following interface:

        ```python
        class GeneratedPolicy:
            def __init__(self, get_actions, get_next_state, get_cost, check_goal):
                \"\"\"Initialize with environment transition functions.

                Args:
                    get_actions(): Returns a list of all possible actions (integers).
                    get_next_state(state, action): Returns the resulting state after
                        taking an action. If blocked (e.g. by a wall), returns the
                        same state unchanged. States are tuples like (row, col).
                    get_cost(): Returns the cost of a single action (a float).
                    check_goal(state, goal): Returns True if the state matches the goal.
                \"\"\"
                ...

            def reset(self, obs):
                \"\"\"Called at the start of each episode with the initial observation.\"\"\"
                ...

            def get_action(self, obs):
                \"\"\"Return a valid action for the given observation.

                Called once per timestep. Must ALWAYS return a valid action integer.
                Must NEVER return None.
                \"\"\"
                ...
        ```

        The class can maintain internal state between calls (e.g. a cached A* plan,
        accumulated search metrics, etc.). The `reset` method is called once at the
        start of each episode. The `get_action` method is called each step.

        ## A* Planner

        You have access to an A* search planner. Import it at the top of the file:

        ```python
        from prpl_utils.search import run_astar, SearchMetrics
        ```

        {PLANNER_DOC}

        To build a `get_successors` function for A*, use the `get_actions`,
        `get_next_state`, and `get_cost` stored in `self`:

        ```python
        def get_successors(state):
            for action in self.get_actions():
                next_state = self.get_next_state(state, action)
                if next_state != state:  # Valid move (not blocked by wall)
                    yield (action, next_state, self.get_cost())
        ```

        ## Search Metrics

        When you call run_astar, accumulate the search metrics on self so they can
        be inspected externally:

        ```python
        _, actions, metrics = run_astar(...)
        self.total_num_evals += metrics.num_evals
        self.total_num_expansions += metrics.num_expansions
        ```

        Initialize `self.total_num_evals = 0` and `self.total_num_expansions = 0`
        in `__init__`.

        ## Plan Caching

        Since this is a class, you can cache plans across calls using self:

        ```python
        # In __init__:
        self.plan = []
        self.expected_obs = None

        # In get_action:
        if self.plan and self.expected_obs == obs:
            action = self.plan.pop(0)
            self.expected_obs = self.get_next_state(obs, action)
            return action
        ```

        IMPORTANT: A cached plan is only valid if the current obs matches where the
        plan expects you to be. If you use reactive control for some steps and then
        switch to a cached plan, the plan will be wrong because it was computed from a
        different state. Either replan from the current obs, or only cache a plan when
        you will follow it without interruption.

        ## Environment Description

        {environment_description}

        ## Action Space

        {action_space}

        ## Example Observation

        {example_observation}

        ## Instructions

        Your policy should be both effective (reach the goal) and efficient (minimize
        unnecessary computation). Consider the following:

        - A reactive policy (simple rules based on observation and goal) is computationally
        cheap and should be preferred when the path to the goal is straightforward, e.g.,
        when there are no obstacles or constraints between the current state and the goal.
        - The A* planner is powerful but expensive. It should only be invoked when the
        situation genuinely requires search, e.g., when there are obstacles, constraints,
        or complex structure that a simple reactive rule cannot handle.
        - A good policy uses reactive control when it can and only falls back to planning
        when it must.
        - get_action must ALWAYS return a valid action integer. Never return None.

        Return only the Python file with the class definition (and any necessary imports
        at the top). Do not include example usages or test code.
        """
    )

    reprompt_checks = [SyntaxRepromptCheck()]

    response = query_with_reprompts(llm, query, reprompt_checks)
    code_str = parse_python_code_from_text(response.text)
    if code_str is None:
        raise RuntimeError("No python code found in LLM response.")

    logger.info("Synthesized GeneratedPolicy class:\n%s", code_str)
    print(code_str)

    return code_str
