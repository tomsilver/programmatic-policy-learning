"""An approach that uses an LLM to generate a class-based policy with planner
access."""

import ast
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, List, TypeVar

from gymnasium.spaces import Space
from prpl_llm_utils.models import PretrainedLargeModel
from prpl_llm_utils.reprompting import (
    RepromptCheck,
    create_reprompt_from_error_message,
    query_with_reprompts,
)
from prpl_llm_utils.structs import Query, Response

from programmatic_policy_learning.approaches.base_approach import BaseApproach

logger = logging.getLogger(__name__)

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")
_State = tuple[int, int]

PLANNER_DOC = '''
    ## Planner

    You have access to an A* search planner. Import it at the top of each file:

    ```
    from prpl_utils.search import run_astar, SearchMetrics
    ```

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

    To build a `get_successors` function for A*, use the `get_actions`,
    `get_next_state`, and `get_cost` stored in `self`:

    ```
    def get_successors(state):
        for action in self.get_actions():
            next_state = self.get_next_state(state, action)
            if next_state != state:  # Valid move (not blocked)
                yield (action, next_state, self.get_cost())
    ```

    ## Plan Caching

    Since this is a class, you can cache plans across calls using self.
    `expected_obs` should be set to the CURRENT observation when a plan is
    first created, then advanced each time an action is consumed:

    ```
    # After planning:
    self.plan = list(actions)          # action sequence from planner
    self.expected_obs = obs            # obs we planned FROM (current obs)

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
'''


class AgenticIntegratedApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that uses an LLM to generate candidate policies with planner
    access, scores them via simulation, and keeps the best.

    The LLM generates `num_candidates` distinct `GeneratedPolicy` classes
    with varying planning/reactivity trade-offs. Each is scored by
    simulating an episode, and the best-scoring policy is used for
    evaluation.
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
        num_candidates: int = 7,
        scoring_max_timesteps: int = 1000,
    ) -> None:
        super().__init__(environment_description, observation_space, action_space, seed)
        self._seed = seed
        self._llm = llm
        self._get_actions = get_actions
        self._get_next_state = get_next_state
        self._get_cost = get_cost
        self._check_goal = check_goal
        self._num_candidates = num_candidates
        self._scoring_max_timesteps = scoring_max_timesteps
        self._generated: Any = None
        self._best_code: str = ""
        self._all_candidate_codes: list[str] = []
        self._all_candidate_scores: list[tuple[bool, int] | None] = []

    def update_env_callables(
        self,
        get_actions: Callable[[], List[_ActType]],
        get_next_state: Callable[[_State, _ActType], _State],
        get_cost: Callable[[], float],
        check_goal: Callable[[_State, Any], bool],
    ) -> None:
        """Update the environment-specific transition functions.

        Call this before reset() when evaluating on a new environment
        instance so the generated policy uses the correct dynamics.
        """
        self._get_actions = get_actions
        self._get_next_state = get_next_state
        self._get_cost = get_cost
        self._check_goal = check_goal

    def reset(self, *args: Any, **kwargs: Any) -> None:
        super().reset(*args, **kwargs)
        assert self._last_observation is not None
        if self._generated is None:
            assert self._last_info is not None
            goal = self._last_info["goal"]

            code_strings = _synthesize_policy_classes(
                self._environment_description,
                self._llm,
                self._last_observation,
                self._action_space,
                self._num_candidates,
                seed=self._seed,
            )

            best_code = None
            best_score: tuple[bool, int] = (False, -(2**63))
            self._all_candidate_codes = list(code_strings)
            self._all_candidate_scores = []

            for i, code_str in enumerate(code_strings):
                try:
                    policy = _load_generated_policy(
                        code_str,
                        self._get_actions,
                        self._get_next_state,
                        self._get_cost,
                        self._check_goal,
                    )
                    score = _score_policy(
                        policy,
                        self._last_observation,
                        self._last_info,
                        goal,
                        self._get_next_state,
                        self._check_goal,
                        self._scoring_max_timesteps,
                    )
                    logger.info("Policy %d score: %s", i, score)
                    print(f"Policy {i} score: {score}")
                    self._all_candidate_scores.append(score)
                    if score > best_score:
                        best_score = score
                        best_code = code_str
                except Exception as e:  # pylint: disable=broad-except
                    logger.warning("Policy %d failed to load/score: %s", i, e)
                    print(f"Policy {i} failed: {e}")
                    self._all_candidate_scores.append(None)

            if best_code is None:
                raise RuntimeError("No valid policies were generated.")

            self._best_code = best_code
            self._generated = _load_generated_policy(
                best_code,
                self._get_actions,
                self._get_next_state,
                self._get_cost,
                self._check_goal,
            )

        self._generated.get_actions = self._get_actions
        self._generated.get_next_state = self._get_next_state
        self._generated.get_cost = self._get_cost
        self._generated.check_goal = self._check_goal
        self._generated.reset(self._last_observation, self._last_info)

    def _get_action(self) -> _ActType:
        assert self._generated is not None, "Call reset() first."
        assert self._last_observation is not None
        return self._generated.get_action(self._last_observation)


def _read_and_clear_astar_metrics() -> int:
    """Read total expansions from the astar metrics JSON file, then clear it.

    Returns 0 if the env var is unset or the file doesn't exist.
    """
    metrics_path_str = os.environ.get("astar_metrics_path")
    if not metrics_path_str:
        return 0
    path = Path(metrics_path_str)
    if not path.exists():
        return 0
    total_expansions = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                total_expansions += json.loads(line)["num_expansions"]
    path.write_text("", encoding="utf-8")
    return total_expansions


def _score_policy(
    policy: Any,
    initial_obs: Any,
    initial_info: Any,
    goal: Any,
    get_next_state: Callable,
    check_goal: Callable,
    max_timesteps: int,
) -> tuple[bool, int]:
    """Score a policy by simulating an episode.

    Returns (goal_reached, -num_expansions) for tuple comparison.
    Higher is better: goal_reached=True beats False, then fewer
    expansions wins.  Expansion counts come from the astar_metrics JSON
    file written by run_astar, not from the policy itself.
    """
    _read_and_clear_astar_metrics()
    policy.reset(initial_obs, initial_info)
    state = initial_obs
    goal_reached = False
    for _ in range(max_timesteps):
        action = policy.get_action(state)
        state = get_next_state(state, action)
        if check_goal(state, goal):
            goal_reached = True
            break
    num_expansions = _read_and_clear_astar_metrics()
    return (goal_reached, -num_expansions)


def _load_generated_policy(
    code_str: str,
    get_actions: Callable,
    get_next_state: Callable,
    get_cost: Callable,
    check_goal: Callable,
) -> Any:
    """Load a GeneratedPolicy class from a code string and instantiate it."""
    namespace: dict[str, Any] = {}
    exec(  # pylint: disable=exec-used  # noqa: S102
        compile(code_str, "<generated_policy>", "exec"), namespace
    )
    cls = namespace["GeneratedPolicy"]
    return cls(
        get_actions=get_actions,
        get_next_state=get_next_state,
        get_cost=get_cost,
        check_goal=check_goal,
    )


def _parse_all_python_code_blocks(text: str) -> list[str]:
    """Extract all ```python ...

    ``` blocks from text.
    """
    blocks = []
    prefix = "```python"
    suffix = "```"
    remaining = text
    while prefix in remaining:
        start = remaining.index(prefix) + len(prefix)
        remaining = remaining[start:]
        if suffix in remaining:
            end = remaining.index(suffix)
            blocks.append(remaining[:end])
            remaining = remaining[end + len(suffix) :]
        else:
            blocks.append(remaining)
            break
    return blocks


class _MultiBlockSyntaxCheck(RepromptCheck):
    """Validate that the response contains N syntactically valid Python
    blocks."""

    def __init__(self, expected_count: int) -> None:
        self._expected_count = expected_count

    def get_reprompt(self, query: Query, response: Response) -> Query | None:
        blocks = _parse_all_python_code_blocks(response.text)
        if len(blocks) < self._expected_count:
            error_msg = (
                f"Expected {self._expected_count} ```python blocks but found "
                f"{len(blocks)}. Please provide exactly {self._expected_count} "
                f"separate ```python code blocks."
            )
            return create_reprompt_from_error_message(query, response, error_msg)
        for i, block in enumerate(blocks):
            try:
                ast.parse(block)
            except SyntaxError as e:
                error_msg = f"Policy {i} has a syntax error: {e}"
                return create_reprompt_from_error_message(query, response, error_msg)
        return None


def _synthesize_policy_classes(
    environment_description: str,
    llm: PretrainedLargeModel,
    example_observation: _ObsType,
    action_space: Space[_ActType],
    num_candidates: int,
    seed: int = 0,
) -> list[str]:
    """Use the LLM to synthesize N GeneratedPolicy classes and return their
    code strings."""

    query = Query(
        f"""Generate {num_candidates} distinct Python files, each containing a class
        `GeneratedPolicy` with the following interface:

        ```
        class GeneratedPolicy:
            def __init__(self, get_actions, get_next_state, get_cost, check_goal):
                \"\"\"Initialize with environment transition functions.

                Args:
                    get_actions(): Returns a list of all possible actions.
                    get_next_state(state, action): Returns the resulting state after
                        taking an action. If the action is invalid or blocked,
                        returns the same state unchanged.
                    get_cost(): Returns the cost of a single action (a float).
                    check_goal(state, goal): Returns True if the state satisfies
                        the goal condition.
                \"\"\"
                ...

            def reset(self, obs, info):
                \"\"\"Called at the start of each episode with the initial observation
                and info dict (which may contain environment metadata).\"\"\"
                ...

            def get_action(self, obs):
                \"\"\"Return a valid action for the given observation.

                Called once per timestep. Must ALWAYS return a valid action integer.
                Must NEVER return None.
                \"\"\"
                ...
        ```

        Each class can maintain internal state between calls (e.g. a cached plan,
        accumulated search metrics, etc.). The `reset` method is called once at the
        start of each episode. The `get_action` method is called each step.

        {PLANNER_DOC}

        ## Environment Description

        {environment_description}

        ## Action Space

        {action_space}

        ## Example Observation

        {example_observation}

        ## Instructions

        Generate exactly {num_candidates} policies as separate ```python code blocks.
        Each policy must be a complete, self-contained Python file with the
        `GeneratedPolicy` class and any necessary imports.

        Each policy should represent a DIFFERENT strategy along the
        planning-reactivity spectrum:

        - REACTIVE policies use simple heuristic rules based on the current
          observation (e.g., greedily reducing distance to the goal). They are
          computationally cheap but may fail in complex environments.
        - PLANNING-HEAVY policies use the planner liberally, even when simpler
          strategies might suffice. They are computationally expensive but robust.
        - HYBRID policies use reactive control when the situation is simple and
          fall back to the planner only when reactive control is insufficient.
          These are often the best approach: they minimize computation by only
          invoking the planner when genuinely needed.

        The PRIMARY objective is robustness: the policy must reliably reach
        the goal across diverse, unseen instances of the environment — not just
        the small samples shown during training. A policy that fails on novel
        instances is strictly worse than one that uses more computation but
        succeeds reliably.
        
        A good policy should be both effective (reach the goal) and efficient
        (minimize unnecessary computation). Prefer reactive control when the
        path is straightforward and only fall back to planning when it is
        genuinely required.

        Vary the strategies meaningfully across the {num_candidates} policies. For
        example, one policy might be purely reactive, another might always plan, and
        others might use different criteria for when to switch between reactive and
        planning modes.

        get_action must ALWAYS return a valid action. Never return None.

        Do not include example usages or test code.
        """,
        hyperparameters={"seed": seed},
    )

    reprompt_checks: list[RepromptCheck] = [
        _MultiBlockSyntaxCheck(num_candidates),
    ]

    response = query_with_reprompts(llm, query, reprompt_checks)
    blocks = _parse_all_python_code_blocks(response.text)[:num_candidates]

    for i, code_str in enumerate(blocks):
        logger.info("Synthesized GeneratedPolicy class %d:\n%s", i, code_str)
        print(f"\n=== Policy {i} ===\n{code_str}")

    return blocks
