"""Tests for integrated_approach.py with maze environment."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel, OrderedResponseModel
from prpl_llm_utils.structs import Response

from programmatic_policy_learning.approaches.integrated_approach import (
    IntegratedApproach,
)
from programmatic_policy_learning.envs.providers.maze_provider import MazeEnv

runllms = pytest.mark.skipif("not config.getoption('runllms')")


def test_integrated_approach_maze() -> None:
    """Tests for IntegratedApproach() with maze environment."""
    # Create maze environment (same as test_maze_expert_approach.py)
    inner_maze = np.array(
        [
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
        ],
        dtype=np.int8,
    )
    outer_margin = 10
    enable_render = False
    env = MazeEnv(
        inner_maze=inner_maze, outer_margin=outer_margin, enable_render=enable_render
    )

    # Sample a constant action for testing
    env.action_space.seed(123)
    constant_action = env.action_space.sample()

    environment_description = (
        "A maze environment with an outer void region and an inner maze. "
        "The agent starts in the outer world and must navigate to the entrance "
        "at position (-1, 0), then solve the maze to reach the goal at the "
        "bottom-right corner. Actions are: 0=North, 1=South, 2=East, 3=West."
    )

    cache_path = Path(tempfile.NamedTemporaryFile(suffix=".db").name)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    response = Response(
        f"""```python
def _policy(obs, get_actions, get_next_state, get_cost, check_goal):
    return {constant_action}
```
""",
        {},
    )
    llm = OrderedResponseModel([response], cache)

    approach = IntegratedApproach(
        environment_description,
        env.observation_space,
        env.action_space,
        seed=123,
        llm=llm,
        get_actions=env.get_actions,
        get_next_state=env.get_next_state,
        get_cost=env.get_cost,
        check_goal=env.check_goal,
    )

    obs, info = env.reset(seed=123)
    approach.reset(obs, info)
    for _ in range(5):
        action = approach.step()
        assert env.action_space.contains(action)
        obs, reward, terminated, _, info = env.step(action)
        approach.update(obs, float(reward), terminated, info)


@runllms
def test_integrated_approach_maze_with_real_llm() -> None:
    """Tests for IntegratedApproach() with real LLM on maze environment."""
    # inner_maze = np.array(
    #     [
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #     ],
    #     dtype=np.int8,
    # )
    inner_maze = np.array(
        [
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
        ],
        dtype=np.int8,
    )
    outer_margin = 10
    enable_render = True
    env = MazeEnv(
        inner_maze=inner_maze, outer_margin=outer_margin, enable_render=enable_render
    )

    env.action_space.seed(None)
    environment_description = f"""
        A maze environment with two regions: an outer void and an inner maze.

        COORDINATE SYSTEM:
        - Observation is (row, col).
        - Rows increase going South, columns increase going East.
        - The inner maze occupies rows 0..{env.inner_h - 1} and cols 0..{env.inner_w - 1}.

        REGIONS:
        - Outer void: open area surrounding the maze. No obstacles. The agent
          can be at any (row, col) outside the wall border.
        - Wall border: a solid rectangular wall that fully encloses the maze.
          It occupies row=-1 (from col=-1 to col={env.inner_w}),
          row={env.inner_h}, col=-1, and col={env.inner_w}. Any attempt to move
          into a wall cell is blocked (the agent stays in place).
        - Entrance: the ONLY gap in the wall border, at cell (-1, 0). This is
          the only way into the maze from the outer void.
        - Inner maze: a {env.inner_h}x{env.inner_w} grid with internal walls.
          The layout is not known in advance.

        IMPORTANT: The wall border is a solid barrier. The entire row=-1
        (except the entrance at (-1, 0)) is wall. If you navigate to row=-1
        at any column other than 0, you will be stuck on the wall and unable
        to move East or West along that row. To avoid this, always go to
        row=-2 (which is in the open outer void) before moving horizontally.

        ACTIONS:
        - Action 0 (North): row -= 1.
        - Action 1 (South): row += 1.
        - Action 2 (East):  col += 1.
        - Action 3 (West):  col -= 1.

        TASK:
        Step 1: Navigate from a random outer void position to the entrance (-1, 0).
                The wall border blocks direct paths, so you must go AROUND it:
                a) First, move to row=-2 (above the wall border). Row -2 is in
                   the outer void and completely obstacle-free.
                   - If row > -2, move North (action 0) until row == -2.
                   - If row < -2, move South (action 1) until row == -2.
                b) Then, move horizontally to col=0 (the entrance column).
                   - If col > 0, move West (action 3) until col == 0.
                   - If col < 0, move East (action 2) until col == 0.
                c) Finally, move South (action 1) once to reach (-1, 0).
        Step 2: From (-1, 0), move South (action 1) to enter the maze at (0, 0).
        Step 3: Navigate through the inner maze to the goal at {env.goal_pos}.
                The maze has internal walls, so use the A* planner with
                get_next_state to find a path.
    """

    cache_path = Path(tempfile.NamedTemporaryFile(suffix=".db").name)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    llm = OpenAIModel("gpt-4.1", cache)

    approach = IntegratedApproach(
        environment_description,
        env.observation_space,
        env.action_space,
        seed=0,
        llm=llm,
        get_actions=env.get_actions,
        get_next_state=env.get_next_state,
        get_cost=env.get_cost,
        check_goal=env.check_goal,
    )

    obs, info = env.reset(seed=None)
    approach.reset(obs, info)
    # print(approach._policy)  # pylint: disable=protected-access
    goal_reached = False
    for _ in range(1000):
        action = approach.step()
        assert env.action_space.contains(action)
        obs, reward, terminated, _, info = env.step(action)
        env.render()
        approach.update(obs, float(reward), terminated, info)
        if terminated:
            goal_reached = True
            break

    print("Goal reached:", goal_reached)

    # Print search metrics (accumulated by the synthesized policy)
    policy_fn = approach._policy  # pylint: disable=protected-access
    print("Search Metrics:")
    print("Num_evals:", getattr(policy_fn, "total_num_evals", 0))
    print("Num_expansions:", getattr(policy_fn, "total_num_expansions", 0))
