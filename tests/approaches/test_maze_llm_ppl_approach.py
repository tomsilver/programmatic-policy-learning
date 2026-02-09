"""Tests for llm_ppl_approach.py with maze environment."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel, OrderedResponseModel
from prpl_llm_utils.structs import Response

from programmatic_policy_learning.approaches.llm_ppl_approach import LLMPPLApproach
from programmatic_policy_learning.envs.providers.maze_provider import MazeEnv

runllms = pytest.mark.skipif("not config.getoption('runllms')")


def test_llm_ppl_approach_maze() -> None:
    """Tests for LLMPPLApproach() with maze environment."""
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
def _policy(obs):
    return {constant_action}
```
""",
        {},
    )
    llm = OrderedResponseModel([response], cache)

    approach = LLMPPLApproach(
        environment_description,
        env.observation_space,
        env.action_space,
        seed=123,
        llm=llm,
    )

    obs, info = env.reset(seed=123)
    approach.reset(obs, info)
    for _ in range(5):
        action = approach.step()
        assert action == constant_action
        obs, reward, terminated, _, info = env.step(action)
        approach.update(obs, float(reward), terminated, info)


@runllms
def test_llm_ppl_approach_maze_with_real_llm() -> None:
    """Tests for LLMPPLApproach() with real LLM on maze environment."""
    inner_maze = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    outer_margin = 10
    enable_render = False
    env = MazeEnv(
        inner_maze=inner_maze, outer_margin=outer_margin, enable_render=enable_render
    )

    env.action_space.seed(123)
    environment_description = f"""
        Navigate a RxC grid to reach the goal.

        COORDINATE SYSTEM:
        - Your observation is (row, col)
        - Row 0 is at the top, row R is at the bottom
        - Col 0 is at the left, col C is at the right
        - Larger row values mean further down, larger col values mean further right

        ACTIONS:
        - 0 = North: row decreases by 1
        - 1 = South: row increases by 1
        - 2 = East: col increases by 1
        - 3 = West: col decreases by 1

        TASK:
        - The agent begins at a random position in the grid.
        - The goal is at position {env.goal_pos}.
        - The grid is completely open with no obstacles.
        - Navigate from start to goal.

        Write a policy function that takes (row, col) and returns an action (0, 1, 2, or 3).
    """

    cache_path = Path(tempfile.NamedTemporaryFile(suffix=".db").name)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    llm = OpenAIModel("gpt-4o-mini", cache)

    approach = LLMPPLApproach(
        environment_description,
        env.observation_space,
        env.action_space,
        seed=123,
        llm=llm,
    )

    obs, info = env.reset(seed=123)
    approach.reset(obs, info)

    # Uncomment if curious.
    # print(approach._policy)
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
