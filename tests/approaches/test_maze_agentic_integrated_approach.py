"""Tests for agentic_integrated_approach.py with maze environment."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel

from programmatic_policy_learning.approaches.agentic_integrated_approach import (
    AgenticIntegratedApproach,
)
from programmatic_policy_learning.envs.providers.maze_provider import MazeEnv

runllms = pytest.mark.skipif("not config.getoption('runllms')")


@runllms
def test_agentic_integrated_approach_maze_with_real_llm() -> None:
    """Tests for AgenticIntegratedApproach() with real LLM on maze
    environment."""
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

    env.action_space.seed(123)

    obs, info = env.reset(seed=123)
    environment_description = """
        A maze environment with an outer void, a wall border, and an inner maze.

        COORDINATE SYSTEM:
        - Observation is (row, col).
        - Rows increase going South, columns increase going East.
        - The inner maze occupies rows [0 .. inner_height-1], cols [0 .. inner_width-1].
          The inner dimensions are not known in advance and vary between episodes.

        LAYOUT:
        - Outer void: an obstacle-free area surrounding the maze on all four
          sides. The agent starts here each episode. The void extends
          arbitrarily far in every direction.
        - Wall border: a one-cell-thick solid rectangle enclosing the inner
          maze with exactly ONE gap — the entrance at (-1, 0). Moving into
          any wall cell is blocked (get_next_state returns the same state).
        - Inner maze: a grid starting at (0, 0) with unknown internal walls.


        ACTIONS:
        - Action 0 (North): row -= 1.
        - Action 1 (South): row += 1.
        - Action 2 (East):  col += 1.
        - Action 3 (West):  col -= 1.

        WALL BORDER NAVIGATION:
        The wall border is a continuous barrier. You cannot move along or
        through it — only through the entrance at (-1, 0). Key consequences:
        - Row -1 is entirely wall EXCEPT col 0 (the entrance). If the agent
          reaches row -1 at any other column, it will be stuck — no East/West
          movement is possible along row -1.
        - Col -1 and col inner_width are entirely wall. No North/South
          movement is possible along those columns.
        - To navigate from the void to the entrance, the agent must stay clear
          of the wall border (row <= -2 or row >= inner_height+1 for horizontal
          movement, col <= -2 or col >= inner_width+1 for vertical movement)
          and approach the entrance at (-1, 0) from row -2, col 0.
        - From the entrance (-1, 0), moving South enters the maze at (0, 0).

        TASK:
        Navigate from the starting position in the outer void to the goal at the
        bottom-right corner of the inner maze. The goal position is provided in
        the info dict passed to reset(obs, info) as info["goal"].
    """

    os.environ["astar_metrics_path"] = str(
        Path(__file__).parents[2]
        / "src"
        / "programmatic_policy_learning"
        / "metrics"
        / "astar_metrics.json"
    )

    cache_path = Path("llm_cache.db")
    # cache_path = Path(tempfile.NamedTemporaryFile(suffix=".db").name)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    llm = OpenAIModel("gpt-5.2", cache)

    approach = AgenticIntegratedApproach(
        environment_description,
        env.observation_space,
        env.action_space,
        seed=126,
        llm=llm,
        get_actions=env.get_actions,
        get_next_state=env.get_next_state,
        get_cost=env.get_cost,
        check_goal=env.check_goal,
    )

    # TRAINING PHASE (obs and info already obtained from env.reset above).
    approach.reset(obs, info)
    env.close()

    # EVALUATION PHASE.
    inner_maze = np.array(
        [
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
        ],
        dtype=np.int8,
    )
    outer_margin = 20
    enable_render = True
    env = MazeEnv(
        inner_maze=inner_maze, outer_margin=outer_margin, enable_render=enable_render
    )
    obs, info = env.reset(seed=123)
    approach.update_env_callables(
        get_actions=env.get_actions,
        get_next_state=env.get_next_state,
        get_cost=env.get_cost,
        check_goal=env.check_goal,
    )
    approach.reset(obs, info)

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

    # Print search metrics from the JSON file written by run_astar
    metrics_path = Path(os.environ["astar_metrics_path"])
    total_evals = 0
    total_expansions = 0
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    total_evals += entry["num_evals"]
                    total_expansions += entry["num_expansions"]
    print("Search Metrics:")
    print("Num_evals:", total_evals)
    print("Num_expansions:", total_expansions)
