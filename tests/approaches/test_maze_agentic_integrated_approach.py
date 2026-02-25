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
    # inner_maze = np.array(
    #     [
    #         [0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0],
    #     ]
    # )
    outer_margin = 10
    enable_render = True
    env = MazeEnv(
        inner_maze=inner_maze, outer_margin=outer_margin, enable_render=enable_render
    )

    env.action_space.seed(123)

    obs, info = env.reset(seed=123)
    goal = info["goal"]
    environment_description = f"""The observation is a (row, col) position tuple. The goal is always at
the bottom-right corner of the inner maze grid, i.e.,
(inner_height - 1, inner_width - 1). The maze dimensions can vary across
episodes, so the goal position must be determined dynamically by probing
the environment with get_next_state to find the inner maze boundaries.

Here is one example from this environment:
- Observation: {obs}
- Goal: {goal}

This is just one example. Both the starting position and the maze dimensions
(and therefore the goal and observation bounds) can change across episodes.
The written policy must not hardcode positions or dimensions. Use
get_next_state to discover the environment structure dynamically."""

    os.environ["astar_metrics_path"] = str(
        Path(__file__).parents[2]
        / "src"
        / "programmatic_policy_learning"
        / "metrics"
        / "astar_metrics.json"
    )

    cache_path = Path("llm_cache.db")
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    llm = OpenAIModel("gpt-4.1", cache)

    approach = AgenticIntegratedApproach(
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

    # TRAINING PHASE (obs and info already obtained from env.reset above).
    approach.reset(obs, info)
    env.close()

    # EVALUATION PHASE.
    # TODO: add multiple mazes.
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
