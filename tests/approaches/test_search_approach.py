"""Tests for search_approach.py."""

from programmatic_policy_learning.envs.providers.maze_provider import (
    MazeEnv,
)
from programmatic_policy_learning.approaches.search_approach import SearchApproach
import numpy as np


def test_search_approach_maze_env():
    """Tests SearchApproach() in a maze environment."""

    # Create maze environment.
    inner_maze = np.array(
    [
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    ],
    dtype=np.int8,
)
    outer_margin = 10
    enable_render = True  # TODO change to False later
    env = MazeEnv(
        inner_maze=inner_maze, outer_margin=outer_margin, enable_render=enable_render
    )

    # Create the search approach.
    approach = SearchApproach(
        environment_description="Not used",
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=123,
        get_actions=env.get_actions,
        get_next_state=env.get_next_state,
        get_cost=env.get_cost,
        check_goal=env.check_goal,
    )

    # Run the approach in the environment.
    obs, info = env.reset(seed=123)
    approach.reset(obs, info)
    for _ in range(1000):
        action = approach.step()
        obs, rew, done, truncated, info = env.step(action)
        reward = float(rew)
        env.render()
        assert not truncated
        approach.update(obs, reward, done, info)
        if done:
            break
    else:
        assert False, "Goal was not reached!"

