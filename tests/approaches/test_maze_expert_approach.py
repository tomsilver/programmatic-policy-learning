"""Tests for ExpertApproach with the maze expert."""

import numpy as np

from programmatic_policy_learning.approaches.expert_approach import ExpertApproach
from programmatic_policy_learning.approaches.experts.maze_experts import (
    create_expert_maze_with_outer_world_policy,
)
from programmatic_policy_learning.envs.providers.maze_provider import (
    MazeEnv,
)


def test_maze_expert_approach() -> None:
    """Tests for ExpertApproach with the maze expert."""

    # Create maze environment.
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

    # Create the expert approach.
    expert_policy = create_expert_maze_with_outer_world_policy(
        grid=env.grid.copy(),
        goal=env.goal_pos,
        inner_h=env.inner_h,
        inner_w=env.inner_w,
        get_actions=env.get_actions,
        get_next_state=env.get_next_state,
        get_cost=env.get_cost,
        check_goal=env.check_goal,
    )

    approach: ExpertApproach = ExpertApproach(
        environment_description="Not used",
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=123,
        expert_fn=expert_policy,
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
