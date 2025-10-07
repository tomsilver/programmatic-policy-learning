"""Tests for Maze environment provider."""

import numpy as np
from omegaconf import DictConfig, OmegaConf

from programmatic_policy_learning.envs.providers.maze_provider import (
    create_maze_env,
)


def test_maze_env_creation() -> None:
    """Test Maze environment creation and basic API."""
    cfg: DictConfig = OmegaConf.create(
        {
            "make_kwargs": {
                "outer_margin": 2,
                "enable_render": False, 
            }
        }
    )

    # Create environment via provider
    env = create_maze_env(cfg)
    assert env is not None, "Maze environment should be successfully created"

    # Reset environment and check observation
    obs, info = env.reset()
    assert obs is not None, "Observation after reset should not be None"
    assert isinstance(obs, np.ndarray), "Observation should be a numpy array"

    # Verify action space exists and sample an action
    assert env.action_space is not None, "Action space must be defined"
    action = env.action_space.sample()
    assert isinstance(
        action, (int, np.integer)
    ), "Sampled action should be an integer type"

    # Take a step and verify return structure
    step_result = env.step(action)
    assert isinstance(step_result, tuple), "Step should return a tuple"
    assert len(step_result) == 5, "Gymnasium step should return 5 elements"

    next_obs, reward, terminated, truncated, info = step_result
    assert next_obs is not None, "Next observation should not be None"
    assert isinstance(reward, float), "Reward should be a float"
    assert isinstance(terminated, bool), "Terminated flag should be a bool"
    assert isinstance(truncated, bool), "Truncated flag should be a bool"
    assert isinstance(info, dict), "Info should be a dictionary"
