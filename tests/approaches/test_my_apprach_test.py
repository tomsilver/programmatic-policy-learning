"""Tests for same_action.py."""

import gymnasium

from programmatic_policy_learning.approaches.my_apprach_test import MyApproachTest


def test_same_action_approach():
    """Tests for RandomActionsApproach()."""
    # Just test that this runs without crashing.
    env = gymnasium.make("LunarLander-v3")
    approach = MyApproachTest(
        "N/A", env.observation_space, env.action_space, seed=123
    )
    obs, info = env.reset()
    approach.reset(obs, info)
    for _ in range(5):
        action = approach.step()
        obs, reward, terminated, _, info = env.step(action)
        approach.update(obs, reward, terminated, info)