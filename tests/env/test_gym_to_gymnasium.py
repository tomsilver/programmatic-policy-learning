"""Tests for GymToGymnasium adapter."""

import gym

from programmatic_policy_learning.envs.utils.gym_to_gymnasium import GymToGymnasium


def test_adapter_step_and_reset() -> None:
    """Test GymToGymnasium step and reset outputs."""
    env = gym.make("CartPole-v1")
    wrapped = GymToGymnasium(env)
    obs, info = wrapped.reset()
    assert obs is not None
    obs, reward, terminated, truncated, info = wrapped.step(env.action_space.sample())
    assert isinstance(obs, type(env.observation_space.sample()))
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
