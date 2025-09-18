"""Tests for PRBench environment provider."""

from programmatic_policy_learning.envs.providers.ggg_provider import create_ggg_env


class DummyEnvConfig:
    """Minimal config for testing GGG provider."""

    class make_kwargs:
        """Minimal kwargs for GGG env creation."""

        id = "TwoPileNim0-v0"


def test_ggg_env_creation():
    """Test PRBench environment creation and basic API."""
    env = create_ggg_env(DummyEnvConfig())
    assert env is not None
    obs, _ = env.reset()
    assert obs is not None
    action = env.action_space.sample()
    step_result = env.step(action)
    assert isinstance(step_result, tuple)
    # img = env.render()
    # assert img is not None
