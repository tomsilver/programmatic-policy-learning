"""Tests for PRBench environment provider."""

from programmatic_policy_learning.envs.providers.pr_bench import pr_bench


class DummyEnvConfig:
    """Minimal config for testing PRBench provider."""

    class make_kwargs:
        """Minimal kwargs for PRBench env creation."""

        id = "prbench/Motion2D-p1-v0"  # Use a valid PRBench env id


def test_prbench_env_creation():
    """Test PRBench environment creation and basic API."""
    DummyEnvConfig.make_kwargs = {
        k: v
        for k, v in vars(DummyEnvConfig.make_kwargs).items()
        if not k.startswith("__")
    }
    env = pr_bench(DummyEnvConfig())
    assert env is not None
    obs, _ = env.reset()
    assert obs is not None
    action = env.action_space.sample()
    step_result = env.step(action)
    assert isinstance(step_result, tuple)
    # img = env.render()
    # assert img is not None
