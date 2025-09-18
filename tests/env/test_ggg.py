"""Tests for GGG environment provider."""

from omegaconf import DictConfig, OmegaConf

from programmatic_policy_learning.envs.providers.ggg_provider import create_ggg_env


def test_ggg_env_creation() -> None:
    """Test GGG environment creation and basic API."""
    cfg: DictConfig = OmegaConf.create(
        {
            "make_kwargs": {"id": "TwoPileNim0-v0"},
        }
    )
    env = create_ggg_env(cfg)
    assert env is not None
    obs, _ = env.reset()
    assert obs is not None
    assert env.action_space is not None
    action = env.action_space.sample()
    step_result = env.step(action)
    assert isinstance(step_result, tuple)
    # img = env.render()
    # assert img is not None
