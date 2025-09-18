"""Tests for PRBench environment provider."""

from omegaconf import DictConfig, OmegaConf

from programmatic_policy_learning.envs.providers.prbench_provider import (
    create_prbench_env,
)


def test_prbench_env_creation() -> None:
    """Test PRBench environment creation and basic API."""
    cfg: DictConfig = OmegaConf.create(
        {
            "make_kwargs": {"id": "prbench/Motion2D-p1-v0"},
        }
    )
    env = create_prbench_env(cfg)
    assert env is not None
    obs, _ = env.reset()
    assert obs is not None
    action = env.action_space.sample()
    step_result = env.step(action)
    assert isinstance(step_result, tuple)
    # img = env.render()
    # assert img is not None
