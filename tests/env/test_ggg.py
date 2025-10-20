"""Tests for GGG environment provider."""

import logging

from omegaconf import OmegaConf

from programmatic_policy_learning.envs.providers.ggg_provider import (
    GGGEnvWithTypes,
    create_ggg_env,
)


def test_ggg_env_creation() -> None:
    """Test GGG environment creation and basic API."""
    cfg = OmegaConf.create(
        {
            "provider": "ggg",
            "make_kwargs": {"base_name": "TwoPileNim", "id": "TwoPileNim0-v0"},
            "instance_num": 0,
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


def test_ggg_env_with_types_classname_extraction() -> None:
    """Test GGGEnvWithTypes class name extraction and object types."""
    instance_num = 1
    cfg = OmegaConf.create(
        {
            "provider": "ggg",
            "make_kwargs": {"base_name": "TwoPileNim", "id": "TwoPileNim0-v0"},
            "instance_num": instance_num,
        }
    )
    env = create_ggg_env(cfg)
    # Check that the wrapper is used
    assert isinstance(env, GGGEnvWithTypes)
    obs, _ = env.reset()
    logging.info(obs)
    class_name = env.env.unwrapped.__class__.__name__
    assert class_name == "TwoPileNimGymEnv1"
    object_types = env.get_object_types()
    assert "tpn.EMPTY" in object_types
    assert "tpn.TOKEN" in object_types
