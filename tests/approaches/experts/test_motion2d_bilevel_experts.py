"""Minimal integration test for Motion2D via the generic KinDER bilevel
expert."""

from __future__ import annotations

import numpy as np
import pytest

from programmatic_policy_learning.approaches.experts.kinder_bilevel_experts import (
    create_kinder_bilevel_expert,
)


def test_motion2d_bilevel_expert_returns_valid_action() -> None:
    """The real bilevel expert should produce a valid Motion2D action."""

    kinder = pytest.importorskip("kinder")
    pytest.importorskip("kinder_bilevel_planning")
    pytest.importorskip("bilevel_planning")
    pytest.importorskip("kinder_models")

    # pylint: disable=import-outside-toplevel
    from gymnasium.envs.registration import register, registry

    env_id = "kinder/Motion2D-p0-v0"
    if env_id not in registry:
        register(
            id=env_id,
            entry_point="kinder.envs.kinematic2d.motion2d:Motion2DEnv",
            kwargs={"num_passages": 0},
        )

    env = kinder.make(env_id, render_mode="rgb_array")
    try:
        obs, info = env.reset(seed=0)
        expert = create_kinder_bilevel_expert(
            env.observation_space,
            env.action_space,
            seed=0,
            env_name="Motion2D",
            env_model_name="motion2d",
            model_kwargs={"num_passages": 0},
        )

        expert.reset(obs, info)
        action = expert.step()

        assert isinstance(action, np.ndarray)
        assert action.shape == env.action_space.shape
        assert action.dtype == env.action_space.dtype
        assert np.all(action >= env.action_space.low)
        assert np.all(action <= env.action_space.high)
    finally:
        env.close()
