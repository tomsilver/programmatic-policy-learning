"""Unit tests for generic KinDER bilevel expert wrappers."""

from __future__ import annotations

from typing import Any

import pytest

from programmatic_policy_learning.approaches.experts import kinder_bilevel_experts


def test_resolve_bilevel_model_name_known_envs() -> None:
    """Known KinDER envs should map to the expected model module names."""
    assert (
        kinder_bilevel_experts.resolve_bilevel_model_name(env_name="Motion2D")
        == "motion2d"
    )
    assert (
        kinder_bilevel_experts.resolve_bilevel_model_name(env_name="PushPullHook2D")
        == "pushpullhook2d"
    )


def test_create_kinder_bilevel_expert_forwards_model_name_and_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory should resolve the env model name and forward model kwargs."""
    calls: dict[str, Any] = {}

    class _FakeAgent:
        def __init__(self, env_models: Any, seed: int, **kwargs: Any) -> None:
            """Record constructor inputs for assertions."""
            calls["env_models"] = env_models
            calls["seed"] = seed
            calls["agent_kwargs"] = kwargs

        def reset(self, obs: Any, info: dict[str, Any]) -> None:
            """Stub reset method for the fake planning agent."""
            del obs, info
            return None

        def step(self) -> Any:
            """Stub step method for the fake planning agent."""
            return None

        def update(
            self, obs: Any, reward: float, done: bool, info: dict[str, Any]
        ) -> None:
            """Stub update method for the fake planning agent."""
            del obs, reward, done, info
            return None

    def _fake_create_models(
        env_name: str, observation_space: Any, action_space: Any, **kwargs: Any
    ) -> dict[str, Any]:
        calls["model_name"] = env_name
        calls["observation_space"] = observation_space
        calls["action_space"] = action_space
        calls["model_kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(
        kinder_bilevel_experts,
        "_load_bilevel_components",
        lambda: (_FakeAgent, _fake_create_models),
    )

    obs_space = object()
    act_space = object()
    expert = kinder_bilevel_experts.create_kinder_bilevel_expert(
        obs_space,
        act_space,
        seed=7,
        env_name="Motion2D",
        model_kwargs={"num_passages": 2},
        planning_timeout=12.5,
    )

    assert isinstance(expert, kinder_bilevel_experts.KinderBilevelPlanningExpert)
    assert calls["model_name"] == "motion2d"
    assert calls["observation_space"] is obs_space
    assert calls["action_space"] is act_space
    assert calls["model_kwargs"] == {"num_passages": 2}
    assert calls["seed"] == 7
    assert calls["agent_kwargs"]["planning_timeout"] == 12.5


def test_create_kinder_bilevel_expert_missing_model_has_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing upstream env models should raise a domain-specific error."""

    class _FakeAgent:
        def __init__(self, env_models: Any, seed: int, **kwargs: Any) -> None:
            """Ignore constructor inputs for the failing-path test double."""
            del env_models, seed, kwargs

    def _fake_create_models(
        env_name: str, observation_space: Any, action_space: Any, **kwargs: Any
    ) -> Any:
        del observation_space, action_space, kwargs
        raise FileNotFoundError(env_name)

    monkeypatch.setattr(
        kinder_bilevel_experts,
        "_load_bilevel_components",
        lambda: (_FakeAgent, _fake_create_models),
    )

    with pytest.raises(FileNotFoundError, match="pushpullhook2d"):
        kinder_bilevel_experts.create_kinder_bilevel_expert(
            object(),
            object(),
            env_name="PushPullHook2D",
        )
