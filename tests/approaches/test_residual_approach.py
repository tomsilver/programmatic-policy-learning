"""Tests for ResidualApproach (SB3 TD3/DDPG backends) on Pendulum-v1."""

from __future__ import annotations

import contextlib
import io
from typing import Any, Callable, Literal

import gymnasium as gym
import numpy as np
import pytest

from programmatic_policy_learning.approaches.base_approach import BaseApproach
from programmatic_policy_learning.approaches.expert_approach import ExpertApproach
from programmatic_policy_learning.approaches.experts.pendulum_experts import (
    create_manual_pendulum_policy,
)
from programmatic_policy_learning.approaches.residual_approach import ResidualApproach

ENV_ID = "Pendulum-v1"
SEED = 42
TRAIN_STEPS = 1
EVAL_EPISODES = 5
BACKENDS: list[Literal["sb3-td3", "sb3-ddpg"]] = ["sb3-td3", "sb3-ddpg"]


def build_env() -> gym.Env:
    """Construct a fresh Gymnasium environment."""
    return gym.make(ENV_ID)


def env_factory(_instance_num: int, **_kwargs: Any) -> gym.Env:
    """Hydra-style env factory; tests ignore instance num and extra kwargs."""
    return build_env()


class _FnExpert:
    """Minimal expert wrapper exposing get_action(obs) around a base_fn."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self._fn = fn

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Return the action for `obs` computed by the wrapped policy."""
        return self._fn(obs)


def eval_policy(
    approach: Any, n_episodes: int = EVAL_EPISODES, seed: int = SEED
) -> np.ndarray:
    """Run n_episodes and return per-episode returns using public API."""
    returns: list[float] = []
    for ep in range(n_episodes):
        env = build_env()
        obs, info = env.reset(seed=seed + ep)

        # Silence any prints from envs/policies to keep CI logs clean.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            approach.reset(obs, info)

            terminated = truncated = False
            ep_ret = 0.0
            while not (terminated or truncated):
                action = approach.step()
                obs, rew, terminated, truncated, info = env.step(action)
                approach.update(obs, float(rew), terminated, info)
                ep_ret += float(rew)

        returns.append(ep_ret)
    return np.asarray(returns, dtype=np.float32)


@pytest.mark.parametrize("backend", BACKENDS)
def test_residual_vs_base_runs(backend: Literal["sb3-td3", "sb3-ddpg"]) -> None:
    """Smoke test: residual approach runs and is not catastrophically worse than base."""
    tmp = build_env()
    assert isinstance(tmp.action_space, gym.spaces.Box)

    # Base (manual) policy used as the expert's action.
    base_policy: Callable[[np.ndarray], np.ndarray] = create_manual_pendulum_policy(
        tmp.action_space
    )

    # Baseline returns using ExpertApproach (deterministic expert only).
    base: BaseApproach[np.ndarray, np.ndarray] = ExpertApproach(
        f"residual-{backend}-base",
        tmp.observation_space,
        tmp.action_space,
        seed=SEED,
        expert_fn=base_policy,
    )
    base_returns = eval_policy(base, n_episodes=EVAL_EPISODES, seed=SEED)
    assert base_returns.shape == (EVAL_EPISODES,)
    assert np.isfinite(base_returns).all()

    # Residual learner on top of the same expert, Hydra-style ctor.
    residual = ResidualApproach(
        f"residual-{backend}",
        tmp.observation_space,
        tmp.action_space,
        seed=SEED,
        expert=_FnExpert(base_policy),
        env_factory=env_factory,
        backend=backend,
        total_timesteps=TRAIN_STEPS,  # keep it tiny for speed
        verbose=0,
    )

    # Train briefly, then evaluate via public API.
    residual.train()
    residual_returns = eval_policy(residual, n_episodes=EVAL_EPISODES, seed=SEED + 123)
    assert residual_returns.shape == (EVAL_EPISODES,)
    assert np.isfinite(residual_returns).all()

    # Loose sanity bound: residual should not be catastrophically worse than base.
    assert residual_returns.mean() > base_returns.mean() - 100.0
