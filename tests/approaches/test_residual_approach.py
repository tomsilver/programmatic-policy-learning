"""Tests for ResidualApproach (SB3 backend) on Pendulum-v1, with optional LunarLanderContinuous-v3.

Minimal changes from the original Pendulum-only test:
- Add ENV_ID / ENV_IDS and parametrize over env_id (optional).
- Add a tiny helper to build the right expert for each env.
- Keep a SINGLE backend (no backend parametrization).
"""

from __future__ import annotations

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

# --- ADD: import your LunarLander expert creator (adjust path/name to match your project) ---
# If you haven't created this file/function yet, comment this import out and also comment out
# "LunarLanderContinuous-v3" in ENV_IDS below.
from programmatic_policy_learning.approaches.experts.lundar_lander_experts import (
    create_manual_lunarlander_continuous_policy,
)

EnvId = Literal["Pendulum-v1", "LunarLanderContinuous-v3"]

# --- ORIGINAL: Pendulum constants ---
ENV_ID: EnvId = "Pendulum-v1"
SEED = 42
TRAIN_STEPS = 1
EVAL_EPISODES = 5

# --- CHANGE: single backend, fixed (no parametrization) ---
BACKEND: Literal["sb3-td3"] = "sb3-td3"

# --- ADD: list of envs to test; keep Pendulum first to match old behavior ---
ENV_IDS: list[EnvId] = [ENV_ID, "LunarLanderContinuous-v3"]


def build_env(env_id: str = ENV_ID) -> gym.Env:
    """Construct a fresh Gymnasium environment."""
    return gym.make(env_id)


def env_factory(_instance_num: int, **_kwargs: Any) -> gym.Env:
    """Hydra-style env factory; tests ignore instance num and extra kwargs."""
    # NOTE: this will be shadowed by a closure per env inside the test below.
    return build_env()


class _FnExpert:
    """Minimal expert wrapper exposing get_action(obs) around a base_fn."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self._fn = fn

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Return the action for `obs` computed by the wrapped policy."""
        return self._fn(obs)


def _make_expert_policy(env_id: EnvId, action_space: gym.spaces.Box) -> Callable[[np.ndarray], np.ndarray]:
    """Return the correct manual expert for the chosen env."""
    if env_id == "Pendulum-v1":
        return create_manual_pendulum_policy(action_space)
    if env_id == "LunarLanderContinuous-v3":
        return create_manual_lunarlander_continuous_policy(action_space)
    raise ValueError(f"Unsupported env_id: {env_id!r}")


def eval_policy(
    env_id: EnvId,
    approach: Any,
    n_episodes: int = EVAL_EPISODES,
    seed: int = SEED,
) -> np.ndarray:
    """Run n_episodes and return per-episode returns using public API."""
    returns: list[float] = []
    for ep in range(n_episodes):
        env = build_env(env_id)
        obs, info = env.reset(seed=seed + ep)

        approach.reset(obs, info)

        terminated = truncated = False
        ep_ret = 0.0
        while not (terminated or truncated):
            action = approach.step()
            obs, rew, terminated, truncated, info = env.step(action)
            # original code used `terminated` as done; keep that style but make it robust:
            approach.update(obs, float(rew), terminated or truncated, info)
            ep_ret += float(rew)

        returns.append(ep_ret)
    return np.asarray(returns, dtype=np.float32)


# --- MINIMAL CHANGE: parametrize env_id (instead of only Pendulum) ---
@pytest.mark.parametrize("env_id", ENV_IDS)
def test_residual_vs_base_runs(env_id: EnvId) -> None:
    """Smoke test: residual approach runs and is not catastrophically worse than base."""
    tmp = build_env(env_id)
    assert isinstance(tmp.action_space, gym.spaces.Box)
    assert isinstance(tmp.observation_space, gym.spaces.Box)

    # Base (manual) policy used as the expert's action.
    base_policy: Callable[[np.ndarray], np.ndarray] = _make_expert_policy(
        env_id, tmp.action_space
    )

    # Baseline returns using ExpertApproach (deterministic expert only).
    base: BaseApproach[np.ndarray, np.ndarray] = ExpertApproach(
        f"residual-{env_id}-{BACKEND}-base",
        tmp.observation_space,
        tmp.action_space,
        seed=SEED,
        expert_fn=base_policy,
    )
    base_returns = eval_policy(env_id, base, n_episodes=EVAL_EPISODES, seed=SEED)
    assert base_returns.shape == (EVAL_EPISODES,)
    assert np.isfinite(base_returns).all()

    # Env factory for THIS env_id (closure keeps it minimal and local)
    def _env_factory(_instance_num: int, **_kwargs: Any) -> gym.Env:
        return build_env(env_id)

    # Residual learner on top of the same expert, Hydra-style ctor.
    residual = ResidualApproach(
        f"residual-{env_id}-{BACKEND}",
        tmp.observation_space,
        tmp.action_space,
        seed=SEED,
        expert=_FnExpert(base_policy),
        env_factory=_env_factory,
        backend=BACKEND,
        total_timesteps=TRAIN_STEPS,  # keep it tiny for speed
        verbose=0,
    )

    # Train briefly, then evaluate via public API.
    residual.train()
    residual_returns = eval_policy(env_id, residual, n_episodes=EVAL_EPISODES, seed=SEED + 123)
    assert residual_returns.shape == (EVAL_EPISODES,)
    assert np.isfinite(residual_returns).all()

    # Loose sanity bound: residual should not be catastrophically worse than base.
    # LunarLander reward scale can be wider, so keep this generous.
    slack = 100.0 if env_id == "Pendulum-v1" else 200.0
    assert residual_returns.mean() > base_returns.mean() - slack
