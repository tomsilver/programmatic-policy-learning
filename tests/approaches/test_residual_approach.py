"""Tests for the ResidualApproach (SB3 TD3/DDPG backends) on Pendulum-v1."""

# pylint: disable=protected-access

import contextlib
import io
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import pytest

from programmatic_policy_learning.approaches.pendulum_stupid_approach import (
    PendulumStupidAlgorithm,
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


def eval_policy(
    approach: Any, n_episodes: int = EVAL_EPISODES, seed: int = SEED
) -> np.ndarray:
    """Run n_episodes and return per-episode returns; suppress base prints so
    output is clean."""
    returns: list[float] = []
    for ep in range(n_episodes):
        env = build_env()
        obs, _ = env.reset(seed=seed + ep)
        terminated = truncated = False
        ep_ret = 0.0
        while not (terminated or truncated):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                approach._last_observation = np.asarray(
                    obs, dtype=np.float32
                )  # pylint: disable=protected-access
                a = approach._get_action()  # pylint: disable=protected-access
            obs, r, terminated, truncated, _ = env.step(a)
            ep_ret += float(r)
        returns.append(ep_ret)
    return np.asarray(returns, dtype=np.float32)


@pytest.mark.parametrize("backend", BACKENDS)
def test_residual_vs_base_runs(backend: Literal["sb3-td3", "sb3-ddpg"]) -> None:
    """Smoke test: residual approach runs and is not catastrophically worse than base."""
    tmp = build_env()
    base = PendulumStupidAlgorithm(
        "base", tmp.observation_space, tmp.action_space, seed=SEED
    )

    base_returns = eval_policy(base, n_episodes=EVAL_EPISODES, seed=SEED)
    assert base_returns.shape == (EVAL_EPISODES,)
    assert np.isfinite(base_returns).all()

    tmp = build_env()
    residual = ResidualApproach(
        f"residual-{backend}",
        tmp.observation_space,
        tmp.action_space,
        seed=SEED,
        env_builder=build_env,
        base_approach_instance=base,
        backend=backend,
        total_timesteps=TRAIN_STEPS,
        verbose=0,
    )

    try:
        residual._backend._model.learn(  # pylint: disable=protected-access
            total_timesteps=TRAIN_STEPS, progress_bar=False
        )
    except Exception:  # pylint: disable=broad-exception-caught
        residual.train()
    finally:
        pass

    residual_returns = eval_policy(residual, n_episodes=EVAL_EPISODES, seed=SEED + 123)
    assert residual_returns.shape == (EVAL_EPISODES,)
    assert np.isfinite(residual_returns).all()

    assert residual_returns.mean() > base_returns.mean() - 100.0
