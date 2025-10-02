# tests/approaches/test_residual_approach.py
import contextlib
import io
import sys

import gymnasium as gym
import numpy as np
import pytest

from programmatic_policy_learning.approaches.pendulum_stupid_approach import (
    PendulumStupidAlgorithm,
)
from programmatic_policy_learning.approaches.residual_approach import ResidualApproach

ENV_ID = "Pendulum-v1"
SEED = 42
TRAIN_STEPS = 10_000
EVAL_EPISODES = 5
BACKENDS = ["sb3-td3", "sb3-ddpg"]


def build_env() -> gym.Env:
    return gym.make(ENV_ID)


def eval_policy(
    approach, n_episodes: int = EVAL_EPISODES, seed: int = SEED
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
            # silence any print() from the base policy when it computes its action
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                approach._last_observation = np.asarray(obs, dtype=np.float32)
                a = approach._get_action()
            obs, r, terminated, truncated, _ = env.step(a)
            ep_ret += float(r)
        env.close()  # type: ignore[no-untyped-call]
        returns.append(ep_ret)
    return np.asarray(returns, dtype=np.float32)


# def _sb3_has_sac() -> bool:
#     try:
#         from stable_baselines3 import SAC  # noqa: F401
#         return True
#     except Exception:
#         return False


@pytest.mark.parametrize("backend", BACKENDS)
def test_residual_vs_base_runs(backend: str) -> None:

    tmp = build_env()
    base = PendulumStupidAlgorithm(
        "base", tmp.observation_space, tmp.action_space, seed=SEED
    )
    tmp.close()

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
    tmp.close()

    try:
        residual._backend._model.learn(total_timesteps=TRAIN_STEPS, progress_bar=False)
    except Exception:
        residual.train()

    residual_returns = eval_policy(residual, n_episodes=EVAL_EPISODES, seed=SEED + 123)
    assert residual_returns.shape == (EVAL_EPISODES,)
    assert np.isfinite(residual_returns).all()

    # Soft sanity: residual shouldn't be catastrophically worse than base
    assert residual_returns.mean() > base_returns.mean() - 100.0

    # ----- PRINT SUMMARY (force to real stdout) -----
    b_mean, b_std = float(base_returns.mean()), float(base_returns.std())
    r_mean, r_std = float(residual_returns.mean()), float(residual_returns.std())
    sys.__stdout__.write("\n=== Summary (higher / less negative is better) ===\n")
    sys.__stdout__.write(f"Backend: {backend}\n")
    sys.__stdout__.write(
        f"Base-only         : mean {b_mean:8.1f} ± {b_std:5.1f} over {EVAL_EPISODES} eps\n"
    )
    sys.__stdout__.write(
        f"Residual (base+R) : mean {r_mean:8.1f} ± {r_std:5.1f} over {EVAL_EPISODES} eps\n"
    )
    sys.__stdout__.flush()
