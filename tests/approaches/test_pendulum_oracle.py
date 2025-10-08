"""Tests for the parametric pendulum controller (grid search + GIF
rendering)."""

import math
from typing import Any, Callable, Iterable, List, Tuple, cast

import gymnasium
import imageio
import numpy as np

from programmatic_policy_learning.approaches.pendulum_oracle import PendulumParametric


def make_env(render: bool = False):
    """Create a Pendulum-v1 env, optionally with RGB rendering."""
    return gymnasium.make("Pendulum-v1", render_mode="rgb_array" if render else None)


def _theta_from_obs(obs: np.ndarray) -> float:
    """Return theta from (cos(theta), sin(theta), theta_dot) observation."""
    return float(np.arctan2(float(obs[1]), float(obs[0])))


def rollout(
    env,
    policy,
    steps: int,
    seed: int | None,
    collect_frames: bool = False,
    min_abs_theta: float = 0.10,
) -> Tuple[float, list[np.ndarray]]:
    """Run one episode rollout; optionally collect rendered frames."""
    if seed is None:
        obs, info = env.reset()
    else:
        obs, info = env.reset(seed=int(seed))

    if min_abs_theta > 0.0 and abs(_theta_from_obs(obs)) < min_abs_theta:
        obs, info = env.reset()

    policy.reset(obs, info)
    total = 0.0
    frames: list[np.ndarray] = []

    for _ in range(steps):
        action = policy.step()
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        done = bool(terminated or truncated)
        policy.update(obs, float(reward), done, info)
        if collect_frames:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        if done:
            break

    return float(total), frames


def evaluate_avg_and_best(
    policy_factory: Callable[[], object],
    steps: int,
    seeds: Iterable[int | None],
    min_abs_theta: float = 0.10,
) -> Tuple[float, float, int | None]:
    """Evaluate average and best return across seeds."""
    returns: list[float] = []
    best_ret: float = -math.inf
    best_seed: int | None = None
    for seed in seeds:
        env = make_env(render=False)
        try:
            pi = policy_factory()
            ret, _ = rollout(
                env,
                pi,
                steps=steps,
                seed=seed,
                collect_frames=False,
                min_abs_theta=min_abs_theta,
            )
            returns.append(float(ret))
            if ret > best_ret:
                best_ret, best_seed = float(ret), seed
        finally:
            env.close()  # type: ignore[no-untyped-call]
    avg_ret = float(np.mean(returns)) if returns else -math.inf
    return avg_ret, float(best_ret), best_seed


def render_gif(
    policy_factory: Callable[[], object],
    steps: int,
    seed: int | None,
    path: str,
    min_abs_theta: float = 0.10,
) -> float:
    """Render a GIF of the rollout to `path` and return its return."""
    env = make_env(render=True)
    try:
        pi = policy_factory()
        ret, frames = rollout(
            env,
            pi,
            steps=steps,
            seed=seed,
            collect_frames=True,
            min_abs_theta=min_abs_theta,
        )
    finally:
        env.close()
    imageio.mimsave(path, cast(list[Any], frames), duration=33.0)
    return float(ret)


def grid_search_kp(
    candidates: Iterable[float],
    _episodes: int,
    steps: int,
    seeds: List[int | None],
    min_abs_theta: float = 0.10,
) -> Tuple[float, float]:
    """Grid search over proportional gain `kp` and return (best_kp,
    best_avg)."""
    best_kp: float | None = None
    best_score: float = -math.inf

    for kp in candidates:
        kp_val = float(kp)

        def pf(kp_val=kp_val):
            env_tmp = make_env(render=False)
            try:
                return PendulumParametric(
                    "PendulumParametric",
                    env_tmp.observation_space,
                    env_tmp.action_space,
                    kp=kp_val,
                )
            finally:
                env_tmp.close()

        avg_ret, _, _ = evaluate_avg_and_best(
            pf,
            steps=steps,
            seeds=seeds,
            min_abs_theta=min_abs_theta,
        )
        if avg_ret > best_score:
            best_score, best_kp = float(avg_ret), kp_val

    assert best_kp is not None
    return float(best_kp), float(best_score)


def test_kp_grid_search_and_gif():
    """Tune kp, evaluate, and save a GIF at the best seed (smoke test)."""
    episodes = 7
    steps = 350
    min_abs_theta = 0.10

    rng = np.random.default_rng()
    seeds: List[int | None] = [
        int(s) for s in rng.integers(0, 1_000_000, size=episodes)
    ]

    kp_candidates = np.linspace(6.0, 18.0, num=7)
    best_kp, tuned_grid_avg = grid_search_kp(
        candidates=kp_candidates,
        _episodes=episodes,
        steps=steps,
        seeds=seeds,
        min_abs_theta=min_abs_theta,
    )
    print(f"best kp={best_kp:.2f} | tuned_grid_avg={tuned_grid_avg:.2f}")

    def tunedval():
        env_tmp = make_env(render=False)
        try:
            return PendulumParametric(
                "PendulumParametric",
                env_tmp.observation_space,
                env_tmp.action_space,
                kp=best_kp,
            )
        finally:
            env_tmp.close()

    tuned_avg, tuned_best, tuned_best_seed = evaluate_avg_and_best(
        tunedval,
        steps=steps,
        seeds=seeds,
        min_abs_theta=min_abs_theta,
    )
    print(f"kp={best_kp:.2f} | avg_return={tuned_avg:.2f}")
    print(f"best_single={tuned_best:.2f} (seed={tuned_best_seed})")

    tuned_gif = f"pendulum_tuned_kp_{best_kp:.2f}.gif"
    tuned_best_return = render_gif(
        tunedval,
        steps=steps,
        seed=tuned_best_seed,
        path=tuned_gif,
        min_abs_theta=min_abs_theta,
    )
    print(f"saved {tuned_gif} | ret={tuned_best_return:.2f}")
