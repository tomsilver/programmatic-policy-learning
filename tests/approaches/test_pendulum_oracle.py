import math
from typing import Callable, Iterable, List, Tuple

import gymnasium
import imageio
import numpy as np

from programmatic_policy_learning.approaches.pendulum_oracle import PendulumParametric


def make_env(render: bool = False):
    return gymnasium.make("Pendulum-v1", render_mode="rgb_array" if render else None)


def _theta_from_obs(obs: np.ndarray) -> float:
    return float(np.arctan2(float(obs[1]), float(obs[0])))


def rollout(
    env,
    policy,
    steps: int,
    seed: int | None,
    collect_frames: bool = False,
    min_abs_theta: float = 0.10,
) -> Tuple[float, List[np.ndarray]]:
    """Run one episode. Returns (total_reward, frames)."""
    if seed is None:
        obs, info = env.reset()
    else:
        obs, info = env.reset(seed=int(seed))

    if min_abs_theta > 0.0 and abs(_theta_from_obs(obs)) < min_abs_theta:
        obs, info = env.reset()

    policy.reset(obs, info)
    total = 0.0
    frames: List[np.ndarray] = []

    for _ in range(steps):
        action = policy.step()
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        done = terminated or truncated
        policy.update(obs, reward, done, info)
        if collect_frames:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        if done:
            break

    return total, frames


def evaluate_avg_and_best(
    policy_factory: Callable[[], object],
    episodes: int,
    steps: int,
    seeds: Iterable[int | None],
    min_abs_theta: float = 0.10,
) -> Tuple[float, float, int | None]:
    """Run multiple episodes; return (avg_return, best_single_return, best_seed_used)."""
    returns: List[float] = []
    best_ret, best_seed = -math.inf, None
    for seed in seeds:
        env = make_env(render=False)
        try:
            pi = policy_factory()
            ret, _ = rollout(
                env, pi, steps=steps, seed=seed, collect_frames=False, min_abs_theta=min_abs_theta
            )
            returns.append(ret)
            if ret > best_ret:
                best_ret, best_seed = ret, seed
        finally:
            env.close()
    avg_ret = float(np.mean(returns)) if returns else -math.inf
    return avg_ret, float(best_ret), best_seed


def render_gif(
    policy_factory: Callable[[], object],
    steps: int,
    seed: int | None,
    path: str,
    min_abs_theta: float = 0.10,
) -> float:
    """Render ONE episode to a GIF. Returns that episode's return."""
    env = make_env(render=True)
    try:
        pi = policy_factory()
        ret, frames = rollout(
            env, pi, steps=steps, seed=seed, collect_frames=True, min_abs_theta=min_abs_theta
        )
    finally:
        env.close()
    imageio.mimsave(path, frames, duration=33)
    return ret


def grid_search_kp(
    candidates: Iterable[float],
    episodes: int,
    steps: int,
    seeds: List[int | None],
    min_abs_theta: float = 0.10,
) -> Tuple[float, float]:
    """Return (best_kp, best_avg_return) via simple grid search on kp."""
    best_kp, best_score = None, -math.inf

    for kp in candidates:
        def pf():
            env_tmp = make_env(render=False)
            try:
                return PendulumParametric(
                    "PendulumParametric",
                    env_tmp.observation_space,
                    env_tmp.action_space,
                    kp=float(kp),
                )
            finally:
                env_tmp.close()

        avg_ret, _, _ = evaluate_avg_and_best(
            pf, episodes=episodes, steps=steps, seeds=seeds, min_abs_theta=min_abs_theta
        )
        if avg_ret > best_score:
            best_score, best_kp = avg_ret, float(kp)

    assert best_kp is not None
    return best_kp, best_score


def test_kp_grid_search_and_gif():
    """
    Tune ONE parameter (kp) for the minimal policy, report averages and best single,
    and save a GIF for the best-seed episode.
    """
    episodes = 7
    steps = 350
    min_abs_theta = 0.10 

    rng = np.random.default_rng()
    seeds: List[int | None] = [int(s) for s in rng.integers(0, 1_000_000, size=episodes)]

    kp_candidates = np.linspace(6.0, 18.0, num=7) 

    best_kp, tuned_grid_avg = grid_search_kp(
        candidates=kp_candidates,
        episodes=episodes,
        steps=steps,
        seeds=seeds,
        min_abs_theta=min_abs_theta,
    )
    print(f"[TUNING] best kp={best_kp:.2f} | tuned_grid_avg={tuned_grid_avg:.2f}")

    def tuned_factory():
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
        tuned_factory, episodes=episodes, steps=steps, seeds=seeds, min_abs_theta=min_abs_theta
    )
    print(f"avg_return={tuned_avg:.2f} | best_single={tuned_best:.2f} (seed={tuned_best_seed})")

    tuned_gif = f"pendulum_tuned_kp_{best_kp:.2f}.gif"
    tuned_best_return = render_gif(
        tuned_factory, steps=steps, seed=tuned_best_seed, path=tuned_gif, min_abs_theta=min_abs_theta
    )
    print(f"return={tuned_best_return:.2f}")
