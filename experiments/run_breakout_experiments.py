# experiments/run_breakout_backends_rawonly.py
"""
Compare backends for Breakout residual-hitpoint learning:
  - PPO
  - TD3
  - SAC

What it does:
  * Builds ALE/Breakout-v5 with obs_type="ram"
  * Wraps with your ResidualHitPointControlWrapper + BreakoutHitPointFeatureObsWrapper
  * Trains each backend for a meaningful horizon (default: 2,000,000 timesteps)
  * Evaluates at t=0 and every train_chunk on MULTIPLE env seeds
  * Logs + plots ONLY RAW rewards (from info["reward_raw"])

Outputs:
  - logs/breakout_backend_compare_raw_points.csv
  - logs/breakout_backend_compare_raw_curve.png

Notes:
  - Uses VecNormalize with norm_obs=True and norm_reward=False for stability with off-policy replay.
  - All backends optimize the wrapper's returned reward (shaped), but we report RAW only.
"""

from __future__ import annotations

import copy
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# IMPORTANT: use YOUR current module path here
from programmatic_policy_learning.approaches.residual_discrete_approach import (
    BreakoutRAMTracker,
    BreakoutHitPointFeatureObsWrapper,
    ResidualHitPointControlWrapper,
)

ExpertPolicy = Callable[[Any], int]
Backend = Literal["ppo", "td3", "sac"]


# ============================================================
# Expert (RAM): serve + follow ball_x
# ============================================================
def make_breakout_expert_ram(*, serve_steps: int = 10, deadband_px: float = 2.0) -> ExpertPolicy:
    NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3
    steps = 0
    tracker = BreakoutRAMTracker()

    def reset() -> None:
        nonlocal steps
        steps = 0
        tracker.reset()

    def expert(obs_ram: Any) -> int:
        nonlocal steps
        steps += 1
        if steps <= serve_steps:
            return FIRE

        st = tracker.update(np.asarray(obs_ram))
        if not st.ok or st.paddle_x is None or st.ball_x is None:
            return NOOP

        dx = float(st.ball_x - st.paddle_x)
        if dx > deadband_px:
            return RIGHT
        if dx < -deadband_px:
            return LEFT
        return NOOP

    expert.reset = reset  # type: ignore[attr-defined]
    return expert


# ============================================================
# Debug callback (PPO only)
# ============================================================
class TrainDebugCallback(BaseCallback):
    def __init__(self, print_every: int = 2000, window: int = 2000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.print_every = int(print_every)
        self.window = int(window)
        self._used = deque(maxlen=self.window)
        self._hit = deque(maxlen=self.window)
        self._raw = deque(maxlen=self.window)
        self._tot = deque(maxlen=self.window)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if not infos or not isinstance(infos, (list, tuple)) or not isinstance(infos[0], dict):
            return True
        info0 = infos[0]

        self._used.append(1.0 if info0.get("used_residual", False) else 0.0)
        self._hit.append(1.0 if info0.get("hit_detected", False) else 0.0)
        self._raw.append(float(info0.get("reward_raw", 0.0)))
        self._tot.append(float(info0.get("reward_total", 0.0)))

        if self.num_timesteps % self.print_every == 0:
            used_rate = float(np.mean(self._used)) if self._used else 0.0
            hit_rate = float(np.mean(self._hit)) if self._hit else 0.0
            mean_raw = float(np.mean(self._raw)) if self._raw else 0.0
            mean_tot = float(np.mean(self._tot)) if self._tot else 0.0
            print(
                f"[train] t={self.num_timesteps} "
                f"used_rate={used_rate:.3f} hit_rate={hit_rate:.3f} "
                f"mean_raw_rew={mean_raw:.4f} mean_total_rew={mean_tot:.4f}"
            )
        return True


# ============================================================
# Env factory (your residual-hitpoint setup)
# ============================================================
def make_env(env_id: str, *, seed: int, render_mode: str | None = None) -> gym.Env:
    import ale_py

    gym.register_envs(ale_py)

    env = gym.make(env_id, obs_type="ram", render_mode=render_mode)
    env.reset(seed=seed)

    tracker = BreakoutRAMTracker()
    base = make_breakout_expert_ram(serve_steps=10, deadband_px=2.0)

    def base_np(obs_any: Any) -> int:
        return int(base(np.asarray(obs_any)))

    env_ctrl = ResidualHitPointControlWrapper(
        env,
        base_policy=base_np,
        tracker=tracker,
        gate_px=20.0,
        delta_x_max=6.0,
        cooldown_steps=8,
        posthit_window=45,
        k_post_y=0.07,
        k_post_vx=0.02,
        k_u=0.0007,
        trigger_frac=0.90,
        trigger_margin=0.0,
        serve_steps=10,
        debug=True,
    )

    env_feat = BreakoutHitPointFeatureObsWrapper(env_ctrl, tracker)
    return env_feat


# ============================================================
# Evaluation (multi-seed) — RAW only (from info["reward_raw"])
# ============================================================
@dataclass
class EvalStats:
    mean_raw: float
    std_raw: float


def evaluate_single_seed_raw(
    env_id: str,
    model: Any,
    train_vecnorm: VecNormalize,
    *,
    seed: int,
    num_episodes: int,
    max_steps: int,
) -> float:
    """
    Returns mean RAW return across episodes for one seed.
    We sum info["reward_raw"] (the unshaped game score increments).
    """
    eval_venv = DummyVecEnv([lambda: make_env(env_id, seed=seed, render_mode=None)])
    eval_env = VecNormalize(eval_venv, norm_obs=True, norm_reward=False, clip_reward=10.0)

    # Copy obs normalization stats
    eval_env.obs_rms = copy.deepcopy(train_vecnorm.obs_rms)
    eval_env.training = False
    eval_env.norm_reward = False

    ep_returns: List[float] = []

    for ep in range(num_episodes):
        eval_env.env_method("reset", seed=int(seed + 10_000 + ep))
        obs = eval_env.reset()

        raw_ret = 0.0
        for _ in range(max_steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = eval_env.step(a)

            info0 = infos[0] if isinstance(infos, (list, tuple)) and infos else {}
            raw_ret += float(info0.get("reward_raw", 0.0))

            if bool(done[0]):
                break

        ep_returns.append(float(raw_ret))

    eval_env.close()
    return float(np.mean(ep_returns)) if ep_returns else 0.0


def evaluate_over_seeds_raw(
    env_id: str,
    model: Any,
    train_vecnorm: VecNormalize,
    *,
    base_seed: int,
    seeds: List[int],
    num_episodes_per_seed: int,
    max_steps: int,
) -> EvalStats:
    """
    For each seed -> mean raw return over episodes.
    Then mean±std across seeds (std reflects seed-to-seed variability).
    """
    per_seed_means: List[float] = []
    for s in seeds:
        seed_s = int(base_seed + 10_000 * s)
        m = evaluate_single_seed_raw(
            env_id,
            model,
            train_vecnorm,
            seed=seed_s,
            num_episodes=num_episodes_per_seed,
            max_steps=max_steps,
        )
        per_seed_means.append(m)

    arr = np.asarray(per_seed_means, dtype=np.float32)
    return EvalStats(mean_raw=float(arr.mean()), std_raw=float(arr.std(ddof=0)))


# ============================================================
# Model factory (PPO / TD3 / SAC)
# ============================================================
def make_model(backend: Backend, vec_env: VecNormalize, *, seed: int):
    if backend == "ppo":
        return PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=5e-5,
            n_steps=2048,
            batch_size=512,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            seed=seed,
            use_sde=True,
            sde_sample_freq=4,
            policy_kwargs=dict(log_std_init=-1.0),
        )

    if backend == "td3":
        n_actions = int(vec_env.action_space.shape[0])
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))
        return TD3(
            "MlpPolicy",
            vec_env,
            learning_rate=1e-4,
            buffer_size=500_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            gradient_steps=1,
            action_noise=action_noise,
            verbose=1,
            seed=seed,
        )

    if backend == "sac":
        return SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=1e-4,
            buffer_size=500_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            verbose=1,
            seed=seed,
        )

    raise ValueError(f"Unknown backend: {backend}")


# ============================================================
# Main: multi-backend + multi-seed + raw-only plot
# ============================================================
def main() -> None:
    env_id = "ALE/Breakout-v5"
    seed = 0

    BACKENDS: List[Backend] = ["ppo", "td3", "sac"]

    # Training schedule (enough to be informative)
    total_timesteps = 2_000_000
    train_chunk = 100_000
    num_evals = max(1, total_timesteps // train_chunk)

    # Multi-seed eval config
    EVAL_SEEDS = [0, 1, 2, 3, 4]     # 5 env seeds
    num_eval_episodes_per_seed = 10  # 50 total episodes per checkpoint
    max_eval_steps = 50_000

    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/breakout_backend_compare_raw_points.csv"
    plot_path = "logs/breakout_backend_compare_raw_curve.png"

    points: List[Dict[str, float]] = []

    for backend in BACKENDS:
        print("\n===================================================")
        print(f"Backend: {backend}")
        print("===================================================")

        # Training env: norm_reward OFF (important for off-policy stability)
        raw_vec = DummyVecEnv([lambda: make_env(env_id, seed=seed, render_mode=None)])
        vec_env = VecNormalize(raw_vec, norm_obs=True, norm_reward=False, clip_reward=10.0)

        model = make_model(backend, vec_env, seed=seed)

        cb: Optional[BaseCallback] = TrainDebugCallback(print_every=2000, window=2000) if backend == "ppo" else None

        train_vecnorm = model.get_vec_normalize_env()
        assert train_vecnorm is not None

        # t=0 eval
        stats0 = evaluate_over_seeds_raw(
            env_id,
            model,
            train_vecnorm,
            base_seed=seed + 123,
            seeds=EVAL_SEEDS,
            num_episodes_per_seed=num_eval_episodes_per_seed,
            max_steps=max_eval_steps,
        )
        print(f"[eval] {backend} t=0 raw={stats0.mean_raw:.2f}±{stats0.std_raw:.2f}")
        points.append(dict(backend=backend, timesteps=0, mean_raw=stats0.mean_raw, std_raw=stats0.std_raw))

        trained = 0
        for i in range(1, num_evals + 1):
            steps_this = min(train_chunk, total_timesteps - trained)
            if steps_this <= 0:
                break

            print(f"\n=== [{backend}] Train chunk {i}/{num_evals}: {steps_this} timesteps ===")
            if cb is not None:
                model.learn(total_timesteps=steps_this, progress_bar=True, callback=cb)
            else:
                model.learn(total_timesteps=steps_this, progress_bar=True)
            trained += steps_this

            train_vecnorm = model.get_vec_normalize_env()
            assert train_vecnorm is not None

            stats = evaluate_over_seeds_raw(
                env_id,
                model,
                train_vecnorm,
                base_seed=seed + 123 + i,
                seeds=EVAL_SEEDS,
                num_episodes_per_seed=num_eval_episodes_per_seed,
                max_steps=max_eval_steps,
            )
            print(f"[eval] {backend} t={trained} raw={stats.mean_raw:.2f}±{stats.std_raw:.2f}")
            points.append(dict(backend=backend, timesteps=float(trained), mean_raw=stats.mean_raw, std_raw=stats.std_raw))

        vec_env.close()

    # Save + plot combined
    df = pd.DataFrame(points)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")

    plt.figure(figsize=(7, 5))
    for backend, sub in df.groupby("backend"):
        x = sub["timesteps"].values
        y = sub["mean_raw"].values
        yerr = sub["std_raw"].values
        plt.plot(x, y, marker="o", label=backend)
        if not np.all(np.isnan(yerr)):
            plt.fill_between(x, y - yerr, y + yerr, alpha=0.2)

    plt.xlabel("Training timesteps")
    plt.ylabel("RAW return (mean over seeds)")
    plt.title(f"Breakout residual-hitpoint: PPO vs TD3 vs SAC (RAW only, {len([0,1,2,3,4])} seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
