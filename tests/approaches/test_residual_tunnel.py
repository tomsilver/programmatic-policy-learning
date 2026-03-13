# test_residual_tunnel.py
from __future__ import annotations

import copy
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from programmatic_policy_learning.approaches.residual_tunnel import (
    BreakoutRAMTracker,
    BreakoutTunnelFeatureObsWrapper,
    ResidualTunnelControlWrapper,
)

ExpertPolicy = Callable[[Any], int]


# ============================
# Config: eval over 5 seeds
# ============================
EVAL_SEEDS = [0, 1, 2, 3, 4]  # <- 5 seeds


def make_breakout_expert_ram(*, serve_steps: int = 10, deadband_px: float = 2.0) -> ExpertPolicy:
    """RAM expert: serves, then tracks ball_x."""
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


class TrainDebugCallback(BaseCallback):
    def __init__(self, print_every: int = 2000, window: int = 2000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.print_every = int(print_every)
        self.window = int(window)
        self._used = deque(maxlen=self.window)
        self._hit = deque(maxlen=self.window)
        self._raw = deque(maxlen=self.window)
        self._tot = deque(maxlen=self.window)
        self._above = deque(maxlen=self.window)
        self._edge = deque(maxlen=self.window)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if not infos or not isinstance(infos, (list, tuple)) or not isinstance(infos[0], dict):
            return True
        info0 = infos[0]
        self._used.append(1.0 if info0.get("used_residual", False) else 0.0)
        self._hit.append(1.0 if info0.get("hit_detected", False) else 0.0)
        self._raw.append(float(info0.get("reward_raw", 0.0)))
        self._tot.append(float(info0.get("reward_total", 0.0)))
        self._above.append(1.0 if info0.get("above_bricks", False) else 0.0)
        self._edge.append(float(info0.get("edge_score", 0.0)))

        if self.num_timesteps % self.print_every == 0:
            print(
                f"[train] t={self.num_timesteps} "
                f"used={np.mean(self._used):.3f} hit={np.mean(self._hit):.3f} "
                f"above={np.mean(self._above):.3f} edge_score={np.mean(self._edge):.3f} "
                f"mean_raw={np.mean(self._raw):.4f} mean_total={np.mean(self._tot):.4f}"
            )
        return True


def make_env(env_id: str, *, seed: int, render_mode: str | None = None) -> gym.Env:
    import ale_py

    gym.register_envs(ale_py)

    env = gym.make(env_id, obs_type="ram", render_mode=render_mode)
    env.reset(seed=seed)

    tracker = BreakoutRAMTracker()
    base = make_breakout_expert_ram(serve_steps=10, deadband_px=2.0)

    def base_np(obs_any: Any) -> int:
        return int(base(np.asarray(obs_any)))

    env_ctrl = ResidualTunnelControlWrapper(
        env,
        base_policy=base_np,
        tracker=tracker,
        # control
        gate_px=20.0,
        delta_x_max=6.0,
        cooldown_steps=8,
        serve_steps=10,
        # tunnel shaping (TUNE THESE)
        y_brick=60.0,
        edge_band=12.0,
        k_above=0.002,
        k_edge=0.004,
        # posthit shaping (light)
        posthit_window=35,
        k_post_vx=0.01,
        k_post_up=0.02,
        k_u=0.0007,
        debug=True,
    )

    env_feat = BreakoutTunnelFeatureObsWrapper(env_ctrl, tracker)
    return env_feat


@dataclass
class EvalStats:
    mean_raw: float
    std_raw: float
    mean_total: float
    std_total: float


def evaluate_model_with_vecnorm(
    env_id: str,
    model: PPO,
    train_vecnorm: VecNormalize,
    *,
    seed: int,
    num_episodes: int,
    max_steps: int,
    save_gif: bool = False,
    gif_path: str = "breakout_tunnel_eval_ep0.gif",
    gif_episode_index: int = 0,
    stride: int = 4,
    fps: int = 30,
) -> EvalStats:
    """
    Evaluate for a single seed (returns mean/std across episodes).
    """
    frames: list[np.ndarray] = []

    def _normalize_obs(obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        rms = train_vecnorm.obs_rms
        eps = float(getattr(train_vecnorm, "epsilon", 1e-8))
        clip = float(getattr(train_vecnorm, "clip_obs", 10.0))
        obs_norm = (obs - rms.mean) / np.sqrt(rms.var + eps)
        obs_norm = np.clip(obs_norm, -clip, clip)
        return obs_norm.astype(np.float32)

    eval_venv = DummyVecEnv([lambda: make_env(env_id, seed=seed, render_mode=None)])
    eval_env = VecNormalize(eval_venv, norm_obs=True, norm_reward=True, clip_reward=10.0)
    eval_env.obs_rms = copy.deepcopy(train_vecnorm.obs_rms)
    eval_env.ret_rms = copy.deepcopy(train_vecnorm.ret_rms)
    eval_env.training = False
    eval_env.norm_reward = False  # true reward

    raw_returns: List[float] = []
    total_returns: List[float] = []

    gif_env = None
    prev_lives_g = None
    obs_g = None
    if save_gif:
        gif_env = make_env(env_id, seed=seed + 999_999, render_mode="rgb_array")
        obs_g, info_g = gif_env.reset(seed=seed + 999_999)
        prev_lives_g = info_g.get("lives", None)

    for ep in range(num_episodes):
        eval_env.env_method("reset", seed=int(seed + 10_000 + ep))
        obs = eval_env.reset()

        raw_ret = 0.0
        tot_ret = 0.0

        for t in range(max_steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, r_total, done, infos = eval_env.step(a)

            tot_ret += float(r_total[0])
            info0 = infos[0] if isinstance(infos, (list, tuple)) and infos else {}
            raw_ret += float(info0.get("reward_raw", 0.0))

            if gif_env is not None and ep == gif_episode_index:
                if t % stride == 0:
                    frame = gif_env.render()
                    if frame is not None:
                        frames.append(frame)

                assert obs_g is not None
                obs_g_norm = _normalize_obs(np.asarray(obs_g, dtype=np.float32))
                a_g, _ = model.predict(obs_g_norm, deterministic=True)
                obs_g, _, term_g, trunc_g, info_g = gif_env.step(int(a_g))

                lives_g = info_g.get("lives", None)

                # Episodic-life: keep going until lives==0
                if prev_lives_g is not None and lives_g is not None and lives_g < prev_lives_g:
                    term_g = False
                if lives_g is not None and lives_g > 0:
                    term_g = False
                if lives_g is not None and lives_g == 0:
                    term_g = True

                prev_lives_g = lives_g

                if term_g or trunc_g:
                    break

            if bool(done[0]):
                break

        raw_returns.append(raw_ret)
        total_returns.append(tot_ret)

    eval_env.close()
    if gif_env is not None:
        gif_env.close()

    if save_gif and frames:
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"Saved GIF to {gif_path} ({len(frames)} frames, fps={fps})")

    raw_arr = np.asarray(raw_returns, dtype=np.float32)
    tot_arr = np.asarray(total_returns, dtype=np.float32)
    return EvalStats(
        mean_raw=float(raw_arr.mean()),
        std_raw=float(raw_arr.std(ddof=0)),
        mean_total=float(tot_arr.mean()),
        std_total=float(tot_arr.std(ddof=0)),
    )


def evaluate_over_seeds(
    env_id: str,
    model: PPO,
    train_vecnorm: VecNormalize,
    *,
    base_seed: int,
    seeds: List[int],
    num_episodes: int,
    max_steps: int,
    save_gif: bool = False,
    gif_path: str = "logs/breakout_tunnel_eval_seed0.gif",
) -> EvalStats:
    """
    Evaluate across multiple seeds:
      - For each seed: run evaluate_model_with_vecnorm -> get per-seed mean
      - Aggregate mean±std across seeds (of those per-seed means)
    """
    per_seed_raw_means: List[float] = []
    per_seed_tot_means: List[float] = []

    for j, s in enumerate(seeds):
        # Make each seed distinct per eval point by offsetting with base_seed
        seed_j = int(base_seed + 10_000 * s)

        stats_j = evaluate_model_with_vecnorm(
            env_id,
            model,
            train_vecnorm,
            seed=seed_j,
            num_episodes=num_episodes,
            max_steps=max_steps,
            save_gif=(save_gif and j == 0),  # only seed[0] produces a gif
            gif_path=gif_path,
            gif_episode_index=0,
        )
        per_seed_raw_means.append(stats_j.mean_raw)
        per_seed_tot_means.append(stats_j.mean_total)

    raw_means = np.asarray(per_seed_raw_means, dtype=np.float32)
    tot_means = np.asarray(per_seed_tot_means, dtype=np.float32)

    # std here is variability ACROSS seeds
    return EvalStats(
        mean_raw=float(raw_means.mean()),
        std_raw=float(raw_means.std(ddof=0)),
        mean_total=float(tot_means.mean()),
        std_total=float(tot_means.std(ddof=0)),
    )


def main() -> None:
    env_id = "ALE/Breakout-v5"
    seed = 0

    total_timesteps = 300_000
    train_chunk = 10_000
    num_evals = max(1, total_timesteps // train_chunk)

    # episodes per SEED (so total eval episodes = num_eval_episodes * len(EVAL_SEEDS))
    num_eval_episodes = 10
    max_eval_steps = 60_000

    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/breakout_tunnel_train_eval_points.csv"
    plot_path = "logs/breakout_tunnel_train_eval_curve.png"

    raw_vec = DummyVecEnv([lambda: make_env(env_id, seed=seed, render_mode=None)])
    vec_env = VecNormalize(raw_vec, norm_obs=True, norm_reward=True, clip_reward=10.0)

    model = PPO(
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

    cb = TrainDebugCallback(print_every=2000, window=2000)

    points: List[Dict[str, float]] = []

    train_vecnorm = model.get_vec_normalize_env()
    assert train_vecnorm is not None

    # Phase 0 eval (over 5 seeds)
    stats0 = evaluate_over_seeds(
        env_id,
        model,
        train_vecnorm,
        base_seed=seed + 123,
        seeds=EVAL_SEEDS,
        num_episodes=num_eval_episodes,
        max_steps=max_eval_steps,
        save_gif=True,
        gif_path="logs/breakout_tunnel_eval_seed0_phase0.gif",
    )
    print(
        f"[eval] t=0 (over {len(EVAL_SEEDS)} seeds) "
        f"raw={stats0.mean_raw:.2f}±{stats0.std_raw:.2f} total={stats0.mean_total:.2f}±{stats0.std_total:.2f}"
    )
    points.append(
        dict(
            timesteps=0,
            mean_raw=stats0.mean_raw,
            std_raw=stats0.std_raw,
            mean_total=stats0.mean_total,
            std_total=stats0.std_total,
        )
    )

    trained = 0
    for i in range(1, num_evals + 1):
        steps_this = min(train_chunk, total_timesteps - trained)
        if steps_this <= 0:
            break

        print(f"\n=== Train chunk {i}/{num_evals}: {steps_this} timesteps ===")
        model.learn(total_timesteps=steps_this, progress_bar=True, callback=cb)
        trained += steps_this

        train_vecnorm = model.get_vec_normalize_env()
        assert train_vecnorm is not None

        stats = evaluate_over_seeds(
            env_id,
            model,
            train_vecnorm,
            base_seed=seed + 123 + i,
            seeds=EVAL_SEEDS,
            num_episodes=num_eval_episodes,
            max_steps=max_eval_steps,
            save_gif=(i in {3, 6, 12}),
            gif_path=f"logs/breakout_tunnel_eval_seed0_phase{i}.gif",
        )
        print(
            f"[eval] t={trained} (over {len(EVAL_SEEDS)} seeds) "
            f"raw={stats.mean_raw:.2f}±{stats.std_raw:.2f} total={stats.mean_total:.2f}±{stats.std_total:.2f}"
        )
        points.append(
            dict(
                timesteps=float(trained),
                mean_raw=stats.mean_raw,
                std_raw=stats.std_raw,
                mean_total=stats.mean_total,
                std_total=stats.std_total,
            )
        )

        # plot points
        df = pd.DataFrame(points)
        plt.figure(figsize=(7, 5))
        x = df["timesteps"].values

        plt.errorbar(x, df["mean_raw"].values, yerr=df["std_raw"].values, marker="o", capsize=4, label="RAW return (mean±std across seeds)")

        plt.xlabel("Training timesteps")
        plt.ylabel("Return")
        plt.title(f"Breakout tunnel residual: train → eval (over {len(EVAL_SEEDS)} seeds)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()

        df.to_csv(csv_path, index=False)

    print(f"\nSaved points CSV to {csv_path}")
    print(f"Saved plot to {plot_path}")

    model.save("ppo_breakout_tunnel_residual_ram")
    vec_env.save("ppo_breakout_tunnel_residual_ram_vecnorm.pkl")
    vec_env.close()


if __name__ == "__main__":
    main()
