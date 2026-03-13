# experiments/run_breakout_llm_expert_family_sweep.py
"""
Sweep through ALL generated LLM experts in experiments/breakout_llm_expert_*.py
and plot reward vs training timesteps (phased train -> eval), similar to your
prior sweep script.

What it does:
- Discovers expert files automatically (no hard-coded list).
- For each expert + run_seed:
    * builds ALE/Breakout-v5 (obs_type="ram")
    * wraps with your ResidualTunnelControlWrapper + BreakoutTunnelFeatureObsWrapper
    * trains PPO in phases
    * evaluates on fresh seeds after each phase
    * logs raw return (from info["reward_raw"]) and total (wrapper reward)
- Aggregates across run_seeds and makes a combined reward-vs-timesteps plot.

Outputs:
- logs/llm_family/<expert_name>/breakout_<expert_name>_seed<seed>_phased_episodes.csv
- logs/llm_family/<expert_name>/breakout_<expert_name>_seed<seed>_phased_summary.csv
- logs/llm_family/breakout_llm_family_raw_curve.png
(Optionally also total curve if you enable it below)
"""

from __future__ import annotations

import copy
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from programmatic_policy_learning.approaches.residual_tunnel import (
    BreakoutRAMTracker,
    BreakoutTunnelFeatureObsWrapper,
    ResidualTunnelControlWrapper,
)

ExpertPolicy = Callable[[Any], int]

# -----------------------------
# Config
# -----------------------------
GLOBAL_SEED = 0
N_RUN_SEEDS = 5

ENV_ID = "ALE/Breakout-v5"

# Training schedule
NUM_TRAIN_PHASES = 10
TRAIN_STEPS_PER_PHASE = 50_000  # total = 1,000,000 steps by default

# Evaluation schedule
NUM_EVAL_EPISODES = 10
MAX_EVAL_STEPS = 100_000

# Plotting: raw only by default (as you requested earlier)
PLOT_TOTAL_TOO = False

# PPO params (same vibe as your script)
PPO_KWARGS = dict(
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
    use_sde=True,
    sde_sample_freq=4,
    policy_kwargs=dict(log_std_init=-1.0),
)

# VecNormalize config
VECNORM_KWARGS = dict(
    norm_obs=True,
    norm_reward=True,
    clip_reward=10.0,
)

# Where your generated experts live
EXPERT_DIR = Path("experiments")
EXPERT_GLOB = "breakout_llm_expert_*.py"

# -----------------------------
# Expert loading
# -----------------------------
def discover_llm_experts() -> List[Tuple[str, Path]]:
    """Return [(expert_name, path), ...] sorted by name."""
    paths = sorted(EXPERT_DIR.glob(EXPERT_GLOB))
    out: List[Tuple[str, Path]] = []
    for p in paths:
        name = p.stem.replace("breakout_llm_expert_", "")
        # skip accidental files
        if name.strip() == "" or name.startswith("_"):
            continue
        out.append((name, p))
    if not out:
        raise RuntimeError(f"No experts found at {EXPERT_DIR}/{EXPERT_GLOB}")
    return out


def load_expert_policy_from_file(py_path: Path) -> ExpertPolicy:
    """Dynamically import a module from a file and return expert_policy."""
    spec = importlib.util.spec_from_file_location(py_path.stem, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    if not hasattr(mod, "expert_policy"):
        raise RuntimeError(f"{py_path} does not define expert_policy")
    fn = getattr(mod, "expert_policy")
    if not callable(fn):
        raise RuntimeError(f"{py_path} expert_policy is not callable")
    return fn  # type: ignore[return-value]


# -----------------------------
# Env factory
# -----------------------------
def make_raw_breakout_env(seed: int, render_mode: str | None = None) -> gym.Env:
    import ale_py  # noqa: F401

    gym.register_envs(__import__("ale_py"))
    env = gym.make(ENV_ID, obs_type="ram", render_mode=render_mode)
    env.reset(seed=seed)
    return env


def make_wrapped_env(
    *,
    seed: int,
    expert_policy: ExpertPolicy,
    render_mode: str | None = None,
) -> gym.Env:
    env = make_raw_breakout_env(seed=seed, render_mode=render_mode)

    tracker = BreakoutRAMTracker()

    def base_np(obs_any: Any) -> int:
        return int(expert_policy(np.asarray(obs_any)))

    env_ctrl = ResidualTunnelControlWrapper(
        env,
        base_policy=base_np,
        tracker=tracker,
        # control
        gate_px=20.0,
        delta_x_max=6.0,
        cooldown_steps=8,
        serve_steps=10,
        # tunnel shaping (keep same as your prior script defaults)
        y_brick=60.0,
        edge_band=12.0,
        k_above=0.002,
        k_edge=0.004,
        # posthit shaping
        posthit_window=35,
        k_post_vx=0.01,
        k_post_up=0.02,
        k_u=0.0007,
        debug=False,
    )

    env_feat = BreakoutTunnelFeatureObsWrapper(env_ctrl, tracker)
    return env_feat


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_policy(
    model: PPO,
    train_vecnorm: VecNormalize,
    *,
    expert_policy: ExpertPolicy,
    seed: int,
    num_episodes: int,
    max_steps: int,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    rng = np.random.default_rng(seed)

    # single-env eval venv; new seed each episode via reset()
    def _eval_env_fn() -> gym.Env:
        s = int(rng.integers(0, 1_000_000))
        return make_wrapped_env(seed=s, expert_policy=expert_policy, render_mode=None)

    eval_venv = DummyVecEnv([_eval_env_fn])
    eval_env = VecNormalize(eval_venv, **VECNORM_KWARGS)

    # Copy stats from training
    eval_env.obs_rms = copy.deepcopy(train_vecnorm.obs_rms)
    eval_env.ret_rms = copy.deepcopy(train_vecnorm.ret_rms)
    eval_env.training = False
    eval_env.norm_reward = False  # report true scale

    for ep in range(num_episodes):
        obs = eval_env.reset()
        raw_ret = 0.0
        tot_ret = 0.0
        steps = 0

        for _ in range(max_steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, infos = eval_env.step(a)

            tot_ret += float(r[0])
            info0 = infos[0] if infos and isinstance(infos, (list, tuple)) else {}
            raw_ret += float(info0.get("reward_raw", 0.0))

            steps += 1
            if bool(done[0]):
                break

        rows.append(
            dict(
                episode_idx=float(ep),
                raw_reward=float(raw_ret),
                total_reward=float(tot_ret),
                total_steps=float(steps),
            )
        )

    eval_env.close()
    return rows


# -----------------------------
# Per-expert, per-seed run
# -----------------------------
def run_phased_for_expert_seed(
    *,
    expert_name: str,
    expert_policy: ExpertPolicy,
    run_seed: int,
    out_root: Path,
) -> pd.DataFrame:
    out_dir = out_root / expert_name
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_csv = out_dir / f"breakout_{expert_name}_seed{run_seed}_phased_episodes.csv"
    summary_csv = out_dir / f"breakout_{expert_name}_seed{run_seed}_phased_summary.csv"

    # training env
    train_venv = DummyVecEnv([lambda: make_wrapped_env(seed=run_seed, expert_policy=expert_policy, render_mode=None)])
    vec_env = VecNormalize(train_venv, **VECNORM_KWARGS)

    model = PPO("MlpPolicy", vec_env, seed=run_seed, **PPO_KWARGS)

    train_vecnorm = model.get_vec_normalize_env()
    assert train_vecnorm is not None

    all_episode_rows: List[Dict[str, float]] = []
    cumulative_timesteps = 0

    # Phase 0 eval
    print(f"\n=== [{expert_name} | seed {run_seed}] Phase 0: t=0 ===")
    eps0 = evaluate_policy(
        model,
        train_vecnorm,
        expert_policy=expert_policy,
        seed=run_seed + 123,
        num_episodes=NUM_EVAL_EPISODES,
        max_steps=MAX_EVAL_STEPS,
    )
    for r in eps0:
        r["phase"] = 0.0
        r["timesteps"] = float(cumulative_timesteps)
        r["run_seed"] = float(run_seed)
        r["expert"] = expert_name
    all_episode_rows.extend(eps0)

    # Train/eval phases
    for phase in range(1, NUM_TRAIN_PHASES + 1):
        print(f"\n=== [{expert_name} | seed {run_seed}] Phase {phase}: train {TRAIN_STEPS_PER_PHASE} ===")
        model.learn(total_timesteps=TRAIN_STEPS_PER_PHASE, progress_bar=True)
        cumulative_timesteps += TRAIN_STEPS_PER_PHASE

        train_vecnorm = model.get_vec_normalize_env()
        assert train_vecnorm is not None

        eps = evaluate_policy(
            model,
            train_vecnorm,
            expert_policy=expert_policy,
            seed=run_seed + 123 + phase,
            num_episodes=NUM_EVAL_EPISODES,
            max_steps=MAX_EVAL_STEPS,
        )
        for r in eps:
            r["phase"] = float(phase)
            r["timesteps"] = float(cumulative_timesteps)
            r["run_seed"] = float(run_seed)
            r["expert"] = expert_name
        all_episode_rows.extend(eps)

    episodes_df = pd.DataFrame(all_episode_rows)
    episodes_df.to_csv(episode_csv, index=False)
    print(f"[{expert_name} | seed {run_seed}] Wrote {episode_csv}")

    summary = (
        episodes_df.groupby(["expert", "run_seed", "phase", "timesteps"])
        .agg(
            mean_raw=("raw_reward", "mean"),
            std_raw=("raw_reward", "std"),
            mean_total=("total_reward", "mean"),
            std_total=("total_reward", "std"),
            mean_steps=("total_steps", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(summary_csv, index=False)
    print(f"[{expert_name} | seed {run_seed}] Wrote {summary_csv}")

    vec_env.close()
    return summary


# -----------------------------
# Plotting (aggregate across run_seeds)
# -----------------------------
def plot_family_curves(all_summary: pd.DataFrame, out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    agg = (
        all_summary.groupby(["expert", "phase", "timesteps"])
        .agg(
            mean_raw=("mean_raw", "mean"),
            std_raw=("mean_raw", "std"),
            mean_total=("mean_total", "mean"),
            std_total=("mean_total", "std"),
        )
        .reset_index()
    )

    # RAW curve plot
    plt.figure(figsize=(8, 5))
    for expert, sub in agg.groupby("expert"):
        x = sub["timesteps"].values
        y = sub["mean_raw"].values
        yerr = sub["std_raw"].values
        plt.plot(x, y, marker="o", label=expert)
        if not np.all(np.isnan(yerr)):
            plt.fill_between(x, y - yerr, y + yerr, alpha=0.18)
    plt.xlabel("Cumulative training timesteps")
    plt.ylabel("Raw return (mean over run seeds)")
    plt.title("Breakout LLM expert family: raw return vs training")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path = out_root / "breakout_llm_family_raw_curve.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")

    if PLOT_TOTAL_TOO:
        plt.figure(figsize=(8, 5))
        for expert, sub in agg.groupby("expert"):
            x = sub["timesteps"].values
            y = sub["mean_total"].values
            yerr = sub["std_total"].values
            plt.plot(x, y, marker="o", label=expert)
            if not np.all(np.isnan(yerr)):
                plt.fill_between(x, y - yerr, y + yerr, alpha=0.18)
        plt.xlabel("Cumulative training timesteps")
        plt.ylabel("Total return (shaped) (mean over run seeds)")
        plt.title("Breakout LLM expert family: total return vs training")
        plt.legend(fontsize=8)
        plt.tight_layout()
        out_path2 = out_root / "breakout_llm_family_total_curve.png"
        plt.savefig(out_path2, dpi=200)
        plt.close()
        print(f"Saved {out_path2}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    experts = discover_llm_experts()
    print("Discovered experts:")
    for name, path in experts:
        print(f"  - {name}: {path}")

    out_root = Path("logs") / "llm_family"
    out_root.mkdir(parents=True, exist_ok=True)

    all_summaries: List[pd.DataFrame] = []

    for (expert_name, expert_path) in experts:
        policy = load_expert_policy_from_file(expert_path)

        for run_idx in range(N_RUN_SEEDS):
            run_seed = GLOBAL_SEED + 1000 * run_idx
            print(f"\n########### {expert_name} | RUN {run_idx} (seed={run_seed}) ###########")
            summ = run_phased_for_expert_seed(
                expert_name=expert_name,
                expert_policy=policy,
                run_seed=run_seed,
                out_root=out_root,
            )
            all_summaries.append(summ)

    combined = pd.concat(all_summaries, ignore_index=True)
    combined_csv = out_root / "breakout_llm_family_combined_summary.csv"
    combined.to_csv(combined_csv, index=False)
    print(f"Wrote {combined_csv}")

    plot_family_curves(combined, out_root)


if __name__ == "__main__":
    # avoid MacOS OpenMP weirdness for some installs (optional)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()