# experiments/run_pendulum_algorithms.py
"""Phased training script for Pendulum with a fixed pendulum expert as the base
policy, and multiple RL algorithms used as residual backends:

  - sb3-td3
  - sb3-ddpg
  - sb3-ppo
  - sb3-a2c
  - sb3-sac

For each backend and each random seed:
  * build ResidualApproach(expert=PendulumParametricPolicy, backend=<algo>)
  * phase 0: evaluate untrained residual
  * phases 1..N: train for TRAIN_STEPS_PER_PHASE, then evaluate
  * save:
      logs/pendulum_<algo>_seed<k>_phased_episodes.csv
      logs/pendulum_<algo>_seed<k>_phased_summary.csv
      (optional) per-seed plots

Afterwards, aggregate across seeds per algorithm to create combined plots:
  - logs/pendulum_algos_multi_reward_curve.png
  - logs/pendulum_algos_multi_upright_curve.png
"""

import os
from typing import Dict, List

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium.spaces import Box

from programmatic_policy_learning.approaches.experts.pendulum_experts import (
    PendulumParametricPolicy,
)
from programmatic_policy_learning.approaches.residual_approach import ResidualApproach

# -----------------------------
# Config
# -----------------------------

# Multiple random seeds for robustness
SEEDS = [0, 1, 2, 3, 4]

# Training phases:
NUM_TRAIN_PHASES = 20
TRAIN_STEPS_PER_PHASE = 500

# Evaluation config
NUM_EVAL_EPISODES = 100
MAX_EVAL_STEPS = 199
UPRIGHT_THRESHOLD = 0.1  # |theta| < this counts as upright

# RL backends to compare (must be supported by ResidualApproach/_SB3Backend)
BACKENDS = [
    "sb3-td3",
    "sb3-ddpg",
    "sb3-ppo",
    "sb3-a2c",
    "sb3-sac",
]


def make_pendulum_env(seed: int | None = None) -> gym.Env:
    env = gym.make("Pendulum-v1")
    if seed is not None:
        env.reset(seed=seed)
    return env


def build_pendulum_expert(action_space: Box) -> PendulumParametricPolicy:
    """Pendulum PID-ish expert used as base policy in all variants."""
    expert = PendulumParametricPolicy(
        _env_description=None,
        _observation_space=None,
        _action_space=action_space,
        _seed=None,
        init_params={"kp": 12.0, "kd": 3.0},
        param_bounds={"kp": (-50.0, 50.0), "kd": (0.0, 20.0)},
        min_torque=float(action_space.low[0]),
        max_torque=float(action_space.high[0]),
    )
    return expert


def build_residual_approach(seed: int, backend: str) -> ResidualApproach:
    """Construct ResidualApproach with pendulum expert and given backend."""

    # Template env for spaces
    env = make_pendulum_env(seed=seed)
    obs_space = env.observation_space
    act_space = env.action_space
    assert isinstance(act_space, Box), "Pendulum must have Box action space."

    expert = build_pendulum_expert(act_space)

    def env_factory(instance_num: int) -> gym.Env:
        # Vary seed by instance number for training envs
        return make_pendulum_env(seed=seed + instance_num)

    residual = ResidualApproach(
        environment_description="Gymnasium Pendulum-v1",
        observation_space=obs_space,
        action_space=act_space,
        seed=seed,
        expert=expert,
        env_factory=env_factory,
        backend=backend,
        total_timesteps=TRAIN_STEPS_PER_PHASE,  # per train() call
        lr=1e-3,
        noise_std=0.1,
        verbose=1,
        train_before_eval=False,  # we manually call train() each phase
        train_env_instance=0,
    )

    env.close()
    return residual


def evaluate_policy(
    residual: ResidualApproach,
    seed: int,
    num_episodes: int,
    max_steps: int,
    upright_threshold: float,
) -> List[Dict]:
    """Evaluate current residual policy on fresh Pendulum envs."""
    rng = np.random.default_rng(seed)
    episode_metrics: List[Dict] = []

    for episode_idx in range(num_episodes):
        env = make_pendulum_env(seed=int(rng.integers(0, 1_000_000)))
        obs, info = env.reset()
        residual.reset(np.asarray(obs, dtype=np.float32), info)

        total_reward = 0.0
        thetas: List[float] = []
        steps = 0

        for _ in range(max_steps):
            obs_array = np.asarray(obs, dtype=np.float32)

            # reconstruct theta from [cos, sin]
            if obs_array.shape[0] >= 2:
                x, y = float(obs_array[0]), float(obs_array[1])
                theta = float(np.arctan2(y, x))
                thetas.append(theta)

            action = residual.step()
            obs, rew, terminated, truncated, info = env.step(action)
            total_reward += float(rew)
            steps += 1

            done = bool(terminated or truncated)
            residual.update(obs, float(rew), done, info)
            if done:
                break

        env.close()

        if len(thetas) > 0:
            theta_arr = np.asarray(thetas, dtype=np.float32)
            upright_mask = np.abs(theta_arr) < upright_threshold
            upright_fraction = float(np.mean(upright_mask))
        else:
            upright_fraction = 0.0

        episode_metrics.append(
            {
                "episode_idx": episode_idx,
                "total_reward": total_reward,
                "total_steps": steps,
                "upright_fraction": upright_fraction,
            }
        )

    return episode_metrics


def run_phased_experiment(backend: str, seed: int) -> pd.DataFrame:
    """Run phased training + eval for a single backend and a single random
    seed.

    Returns:
        summary_df with columns:
          ['phase', 'timesteps',
           'mean_reward', 'std_reward',
           'mean_upright_fraction', 'std_upright_fraction',
           'mean_steps', 'std_steps',
           'seed', 'variant']
    """
    os.makedirs("logs", exist_ok=True)

    safe_name = backend.replace("sb3-", "")
    episode_csv = f"logs/pendulum_{safe_name}_seed{seed}_phased_episodes.csv"
    summary_csv = f"logs/pendulum_{safe_name}_seed{seed}_phased_summary.csv"
    reward_plot = f"logs/pendulum_{safe_name}_seed{seed}_reward_curve.png"
    upright_plot = f"logs/pendulum_{safe_name}_seed{seed}_upright_curve.png"

    residual = build_residual_approach(seed=seed, backend=backend)

    all_episode_rows: List[Dict] = []
    cumulative_timesteps = 0

    # Phase 0: no residual training
    print(
        f"\n=== [{backend}] Seed {seed} Phase 0: no residual training (timesteps = 0) ==="
    )
    phase0_eps = evaluate_policy(
        residual,
        seed=seed + 123,
        num_episodes=NUM_EVAL_EPISODES,
        max_steps=MAX_EVAL_STEPS,
        upright_threshold=UPRIGHT_THRESHOLD,
    )
    for row in phase0_eps:
        row["phase"] = 0
        row["timesteps"] = cumulative_timesteps
    all_episode_rows.extend(phase0_eps)

    # Subsequent phases: train then evaluate
    for phase in range(1, NUM_TRAIN_PHASES + 1):
        print(
            f"\n=== [{backend}] Seed {seed} Phase {phase}: "
            f"training for {TRAIN_STEPS_PER_PHASE} timesteps ==="
        )
        residual.train()
        cumulative_timesteps += TRAIN_STEPS_PER_PHASE

        phase_eps = evaluate_policy(
            residual,
            seed=seed + 123 + phase,
            num_episodes=NUM_EVAL_EPISODES,
            max_steps=MAX_EVAL_STEPS,
            upright_threshold=UPRIGHT_THRESHOLD,
        )
        for row in phase_eps:
            row["phase"] = phase
            row["timesteps"] = cumulative_timesteps
        all_episode_rows.extend(phase_eps)

    # Per-episode results (for this backend & seed)
    episodes_df = pd.DataFrame(all_episode_rows)
    episodes_df.to_csv(episode_csv, index=False)
    print(f"\n[{backend} | seed {seed}] Saved per-episode results to {episode_csv}")
    print(episodes_df.head())

    # Per-phase summary for this seed
    summary = (
        episodes_df.groupby(["phase", "timesteps"])
        .agg(
            mean_reward=("total_reward", "mean"),
            std_reward=("total_reward", "std"),
            mean_upright_fraction=("upright_fraction", "mean"),
            std_upright_fraction=("upright_fraction", "std"),
            mean_steps=("total_steps", "mean"),
            std_steps=("total_steps", "std"),
        )
        .reset_index()
    )
    summary["seed"] = seed
    summary["variant"] = safe_name
    summary.to_csv(summary_csv, index=False)
    print(f"\n[{backend} | seed {seed}] Saved per-phase summary to {summary_csv}")
    print(summary)

    # Optional per-seed plots
    x = summary["timesteps"].values

    plt.figure(figsize=(7, 5))
    y = summary["mean_reward"].values
    yerr = summary["std_reward"].values
    plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4)
    plt.xlabel("Cumulative training timesteps (residual)")
    plt.ylabel("Total reward (mean over episodes)")
    plt.title(f"Pendulum residual: {safe_name}, seed={seed}, reward vs training")
    plt.tight_layout()
    plt.savefig(reward_plot, dpi=200)
    plt.close()
    print(f"[{backend} | seed {seed}] Saved reward curve to {reward_plot}")

    plt.figure(figsize=(7, 5))
    y = summary["mean_upright_fraction"].values
    yerr = summary["std_upright_fraction"].values
    plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4)
    plt.xlabel("Cumulative training timesteps (residual)")
    plt.ylabel(f"Upright fraction (|theta| < {UPRIGHT_THRESHOLD:.2f})")
    plt.title(f"Pendulum residual: {safe_name}, seed={seed}, upright vs training")
    plt.tight_layout()
    plt.savefig(upright_plot, dpi=200)
    plt.close()
    print(f"[{backend} | seed {seed}] Saved upright curve to {upright_plot}")

    return summary


def make_multi_plots(all_summary: pd.DataFrame) -> None:
    """Combined plots over all algorithms.

    all_summary is expected to be aggregated across seeds, with columns:
      ['variant', 'phase', 'timesteps',
       'mean_reward', 'std_reward',
       'mean_upright_fraction', 'std_upright_fraction', ...]
    """

    # Reward multi-curve
    plt.figure(figsize=(7, 5))
    for variant, sub in all_summary.groupby("variant"):
        x = sub["timesteps"].values
        y = sub["mean_reward"].values
        yerr = sub["std_reward"].values
        plt.plot(x, y, marker="o", label=variant)
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.2)
    plt.xlabel("Cumulative training timesteps (residual)")
    plt.ylabel("Total reward (mean over seeds)")
    plt.title("Pendulum residual learning: different SB3 algorithms")
    plt.legend()
    plt.tight_layout()
    plt.savefig("logs/pendulum_algos_multi_reward_curve.png", dpi=200)
    plt.close()
    print("Saved combined reward plot to logs/pendulum_algos_multi_reward_curve.png")

    # Upright multi-curve
    if "mean_upright_fraction" in all_summary.columns:
        plt.figure(figsize=(7, 5))
        for variant, sub in all_summary.groupby("variant"):
            x = sub["timesteps"].values
            y = sub["mean_upright_fraction"].values
            yerr = sub["std_upright_fraction"].values
            plt.plot(x, y, marker="o", label=variant)
            plt.fill_between(x, y - yerr, y + yerr, alpha=0.2)
        plt.xlabel("Cumulative training timesteps (residual)")
        plt.ylabel("Upright fraction (mean over seeds)")
        plt.title("Pendulum residual stability vs training (algorithms)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("logs/pendulum_algos_multi_upright_curve.png", dpi=200)
        plt.close()
        print(
            "Saved combined upright plot to "
            "logs/pendulum_algos_multi_upright_curve.png"
        )


def main() -> None:
    per_seed_summaries: List[pd.DataFrame] = []

    # Run every backend for every seed
    for backend in BACKENDS:
        for seed in SEEDS:
            print(f"\n=== Running backend={backend}, seed={seed} ===")
            summary_df = run_phased_experiment(backend, seed)
            per_seed_summaries.append(summary_df)

    combined = pd.concat(per_seed_summaries, ignore_index=True)

    # Aggregate across seeds for each backend / phase / timestep
    aggregated = (
        combined.groupby(["variant", "phase", "timesteps"])
        .agg(
            mean_reward=("mean_reward", "mean"),
            std_reward=("mean_reward", "std"),
            mean_upright_fraction=("mean_upright_fraction", "mean"),
            std_upright_fraction=("mean_upright_fraction", "std"),
            mean_steps=("mean_steps", "mean"),
            std_steps=("mean_steps", "std"),
        )
        .reset_index()
    )

    make_multi_plots(aggregated)


if __name__ == "__main__":
    main()
