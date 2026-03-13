# # experiments/run_lunarlander_td3.py
# """
# Phased residual learning experiment for LunarLanderContinuous-v3
# using TD3 only, across 5 random seeds.

# For each seed:
#   * Build ResidualApproach with a heuristic expert as base policy
#   * Phase 0: evaluate untrained residual
#   * Phases 1..N: train for TRAIN_STEPS_PER_PHASE, then evaluate
#   * Save per-episode and per-phase summaries
#   * Plot reward and contact fraction curves

# After all seeds:
#   * Aggregate mean ± std across seeds
#   * Save combined plots
# """

# from __future__ import annotations

# import os
# from typing import Dict, List

# import gymnasium as gym
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from gymnasium.spaces import Box

# from programmatic_policy_learning.approaches.residual_approach import ResidualApproach
# from programmatic_policy_learning.approaches.experts.lundar_lander_experts import (
#     create_manual_lunarlander_continuous_policy,
# )

# # -------------------------------------------------
# # CONFIG
# # -------------------------------------------------

# ENV_ID = "LunarLanderContinuous-v3"

# SEEDS = [0]

# BACKEND = "sb3-td3"

# NUM_TRAIN_PHASES = 10
# TRAIN_STEPS_PER_PHASE = 20_000  # 200k total per seed

# NUM_EVAL_EPISODES = 50
# MAX_EVAL_STEPS = 1000

# LOG_DIR = "logs"


# # -------------------------------------------------
# # ENV + EXPERT
# # -------------------------------------------------

# def make_env(seed: int | None = None) -> gym.Env:
#     env = gym.make(ENV_ID)
#     if seed is not None:
#         env.reset(seed=seed)
#     return env


# def build_expert(action_space: Box):
#     return create_manual_lunarlander_continuous_policy(action_space)


# def build_residual(seed: int) -> ResidualApproach:
#     env = make_env(seed)
#     obs_space = env.observation_space
#     act_space = env.action_space
#     assert isinstance(act_space, Box)

#     expert_fn = build_expert(act_space)

#     def env_factory(instance_num: int) -> gym.Env:
#         return make_env(seed + instance_num)

#     residual = ResidualApproach(
#         environment_description=f"Gymnasium {ENV_ID}",
#         observation_space=obs_space,
#         action_space=act_space,
#         seed=seed,
#         expert=expert_fn,
#         env_factory=env_factory,
#         backend=BACKEND,
#         total_timesteps=TRAIN_STEPS_PER_PHASE,
#         lr=1e-3,
#         noise_std=0.1,
#         verbose=1,
#         train_before_eval=False,
#         train_env_instance=0,
#     )

#     env.close()
#     return residual


# # -------------------------------------------------
# # EVALUATION
# # -------------------------------------------------

# def evaluate(residual: ResidualApproach, seed: int) -> List[Dict]:
#     rng = np.random.default_rng(seed)
#     results: List[Dict] = []

#     for ep in range(NUM_EVAL_EPISODES):
#         env = make_env(seed=int(rng.integers(0, 1_000_000)))
#         obs, info = env.reset()
#         residual.reset(np.asarray(obs, dtype=np.float32), info)

#         total_reward = 0.0
#         steps = 0
#         both_contact_steps = 0

#         for _ in range(MAX_EVAL_STEPS):
#             obs_arr = np.asarray(obs, dtype=np.float32)

#             # obs = [x, y, vx, vy, angle, ang_vel, leg_l, leg_r]
#             leg_l = float(obs_arr[6])
#             leg_r = float(obs_arr[7])
#             if leg_l > 0.5 and leg_r > 0.5:
#                 both_contact_steps += 1

#             action = residual.step()
#             obs, rew, terminated, truncated, info = env.step(action)

#             total_reward += float(rew)
#             steps += 1

#             done = terminated or truncated
#             residual.update(obs, float(rew), done, info)
#             if done:
#                 break

#         env.close()

#         contact_fraction = both_contact_steps / max(steps, 1)

#         results.append(
#             {
#                 "episode_idx": ep,
#                 "total_reward": total_reward,
#                 "total_steps": steps,
#                 "contact_fraction": contact_fraction,
#             }
#         )

#     return results


# # -------------------------------------------------
# # PHASED TRAINING
# # -------------------------------------------------

# def run_seed(seed: int) -> pd.DataFrame:
#     os.makedirs(LOG_DIR, exist_ok=True)

#     residual = build_residual(seed)

#     all_rows: List[Dict] = []
#     cumulative_timesteps = 0

#     # Phase 0
#     print(f"\n=== Seed {seed} Phase 0 (no training) ===")
#     phase_eps = evaluate(residual, seed + 123)
#     for row in phase_eps:
#         row["phase"] = 0
#         row["timesteps"] = cumulative_timesteps
#     all_rows.extend(phase_eps)

#     # Train phases
#     for phase in range(1, NUM_TRAIN_PHASES + 1):
#         print(f"\n=== Seed {seed} Phase {phase} training ===")
#         residual.train()
#         cumulative_timesteps += TRAIN_STEPS_PER_PHASE

#         phase_eps = evaluate(residual, seed + 123 + phase)
#         for row in phase_eps:
#             row["phase"] = phase
#             row["timesteps"] = cumulative_timesteps
#         all_rows.extend(phase_eps)

#     df = pd.DataFrame(all_rows)

#     csv_path = f"{LOG_DIR}/lander_td3_seed{seed}_episodes.csv"
#     df.to_csv(csv_path, index=False)
#     print(f"Saved {csv_path}")

#     summary = (
#         df.groupby(["phase", "timesteps"])
#         .agg(
#             mean_reward=("total_reward", "mean"),
#             std_reward=("total_reward", "std"),
#             mean_contact=("contact_fraction", "mean"),
#             std_contact=("contact_fraction", "std"),
#             mean_steps=("total_steps", "mean"),
#             std_steps=("total_steps", "std"),
#         )
#         .reset_index()
#     )

#     summary["seed"] = seed
#     summary_csv = f"{LOG_DIR}/lander_td3_seed{seed}_summary.csv"
#     summary.to_csv(summary_csv, index=False)
#     print(f"Saved {summary_csv}")

#     return summary


# # -------------------------------------------------
# # MULTI-SEED AGGREGATION
# # -------------------------------------------------

# def make_plots(all_summary: pd.DataFrame):
#     # Reward
#     plt.figure(figsize=(7, 5))
#     x = all_summary["timesteps"].unique()

#     grouped = all_summary.groupby("timesteps")

#     mean_reward = grouped["mean_reward"].mean()
#     std_reward = grouped["mean_reward"].std()

#     plt.plot(x, mean_reward, marker="o")
#     plt.fill_between(x, mean_reward - std_reward, mean_reward + std_reward, alpha=0.2)

#     plt.xlabel("Training Timesteps")
#     plt.ylabel("Reward (mean ± std over seeds)")
#     plt.title("LunarLanderContinuous TD3 Residual Learning")
#     plt.tight_layout()
#     plt.savefig(f"{LOG_DIR}/lander_td3_multi_reward.png", dpi=200)
#     plt.close()

#     # Contact fraction
#     plt.figure(figsize=(7, 5))
#     mean_contact = grouped["mean_contact"].mean()
#     std_contact = grouped["mean_contact"].std()

#     plt.plot(x, mean_contact, marker="o")
#     plt.fill_between(x, mean_contact - std_contact, mean_contact + std_contact, alpha=0.2)

#     plt.xlabel("Training Timesteps")
#     plt.ylabel("Contact Fraction (mean ± std)")
#     plt.title("LunarLanderContinuous TD3 Stability")
#     plt.tight_layout()
#     plt.savefig(f"{LOG_DIR}/lander_td3_multi_contact.png", dpi=200)
#     plt.close()

#     print("Saved multi-seed plots.")


# # -------------------------------------------------
# # MAIN
# # -------------------------------------------------

# def main():
#     summaries = []
#     for seed in SEEDS:
#         summaries.append(run_seed(seed))

#     combined = pd.concat(summaries, ignore_index=True)
#     make_plots(combined)


# if __name__ == "__main__":
#     main()

# experiments/run_lunarlander_one_seed_gif.py
"""
One-seed LunarLanderContinuous TD3 residual experiment:
- Evaluate before training
- Save BEFORE gif
- Train for N timesteps (single block, not phased)
- Evaluate after training
- Save AFTER gif

Outputs (inside --out-dir):
- before_eval.csv
- after_eval.csv
- before.gif
- after.gif
- summary.txt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from gymnasium.spaces import Box

from programmatic_policy_learning.approaches.residual_approach import ResidualApproach
from programmatic_policy_learning.approaches.experts.lundar_lander_experts import (
    create_manual_lunarlander_continuous_policy,
)

ENV_ID = "LunarLanderContinuous-v3"


# -----------------------------
# Env helpers
# -----------------------------
def make_env(seed: int | None = None, *, render_mode: str | None = None) -> gym.Env:
    env = gym.make(ENV_ID, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env


def build_residual(seed: int, train_timesteps: int) -> ResidualApproach:
    # Create a temporary env to get spaces
    env = make_env(seed=seed, render_mode=None)
    obs_space = env.observation_space
    act_space = env.action_space
    assert isinstance(act_space, Box)

    expert_fn = create_manual_lunarlander_continuous_policy(act_space)

    # For training inside ResidualApproach
    def env_factory(instance_num: int) -> gym.Env:
        # Different seeds per instance if your approach uses it internally
        e = make_env(seed=seed + instance_num, render_mode=None)
        return e

    residual = ResidualApproach(
        environment_description=f"Gymnasium {ENV_ID}",
        observation_space=obs_space,
        action_space=act_space,
        seed=seed,
        expert=expert_fn,
        env_factory=env_factory,
        backend="sb3-td3",
        total_timesteps=train_timesteps,   # <-- train length
        lr=1e-3,
        noise_std=0.1,
        verbose=1,
        train_before_eval=False,
        train_env_instance=0,
    )

    env.close()
    return residual


# -----------------------------
# Eval + GIF
# -----------------------------
def eval_once(
    residual: ResidualApproach,
    *,
    seed: int,
    num_episodes: int,
    max_steps: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []

    for ep in range(num_episodes):
        env = make_env(seed=int(rng.integers(0, 1_000_000)), render_mode=None)
        obs, info = env.reset()
        residual.reset(np.asarray(obs, dtype=np.float32), info)

        total_reward = 0.0
        steps = 0
        both_contact_steps = 0

        for _ in range(max_steps):
            obs_arr = np.asarray(obs, dtype=np.float32)
            leg_l = float(obs_arr[6])
            leg_r = float(obs_arr[7])
            if leg_l > 0.5 and leg_r > 0.5:
                both_contact_steps += 1

            action = residual.step()
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += float(rew)
            steps += 1

            residual.update(obs, float(rew), done, info)
            if done:
                break

        env.close()

        rows.append(
            dict(
                episode_idx=ep,
                total_reward=total_reward,
                total_steps=steps,
                contact_fraction=(both_contact_steps / max(steps, 1)),
            )
        )

    return pd.DataFrame(rows)


def rollout_gif(
    residual: ResidualApproach,
    *,
    seed: int,
    max_steps: int,
    fps: int,
    out_path: Path,
) -> Tuple[float, int]:
    env = make_env(seed=seed, render_mode="rgb_array")
    obs, info = env.reset()
    residual.reset(np.asarray(obs, dtype=np.float32), info)

    frames = []
    total_reward = 0.0
    steps = 0

    for _ in range(max_steps):
        frame = env.render()
        frames.append(frame)

        action = residual.step()
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += float(rew)
        steps += 1

        residual.update(obs, float(rew), done, info)
        if done:
            break

    env.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    return total_reward, steps


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-timesteps", type=int, default=700_000)
    ap.add_argument("--num-eval-episodes", type=int, default=20)
    ap.add_argument("--max-eval-steps", type=int, default=1000)
    ap.add_argument("--gif-seed", type=int, default=12345, help="seed used for the GIF episode")
    ap.add_argument("--gif-fps", type=int, default=30)
    ap.add_argument("--out-dir", type=str, default="logs/lander_one_seed")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    residual = build_residual(seed=args.seed, train_timesteps=args.train_timesteps)

    # ---- BEFORE ----
    print("\n=== BEFORE TRAINING: eval ===", flush=True)
    before_df = eval_once(
        residual,
        seed=args.seed + 111,
        num_episodes=args.num_eval_episodes,
        max_steps=args.max_eval_steps,
    )
    before_df.to_csv(out_dir / "before_eval.csv", index=False)

    print("=== BEFORE TRAINING: gif ===", flush=True)
    before_gif_reward, before_gif_steps = rollout_gif(
        residual,
        seed=args.gif_seed,
        max_steps=args.max_eval_steps,
        fps=args.gif_fps,
        out_path=(out_dir / "before.gif"),
    )

    # ---- TRAIN ----
    print(f"\n=== TRAINING: {args.train_timesteps} timesteps ===", flush=True)
    residual.train()

    # ---- AFTER ----
    print("\n=== AFTER TRAINING: eval ===", flush=True)
    after_df = eval_once(
        residual,
        seed=args.seed + 222,
        num_episodes=args.num_eval_episodes,
        max_steps=args.max_eval_steps,
    )
    after_df.to_csv(out_dir / "after_eval.csv", index=False)

    print("=== AFTER TRAINING: gif ===", flush=True)
    after_gif_reward, after_gif_steps = rollout_gif(
        residual,
        seed=args.gif_seed,
        max_steps=args.max_eval_steps,
        fps=args.gif_fps,
        out_path=(out_dir / "after.gif"),
    )

    # Summary
    def _summ(df: pd.DataFrame) -> str:
        return (
            f"mean_reward={df['total_reward'].mean():.2f} ± {df['total_reward'].std():.2f}, "
            f"mean_contact={df['contact_fraction'].mean():.3f} ± {df['contact_fraction'].std():.3f}"
        )

    summary_lines = [
        f"ENV={ENV_ID}",
        f"seed={args.seed}",
        f"train_timesteps={args.train_timesteps}",
        "",
        "BEFORE:",
        _summ(before_df),
        f"gif_episode_reward={before_gif_reward:.2f}, gif_steps={before_gif_steps}",
        "",
        "AFTER:",
        _summ(after_df),
        f"gif_episode_reward={after_gif_reward:.2f}, gif_steps={after_gif_steps}",
        "",
        f"wrote: {out_dir}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines))
    print("\n".join(summary_lines), flush=True)


if __name__ == "__main__":
    main()