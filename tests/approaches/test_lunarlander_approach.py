#!/usr/bin/env python3
"""
Test/evaluate LunarLanderContinuous heuristic expert (NO training).

What it does:
- Loads your create_manual_lunarlander_continuous_policy(...)
- Optionally loads a params JSON (like your best_params_*.json)
- Runs N episodes across seeds and logs returns
- Saves:
    logs/expert_eval/lunarlander_expert_eval_<tag>.csv
    logs/expert_eval/lunarlander_expert_eval_<tag>_summary.json
- Optional: records a GIF of one episode to inspect behavior.

Usage examples:

  # Quick sanity check (no GIF)
  python experiments/test_lunarlander_expert.py --episodes 20 --seed 0 --tag sanity

  # Evaluate across multiple seeds (recommended)
  python experiments/test_lunarlander_expert.py --seeds 0 1 2 3 4 5 6 7 8 9 --episodes-per-seed 10 --tag eval10x10

  # Use your saved best params JSON from the sweep
  python experiments/test_lunarlander_expert.py --seeds 0 1 2 3 4 5 6 7 8 9 --episodes-per-seed 10 \
      --params-json logs/expert_sweep/best_params_lunarlander_manual_eval_best.json --tag best_eval

  # Record a GIF for seed-group 2, episode 0
  python experiments/test_lunarlander_expert.py --seeds 0 1 2 3 4 --episodes-per-seed 5 \
      --params-json logs/expert_sweep/best_params_lunarlander_manual_eval_best.json \
      --gif --gif-seed 2 --gif-episode 0 --tag best_gif

Notes:
- For GIF recording we use render_mode="rgb_array" (slower).
- Seeds: we treat each "seed group" s as producing episode seeds s*10000 + ep.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

# GIF dependency
import imageio.v2 as imageio

# Change this import to your expert module
from programmatic_policy_learning.approaches.experts.lundar_lander_experts import create_manual_lunarlander_continuous_policy  # noqa: F401


Obs = np.ndarray
Act = np.ndarray


@dataclass
class EpisodeResult:
    seed_group: int
    episode_idx: int
    episode_seed: int
    return_: float
    length: int
    terminated: bool
    truncated: bool


def load_params_json(path: Path) -> Dict[str, float]:
    obj = json.loads(path.read_text())
    # Accept either {"params": {...}} (your sweep format) or just {...}
    if isinstance(obj, dict) and "params" in obj and isinstance(obj["params"], dict):
        return {k: float(v) for k, v in obj["params"].items()}
    return {k: float(v) for k, v in obj.items()}


def make_env(*, record_rgb: bool) -> gym.Env:
    return gym.make("LunarLanderContinuous-v3", render_mode="rgb_array" if record_rgb else None)


def rollout_episode(
    env: gym.Env,
    policy: Callable[[Obs], Act],
    episode_seed: int,
    max_steps: int,
    *,
    record_gif: bool = False,
    gif_every: int = 2,
    gif_max_frames: int = 900,
) -> Tuple[float, int, bool, bool, Optional[List[np.ndarray]]]:
    obs, _ = env.reset(seed=episode_seed)
    total = 0.0
    t = 0
    terminated = False
    truncated = False

    frames: Optional[List[np.ndarray]] = [] if record_gif else None

    while True:
        if record_gif and frames is not None:
            if (t % gif_every == 0) and (len(frames) < gif_max_frames):
                frame = env.render()
                if isinstance(frame, np.ndarray):
                    frames.append(frame)

        act = policy(obs)
        act = np.asarray(act, dtype=np.float32).reshape(env.action_space.shape)

        obs, r, terminated, truncated, _ = env.step(act)
        total += float(r)
        t += 1

        if terminated or truncated or t >= max_steps:
            if (not terminated) and (not truncated) and (t >= max_steps):
                truncated = True
            if record_gif and frames is not None and len(frames) < gif_max_frames:
                frame = env.render()
                if isinstance(frame, np.ndarray):
                    frames.append(frame)
            break

    return total, t, terminated, truncated, frames


def save_gif(path: Path, frames: List[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames_u8 = [f.astype(np.uint8, copy=False) for f in frames]
    imageio.mimsave(path, frames_u8, fps=fps)


def save_csv(path: Path, results: List[EpisodeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed_group", "episode_idx", "episode_seed", "return", "length", "terminated", "truncated"])
        for r in results:
            w.writerow(
                [r.seed_group, r.episode_idx, r.episode_seed, f"{r.return_:.6f}", r.length, int(r.terminated), int(r.truncated)]
            )


def summarize(results: List[EpisodeResult]) -> Dict[str, float]:
    arr = np.asarray([r.return_ for r in results], dtype=np.float32)
    lens = np.asarray([r.length for r in results], dtype=np.float32)
    return {
        "n": float(arr.size),
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "mean_len": float(lens.mean()) if lens.size else 0.0,
        "pct_lt_-400": float(np.mean(arr < -400.0)) if arr.size else 0.0,
        "pct_lt_-600": float(np.mean(arr < -600.0)) if arr.size else 0.0,
        "pct_ge_-200": float(np.mean(arr >= -200.0)) if arr.size else 0.0,
        "pct_ge_0": float(np.mean(arr >= 0.0)) if arr.size else 0.0,
        "pct_ge_200": float(np.mean(arr >= 200.0)) if arr.size else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50, help="Total episodes (single seed-group mode).")
    ap.add_argument("--seed", type=int, default=0, help="Seed group (single seed-group mode).")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="Seed groups to evaluate.")
    ap.add_argument("--episodes-per-seed", type=int, default=10, help="Episodes per seed group (when --seeds is set).")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--outdir", type=str, default="logs/expert_eval")
    ap.add_argument("--tag", type=str, default="manual")

    ap.add_argument("--params-json", type=str, default="", help="Path to JSON containing params or sweep output JSON.")

    # GIF options
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--gif-seed", type=int, default=None, help="Which seed_group to record (default first).")
    ap.add_argument("--gif-episode", type=int, default=0, help="Which episode index within that seed group.")
    ap.add_argument("--gif-fps", type=int, default=30)
    ap.add_argument("--gif-every", type=int, default=2)
    ap.add_argument("--gif-max-frames", type=int, default=900)

    args = ap.parse_args()

    params: Dict[str, float] = {}
    if args.params_json:
        params = load_params_json(Path(args.params_json))

    # Determine if we need rgb_array mode
    record_rgb = bool(args.gif)
    env = make_env(record_rgb=record_rgb)

    policy = create_manual_lunarlander_continuous_policy(env.action_space, params=params)

    results: List[EpisodeResult] = []
    gif_frames: Optional[List[np.ndarray]] = None
    gif_seed_group: Optional[int] = None

    def want_gif(seed_group: int, ep_idx: int) -> bool:
        if not args.gif:
            return False
        target_seed = args.gif_seed if args.gif_seed is not None else (
            (args.seeds[0] if args.seeds else args.seed)
        )
        return (seed_group == target_seed) and (ep_idx == args.gif_episode)

    if args.seeds and len(args.seeds) > 0:
        for sg in args.seeds:
            for ep in range(args.episodes_per_seed):
                episode_seed = int(sg * 10_000 + ep)
                record = want_gif(sg, ep)

                ret, length, terminated, truncated, frames = rollout_episode(
                    env,
                    policy,
                    episode_seed=episode_seed,
                    max_steps=args.max_steps,
                    record_gif=record,
                    gif_every=args.gif_every,
                    gif_max_frames=args.gif_max_frames,
                )

                results.append(
                    EpisodeResult(
                        seed_group=int(sg),
                        episode_idx=int(ep),
                        episode_seed=int(episode_seed),
                        return_=float(ret),
                        length=int(length),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                    )
                )
                print(f"[seed_group={sg} ep={ep:03d}] return={ret:7.1f} len={length}")

                if record and frames is not None:
                    gif_frames = frames
                    gif_seed_group = sg
    else:
        sg = int(args.seed)
        for ep in range(args.episodes):
            episode_seed = int(sg * 10_000 + ep)
            record = want_gif(sg, ep)

            ret, length, terminated, truncated, frames = rollout_episode(
                env,
                policy,
                episode_seed=episode_seed,
                max_steps=args.max_steps,
                record_gif=record,
                gif_every=args.gif_every,
                gif_max_frames=args.gif_max_frames,
            )

            results.append(
                EpisodeResult(
                    seed_group=int(sg),
                    episode_idx=int(ep),
                    episode_seed=int(episode_seed),
                    return_=float(ret),
                    length=int(length),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
            )
            print(f"[seed_group={sg} ep={ep:03d}] return={ret:7.1f} len={length}")

            if record and frames is not None:
                gif_frames = frames
                gif_seed_group = sg

    env.close()

    stats = summarize(results)
    print("\n=== Summary ===")
    for k, v in stats.items():
        if k in {"n"}:
            print(f"{k:>12}: {int(v)}")
        elif k.startswith("pct_"):
            print(f"{k:>12}: {100.0 * v:6.2f}%")
        else:
            print(f"{k:>12}: {v:8.3f}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / f"lunarlander_expert_eval_{args.tag}.csv"
    save_csv(csv_path, results)

    summary_path = outdir / f"lunarlander_expert_eval_{args.tag}_summary.json"
    summary_payload = {
        "tag": args.tag,
        "params_json": args.params_json,
        "params": params,
        "stats": stats,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    print(f"\nSaved CSV:     {csv_path}")
    print(f"Saved summary: {summary_path}")

    if args.gif:
        if gif_frames is None:
            print("WARNING: --gif set but no frames recorded. Check --gif-seed/--gif-episode.")
        else:
            gif_path = outdir / f"lunarlander_expert_{args.tag}_seed{gif_seed_group}_ep{args.gif_episode}.gif"
            save_gif(gif_path, gif_frames, fps=args.gif_fps)
            print(f"Saved GIF:     {gif_path}")


if __name__ == "__main__":
    main()