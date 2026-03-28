"""Run Motion2DBiRRTApproach on Motion2D variants across multiple seeds.

Evaluates 1-, 3-, 5-, and 7-passage environments over 10 seeds and writes
per-seed and aggregate result files to results/.  Pass --video to also save
one MP4 per (passages, seed) trial to results/videos/.

Usage::

    python experiments/motion2d_experiments/run_birrt_motion2d.py
    python experiments/motion2d_experiments/run_birrt_motion2d.py --video
    python experiments/motion2d_experiments/run_birrt_motion2d.py --max-steps 1500
"""

import argparse
import logging
from pathlib import Path

import kinder
import numpy as np
from gymnasium.envs.registration import register, registry

from programmatic_policy_learning.approaches.motion2d_birrt_approach import (
    Motion2DBiRRTApproach,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

SEEDS = list(range(5))
PASSAGES = [1, 2, 3, 4, 5, 7]
MAX_STEPS = 1000
NUM_ATTEMPTS = 20
NUM_ITERS = 500
SMOOTH_AMT = 50

RESULTS_DIR = Path("experiments/motion2d_experiments/results")
VIDEOS_DIR = RESULTS_DIR / "videos"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _env_id(passages: int) -> str:
    return f"kinder/Motion2D-p{passages}-v0"


def _register_envs() -> None:
    for p in PASSAGES:
        env_id = _env_id(p)
        if env_id not in registry:
            register(
                id=env_id,
                entry_point="kinder.envs.kinematic2d.motion2d:Motion2DEnv",
                kwargs={"num_passages": p},
            )


def _get_oc_state(inner_env, obs):
    inner_env.set_state(obs)
    return inner_env._object_centric_env.get_state()  # pylint: disable=protected-access


def _save_video(frames: list[np.ndarray], path: Path, fps: int = 20) -> None:
    from moviepy import ImageSequenceClip  # type: ignore[import-untyped]

    clean = [f[:, :, :3] if f.ndim == 3 and f.shape[2] == 4 else f for f in frames]
    clip = ImageSequenceClip(clean, fps=fps)
    clip.write_videofile(str(path), codec="libx264", logger=None)
    logging.info("Video saved → %s  (%d frames)", path, len(clean))


def run_trial(passages: int, seed: int, save_video: bool = False) -> dict:
    """Run one trial and return a result dict."""
    env_id = _env_id(passages)
    render_mode = "rgb_array" if save_video else None
    env = kinder.make(env_id, render_mode=render_mode, allow_state_access=True)
    inner = env.unwrapped
    obs, info = env.reset(seed=seed)

    approach = Motion2DBiRRTApproach(
        environment_description=f"Motion2D-p{passages}",
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=seed,
        get_object_centric_state=lambda o: _get_oc_state(inner, o),
        num_attempts=NUM_ATTEMPTS,
        num_iters=NUM_ITERS,
        smooth_amt=SMOOTH_AMT,
    )
    approach.reset(obs, info)

    plan_length = len(approach._plan)  # pylint: disable=protected-access
    metrics = approach.metrics

    goal_reached = False
    total_steps = 0
    total_reward = 0.0
    frames: list[np.ndarray] = []

    for step in range(MAX_STEPS):
        if save_video:
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))
        try:
            action = approach.step()
        except ValueError:
            # Plan exhausted with no goal — stop.
            break
        obs, reward, terminated, _, _ = env.step(action)
        approach.update(obs, float(reward), terminated, {})
        total_reward += float(reward)
        total_steps = step + 1
        if terminated:
            goal_reached = True
            if save_video:
                frame = env.render()
                if frame is not None:
                    frames.append(np.asarray(frame))
            break

    env.close()

    if save_video and frames:
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        _save_video(frames, VIDEOS_DIR / f"p{passages}_seed{seed}.mp4")

    result = {
        "passages": passages,
        "seed": seed,
        "goal_reached": goal_reached,
        "total_steps": total_steps,
        "total_reward": total_reward,
        "plan_length": plan_length,
        "num_collision_checks": metrics.num_collision_checks if metrics else 0,
        "num_nodes_extended": metrics.num_nodes_extended if metrics else 0,
    }

    status = "REACHED" if goal_reached else "FAILED"
    logging.info(
        "p%d seed=%d | %s | steps=%d plan=%d collisions=%d nodes=%d",
        passages,
        seed,
        status,
        total_steps,
        plan_length,
        result["num_collision_checks"],
        result["num_nodes_extended"],
    )
    return result


def write_seed_results(seed: int, results: list[dict]) -> None:
    path = RESULTS_DIR / f"seed_{seed}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Motion2D BiRRT — Seed {seed}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Seeds: {seed}\n")
        f.write(f"Passages: {PASSAGES}\n")
        f.write(f"Max steps: {MAX_STEPS}\n")
        f.write(
            f"BiRRT: num_attempts={NUM_ATTEMPTS}, "
            f"num_iters={NUM_ITERS}, smooth_amt={SMOOTH_AMT}\n\n"
        )
        f.write("RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Passages':<10} {'Reached':<9} {'Steps':<7} "
            f"{'Plan':<7} {'Collisions':<12} {'Nodes':<8}\n"
        )
        f.write("-" * 53 + "\n")
        for r in results:
            f.write(
                f"p{r['passages']:<9} {str(r['goal_reached']):<9} "
                f"{r['total_steps']:<7} {r['plan_length']:<7} "
                f"{r['num_collision_checks']:<12} {r['num_nodes_extended']:<8}\n"
            )
        successes = sum(1 for r in results if r["goal_reached"])
        f.write(f"\nSuccess: {successes}/{len(results)}\n")
    logging.info("Seed %d results written to: %s", seed, path)


def write_aggregate_results(all_results: dict[int, list[dict]]) -> None:
    flat = [r for rs in all_results.values() for r in rs]
    path = RESULTS_DIR / "aggregate_results.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Motion2D BiRRT — Aggregate Results ({len(SEEDS)} seeds)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write(f"Passages: {PASSAGES}\n")
        f.write(f"Max steps: {MAX_STEPS}\n")
        f.write(
            f"BiRRT: num_attempts={NUM_ATTEMPTS}, "
            f"num_iters={NUM_ITERS}, smooth_amt={SMOOTH_AMT}\n\n"
        )

        # Per-seed summary
        f.write("PER-SEED SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Seed':<6} {'Solved':<8} {'Success%':<10} {'Avg Steps':<12}\n")
        f.write("-" * 36 + "\n")
        for seed, results in sorted(all_results.items()):
            solved = sum(1 for r in results if r["goal_reached"])
            avg_steps = np.mean([r["total_steps"] for r in results])
            f.write(
                f"{seed:<6} {solved}/{len(results):<5} "
                f"{solved / len(results) * 100:<10.1f} {avg_steps:<12.1f}\n"
            )

        # Per-passages aggregate
        f.write("\nPER-PASSAGES AGGREGATE\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Passages':<10} {'Solved':<10} {'Success%':<10} "
            f"{'Avg Steps':<12} {'Avg Plan':<10} "
            f"{'Avg Collisions':<16} {'Avg Nodes':<10}\n"
        )
        f.write("-" * 68 + "\n")
        for p in PASSAGES:
            p_results = [r for r in flat if r["passages"] == p]
            solved = sum(1 for r in p_results if r["goal_reached"])
            avg_steps = np.mean([r["total_steps"] for r in p_results])
            avg_plan = np.mean([r["plan_length"] for r in p_results])
            avg_coll = np.mean([r["num_collision_checks"] for r in p_results])
            avg_nodes = np.mean([r["num_nodes_extended"] for r in p_results])
            f.write(
                f"p{p:<9} {solved}/{len(p_results):<7} "
                f"{solved / len(p_results) * 100:<10.1f} "
                f"{avg_steps:<12.1f} {avg_plan:<10.1f} "
                f"{avg_coll:<16.1f} {avg_nodes:<10.1f}\n"
            )

        total_solved = sum(1 for r in flat if r["goal_reached"])
        f.write(
            f"\nOverall: {total_solved}/{len(flat)} "
            f"({total_solved / len(flat) * 100:.1f}%)\n"
        )
    logging.info("Aggregate results written to: %s", path)


def main(args: argparse.Namespace) -> None:
    global MAX_STEPS  # pylint: disable=global-statement
    MAX_STEPS = args.max_steps

    _register_envs()

    all_results: dict[int, list[dict]] = {}

    for seed in SEEDS:
        logging.info("%s", "=" * 60)
        logging.info("SEED %d", seed)
        logging.info("%s", "=" * 60)
        seed_results = []
        for passages in PASSAGES:
            result = run_trial(passages, seed, save_video=args.video)
            seed_results.append(result)
        all_results[seed] = seed_results
        write_seed_results(seed, seed_results)

    write_aggregate_results(all_results)

    # Console summary
    flat = [r for rs in all_results.values() for r in rs]
    logging.info("%s", "\n" + "=" * 60)
    logging.info("SUMMARY")
    logging.info("%s", "=" * 60)
    logging.info(
        "%-10s %-10s %-12s %-14s %-10s",
        "Passages",
        "Solved",
        "Success%",
        "Avg Collisions",
        "Avg Nodes",
    )
    logging.info("-" * 56)
    for p in PASSAGES:
        p_results = [r for r in flat if r["passages"] == p]
        solved = sum(1 for r in p_results if r["goal_reached"])
        avg_coll = np.mean([r["num_collision_checks"] for r in p_results])
        avg_nodes = np.mean([r["num_nodes_extended"] for r in p_results])
        logging.info(
            "p%-9d %d/%-7d %-12.1f %-14.1f %-10.1f",
            p,
            solved,
            len(p_results),
            solved / len(p_results) * 100,
            avg_coll,
            avg_nodes,
        )
    total_solved = sum(1 for r in flat if r["goal_reached"])
    logging.info(
        "Overall: %d/%d (%.1f%%)", total_solved, len(flat), total_solved / len(flat) * 100
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Save an MP4 for each trial to results/videos/",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
