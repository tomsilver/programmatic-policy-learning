"""Run LLMPPLApproach on Motion2D environment.

Synthesizes 5 policies (one per approach seed) using p=2 as the example
environment, then evaluates each policy on 10 fixed instances:
  2 instances (seed=0, seed=1) x 6 passage counts (p=0,1,3,4,5,7).

Only 5 LLM calls are made (one per approach seed). All evaluation runs reuse
the synthesized policy without re-querying the LLM.

Usage::

    python experiments/motion2d_experiments/run_llm_ppl_motion2d.py
    python experiments/motion2d_experiments/run_llm_ppl_motion2d.py --video
    python experiments/motion2d_experiments/run_llm_ppl_motion2d.py --max-steps 1000
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import kinder
import numpy as np
from gymnasium.envs.registration import register, registry
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel

from programmatic_policy_learning.approaches.llm_ppl_approach import (
    LLMPPLApproach,
    synthesize_policy_from_environment_description,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# --- Experiment configuration ------------------------------------------------
MODEL = "gpt-5.2"
APPROACH_SEEDS = list(range(5))  # 5 policy synthesis runs
EVAL_PASSAGES = [0, 1, 3, 5, 7]  # passage counts to evaluate on (1 instance each)
EVAL_SEEDS = [
    42,
    43,
    44,
    45,
    46,
]  # one eval seed per passage count (fixed across all approach seeds)
SYNTHESIS_PASSAGES = 2  # passage count used for example obs
SYNTHESIS_SEED = 125  # fixed seed for synthesis env (not in EVAL_SEEDS)
MAX_STEPS = 500

RESULTS_DIR = Path("experiments/motion2d_experiments/results/llm_ppl")
VIDEOS_DIR = RESULTS_DIR / "videos"

# --- Environment description -------------------------------------------------
MOTION2D_ENVIRONMENT_DESCRIPTION = """
    A 2D continuous motion planning environment (Motion2D from the KinDER
    benchmark).

    WORLD:
        - A 2.5 x 2.5 continuous world.
        - A circular robot starts on the left side, and a rectangular target region is on the right side.
        - Between the robot and target are vertical obstacle columns that span the environment from left to right.

    IMPORTANT:
        Each obstacle column is composed of TWO axis-aligned rectangles:
            (1) a bottom rectangle extending upward from the bottom of the world
            (2) a top rectangle extending downward from the top of the world
        These two rectangles share the same x-position and width, and together form a vertical wall
        with a single open gap between them.

        The only traversable region through each wall column is this vertical gap.

        - The number of such columns depends on the environment variant.
        - The x-positions of the columns increase from left to right between the robot and the target.

    OBSERVATION:
        - The observation is a flat numpy array (float32).

        - Robot:
            - obs[0], obs[1]: robot x, y position
            - obs[2]: robot theta (orientation in radians)
            - obs[3]: robot base radius (read from obs[3]; typically ~0.1)
            - The robot is a circle. The full circular body must fit through
              any passage — i.e., the robot center must be at least obs[3]
              away from every obstacle edge.

        - Target region:
            - obs[9], obs[10]: bottom-left corner (x, y)
            - obs[17], obs[18]: width and height

        - Obstacles:
            - Starting from obs[19], obstacles are listed sequentially
            - Each obstacle has 10 values:
                (x, y, theta, static, color_r, color_g, color_b, z_order, width, height)

            - Obstacles corresponding to a wall column appear in pairs:
                - the first rectangle is the bottom segment
                - the second rectangle is the top segment

            - For passage i:
                - bottom obstacle starts at index 19 + 20*i
                - top obstacle starts at index 19 + 20*i + 10

            - For a given column:
                - both rectangles share the same x-position and width
                - the wall x-position is obs[19 + 20*i] (same for both segments)
                - the bottom rectangle spans from its y up to y + height
                - the top rectangle starts at its y and extends upward

            - The open passage (gap) lies between:
                    (bottom_y + bottom_height) and (top_y)
            - Gap heights are on the order of 0.3–0.4 units. Given the robot
              diameter (~0.2), the clearance on each side is only ~0.05
              units. Precise y-alignment before entering the gap is necessary.

    ACTIONS:
        - A 5-dimensional continuous action:
        - action[0]: dx in [-0.05, 0.05]
        - action[1]: dy in [-0.05, 0.05]
        - action[2]: dtheta in [-pi/16, pi/16]
        - action[3]: darm in [-0.1, 0.1] (not needed, set to 0)
        - action[4]: vacuum in [0, 1] (not needed, set to 0)
        - The action array must be dtype float32.

    TASK:
        - Move the robot to the target region while avoiding obstacles.
        - Success occurs when the robot center lies inside the target region.

    REWARD:
        -1.0 per step until success.
"""


def _env_id(passages: int) -> str:
    return f"kinder/Motion2D-p{passages}-v0"


def _register_env(passages: int) -> None:
    env_id = _env_id(passages)
    if env_id not in registry:
        register(
            id=env_id,
            entry_point="kinder.envs.kinematic2d.motion2d:Motion2DEnv",
            kwargs={"num_passages": passages},
        )


def _save_video(frames: list[np.ndarray], path: Path, fps: int = 20) -> None:
    """Save frames as an MP4 video file."""
    # pylint: disable=import-outside-toplevel
    from moviepy import ImageSequenceClip  # type: ignore[import-untyped]

    clean = [f[:, :, :3] if f.ndim == 3 and f.shape[2] == 4 else f for f in frames]
    clip = ImageSequenceClip(clean, fps=fps)
    clip.write_videofile(str(path), codec="libx264", logger=None)
    logging.info("Video saved → %s  (%d frames)", path, len(clean))


def synthesize_policy(approach_seed: int, llm: OpenAIModel) -> tuple[Any, str]:
    """Synthesize one policy using a p=2 environment as the example.

    Returns the callable policy and its source code string.
    """
    _register_env(SYNTHESIS_PASSAGES)
    env = kinder.make(_env_id(SYNTHESIS_PASSAGES), allow_state_access=True)
    obs, _ = env.reset(seed=SYNTHESIS_SEED)
    env.close()  # type: ignore[no-untyped-call]

    policy = synthesize_policy_from_environment_description(
        MOTION2D_ENVIRONMENT_DESCRIPTION,
        llm,
        obs,
        env.action_space,
        seed=approach_seed,
    )
    code_str = policy.code_str  # type: ignore[attr-defined]
    logging.info("Approach seed %d — synthesized policy:\n%s", approach_seed, policy)
    return policy, code_str


def run_eval_trial(
    policy: Any,
    passages: int,
    eval_seed: int,
    approach_seed: int,
    save_video: bool = False,
) -> dict:
    """Evaluate a pre-synthesized policy on one environment instance."""
    _register_env(passages)
    render_mode = "rgb_array" if save_video else None
    env = kinder.make(
        _env_id(passages), render_mode=render_mode, allow_state_access=True
    )
    obs, info = env.reset(seed=eval_seed)

    # Inject the pre-synthesized policy before reset() so synthesis is skipped.
    approach = LLMPPLApproach(
        environment_description=MOTION2D_ENVIRONMENT_DESCRIPTION,
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=approach_seed,
        llm=None,  # type: ignore[arg-type]  # never called; policy injected below
    )
    approach._policy = policy  # pylint: disable=protected-access
    approach.reset(obs, info)

    goal_reached = False
    total_steps = 0
    total_reward = 0.0
    frames: list[np.ndarray] = []

    for step in range(MAX_STEPS):
        if save_video:
            frame: np.ndarray | list[np.ndarray] | None = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))
        action = approach.step()
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

    env.close()  # type: ignore[no-untyped-call]

    if save_video and frames:
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        video_path = VIDEOS_DIR / f"policy{approach_seed}_p{passages}_s{eval_seed}.mp4"
        _save_video(frames, video_path)

    status = "REACHED" if goal_reached else "FAILED"
    logging.info(
        "  policy_seed=%d p%d eval_seed=%d | %s | steps=%d reward=%.1f",
        approach_seed,
        passages,
        eval_seed,
        status,
        total_steps,
        total_reward,
    )

    return {
        "approach_seed": approach_seed,
        "passages": passages,
        "eval_seed": eval_seed,
        "goal_reached": goal_reached,
        "total_steps": total_steps,
        "total_reward": total_reward,
    }


def write_seed_results(
    approach_seed: int, results: list[dict], policy_code: str
) -> None:
    """Write per-policy-seed results to a text file."""
    path = RESULTS_DIR / f"policy_seed_{approach_seed}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"LLM PPL Motion2D — Policy Seed {approach_seed}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {MODEL}\n")
        f.write(f"Synthesis: p={SYNTHESIS_PASSAGES}, seed={SYNTHESIS_SEED}\n")
        f.write(f"Eval passages: {EVAL_PASSAGES}\n")
        f.write(f"Eval seeds: {EVAL_SEEDS}\n")
        f.write(f"Max steps: {MAX_STEPS}\n\n")
        f.write("GENERATED POLICY\n")
        f.write("-" * 50 + "\n")
        f.write(policy_code)
        f.write("\n" + "-" * 50 + "\n\n")
        f.write("RESULTS\n")
        f.write("-" * 50 + "\n")
        header = (
            f"{'Passages':<10} {'Eval Seed':<12} "
            f"{'Reached':<10} {'Steps':<8} {'Reward':<8}\n"
        )
        f.write(header)
        f.write("-" * 50 + "\n")
        for r in results:
            f.write(
                f"p{r['passages']:<9} {r['eval_seed']:<12} "
                f"{str(r['goal_reached']):<10} "
                f"{r['total_steps']:<8} {r['total_reward']:<8.1f}\n"
            )
        successes = sum(1 for r in results if r["goal_reached"])
        f.write(f"\nSuccess: {successes}/{len(results)}\n")
    logging.info("Policy seed %d results written to: %s", approach_seed, path)


def write_aggregate_results(all_results: dict[int, list[dict]]) -> None:
    """Write aggregate results across all policy seeds to a text file."""
    flat = [r for rs in all_results.values() for r in rs]
    path = RESULTS_DIR / "aggregate_results.txt"
    with open(path, "w", encoding="utf-8") as f:
        n_seeds = len(APPROACH_SEEDS)
        f.write(f"LLM PPL Motion2D — Aggregate Results ({n_seeds} policy seeds)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {MODEL}\n")
        f.write(f"Synthesis: p={SYNTHESIS_PASSAGES}, seed={SYNTHESIS_SEED}\n")
        f.write(f"Approach seeds: {APPROACH_SEEDS}\n")
        f.write(f"Eval passages: {EVAL_PASSAGES}\n")
        f.write(f"Eval seeds: {EVAL_SEEDS}\n")
        f.write(f"Max steps: {MAX_STEPS}\n\n")

        # Per-policy-seed summary
        f.write("PER-POLICY-SEED SUMMARY\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Seed':<6} {'Solved':<10} {'Success%':<12} {'Avg Steps':<12}\n")
        f.write("-" * 40 + "\n")
        for seed, results in sorted(all_results.items()):
            solved = sum(1 for r in results if r["goal_reached"])
            avg_steps = np.mean([r["total_steps"] for r in results])
            f.write(
                f"{seed:<6} {solved}/{len(results):<7} "
                f"{solved / len(results) * 100:<12.1f} {avg_steps:<12.1f}\n"
            )

        # Per-passages aggregate
        f.write("\nPER-PASSAGES AGGREGATE\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Passages':<10} {'Solved':<12} {'Success%':<12} {'Avg Steps':<12}\n")
        f.write("-" * 46 + "\n")
        for p in EVAL_PASSAGES:
            p_results = [r for r in flat if r["passages"] == p]
            solved = sum(1 for r in p_results if r["goal_reached"])
            avg_steps = np.mean([r["total_steps"] for r in p_results])
            f.write(
                f"p{p:<9} {solved}/{len(p_results):<9} "
                f"{solved / len(p_results) * 100:<12.1f} {avg_steps:<12.1f}\n"
            )

        total_solved = sum(1 for r in flat if r["goal_reached"])
        f.write(
            f"\nOverall: {total_solved}/{len(flat)} "
            f"({total_solved / len(flat) * 100:.1f}%)\n"
        )
    logging.info("Aggregate results written to: %s", path)


def main(args: argparse.Namespace) -> None:
    """Synthesize policies and evaluate them across passages and seeds."""
    global MAX_STEPS  # pylint: disable=global-statement
    MAX_STEPS = args.max_steps

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cache = SQLite3PretrainedLargeModelCache(Path("llm_cache.db"))
    llm = OpenAIModel(MODEL, cache)

    all_results: dict[int, list[dict]] = {}

    for approach_seed in APPROACH_SEEDS:
        logging.info("%s", "=" * 60)
        logging.info("POLICY SEED %d — synthesizing policy", approach_seed)
        logging.info("%s", "=" * 60)

        policy, policy_code = synthesize_policy(approach_seed, llm)

        seed_results = []
        for passages, eval_seed in zip(EVAL_PASSAGES, EVAL_SEEDS):
            result = run_eval_trial(
                policy=policy,
                passages=passages,
                eval_seed=eval_seed,
                approach_seed=approach_seed,
                save_video=args.video,
            )
            seed_results.append(result)

        all_results[approach_seed] = seed_results
        write_seed_results(approach_seed, seed_results, policy_code)

    write_aggregate_results(all_results)

    # Console summary
    flat = [r for rs in all_results.values() for r in rs]
    logging.info("%s", "\n" + "=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    logging.info("%-10s %-12s %-12s", "Passages", "Solved", "Success%")
    logging.info("-" * 34)
    for p in EVAL_PASSAGES:
        p_results = [r for r in flat if r["passages"] == p]
        solved = sum(1 for r in p_results if r["goal_reached"])
        logging.info(
            "p%-9d %d/%-9d %.1f%%",
            p,
            solved,
            len(p_results),
            solved / len(p_results) * 100,
        )
    total_solved = sum(1 for r in flat if r["goal_reached"])
    logging.info(
        "Overall: %d/%d (%.1f%%)",
        total_solved,
        len(flat),
        total_solved / len(flat) * 100,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Run LLMPPLApproach on Motion2D")
    p.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help=f"Max steps per eval trial (default: {MAX_STEPS})",
    )
    p.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Save an MP4 for each eval trial to results_llm_ppl/videos/",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
