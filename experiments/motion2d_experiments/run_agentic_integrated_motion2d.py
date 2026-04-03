"""Run AgenticIntegratedApproach on Motion2D with BiRRT.

Synthesizes candidate policies (one set per approach seed) on a p=2
training instance, scores them via full gym episodes, then evaluates
the best policy on 5 passage counts (p=0, 1, 3, 5, 7).

Usage::

    PYTHONHASHSEED=0 python experiments/motion2d_experiments/\
run_agentic_integrated_motion2d.py
    PYTHONHASHSEED=0 python experiments/motion2d_experiments/\
run_agentic_integrated_motion2d.py --video
"""

import argparse
import json
import logging
import os
import tempfile
from functools import partial
from pathlib import Path
from typing import Any

import kinder
import numpy as np
from gymnasium.envs.registration import register, registry
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel

from programmatic_policy_learning.approaches.agentic_integrated_approach import (
    BIRRT_INIT_DOC,
    BIRRT_PLANNER_DOC,
    AgenticIntegratedApproach,
    score_policy_motion2d,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# --- Experiment configuration ------------------------------------------------
MODEL = "gpt-5.2"
APPROACH_SEEDS = list(range(5))
EVAL_PASSAGES = [0, 1, 3, 5, 7]
EVAL_SEEDS = [42, 43, 44, 45, 46]
TRAIN_PASSAGES = 3
TRAIN_SEED = 125
MAX_STEPS = 1000
NUM_CANDIDATES = 5

RESULTS_DIR = Path("experiments/motion2d_experiments/results/agentic")
VIDEOS_DIR = RESULTS_DIR / "videos"

# --- Environment description (same as LLM PPL motion2d) ---------------------
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


# --- Env helpers -------------------------------------------------------------


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


def _get_oc_state(inner_env: Any, obs: Any) -> Any:
    """Return ObjectCentricState for obs by setting state on the inner env."""
    inner_env.set_state(obs)
    return inner_env._object_centric_env.get_state()  # pylint: disable=protected-access


def _get_robot(inner_env: Any, obs: Any) -> Any:
    """Return the robot Object from an observation."""
    # pylint: disable=import-outside-toplevel
    from kinder.envs.kinematic2d.object_types import CRVRobotType

    state = _get_oc_state(inner_env, obs)
    robots = state.get_objects(CRVRobotType)
    if not robots:
        raise ValueError("No robot found in ObjectCentricState.")
    return robots[0]


def _save_video(frames: list[np.ndarray], path: Path, fps: int = 20) -> None:
    """Save frames as an MP4 video file."""
    # pylint: disable=import-outside-toplevel
    from moviepy import ImageSequenceClip  # type: ignore[import-untyped]

    clean = [f[:, :, :3] if f.ndim == 3 and f.shape[2] == 4 else f for f in frames]
    clip = ImageSequenceClip(clean, fps=fps)
    clip.write_videofile(str(path), codec="libx264", logger=None)
    logging.info("Video saved → %s  (%d frames)", path, len(clean))


# --- Scoring -----------------------------------------------------------------


def _make_score_fn(passages: int, env_seed: int) -> partial:
    """Return a score_fn bound to a specific env configuration."""

    def _env_factory(_passages: int = passages, _seed: int = env_seed) -> Any:
        _register_env(_passages)
        env = kinder.make(_env_id(_passages), allow_state_access=True)
        env.reset(seed=_seed)
        return env

    return partial(
        score_policy_motion2d,
        env_factory=_env_factory,
    )


# --- Evaluation --------------------------------------------------------------


EVAL_TIMEOUT = 300.0  # seconds per eval trial


def run_eval_trial(
    approach: AgenticIntegratedApproach,
    passages: int,
    eval_seed: int,
    approach_seed: int,
    max_steps: int,
    save_video: bool = False,
) -> dict:
    """Evaluate the best policy on one environment instance."""
    import time  # pylint: disable=import-outside-toplevel

    _register_env(passages)
    render_mode = "rgb_array" if save_video else None
    env = kinder.make(
        _env_id(passages), render_mode=render_mode, allow_state_access=True
    )
    inner = env.unwrapped
    obs, info = env.reset(seed=eval_seed)

    # Build planner context for this eval env
    robot = _get_robot(inner, obs)
    planner_context = {
        "get_object_centric_state": lambda o, _inner=inner: _get_oc_state(_inner, o),
        "robot": robot,
        "action_space": env.action_space,
    }
    approach.update_planner_context(planner_context)

    # Clear metrics before eval
    metrics_path_str = os.environ.get("birrt_metrics_path")
    if metrics_path_str:
        Path(metrics_path_str).write_text("", encoding="utf-8")

    approach.reset(obs, info)

    goal_reached = False
    total_steps = 0
    total_reward = 0.0
    frames: list[np.ndarray] = []
    t0 = time.monotonic()

    for step in range(max_steps):
        if time.monotonic() - t0 > EVAL_TIMEOUT:
            logging.warning(
                "  seed=%d p%d eval=%d | TIMED OUT after %.0fs",
                approach_seed, passages, eval_seed, EVAL_TIMEOUT,
            )
            break
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

    # Read BiRRT metrics
    num_collision_checks = 0
    num_nodes_extended = 0
    if metrics_path_str:
        metrics_path = Path(metrics_path_str)
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        num_collision_checks += entry["num_collision_checks"]
                        num_nodes_extended += entry["num_nodes_extended"]

    env.close()  # type: ignore[no-untyped-call]

    if save_video and frames:
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        video_path = VIDEOS_DIR / f"seed{approach_seed}_p{passages}_eval{eval_seed}.mp4"
        _save_video(frames, video_path)

    status = "REACHED" if goal_reached else "FAILED"
    logging.info(
        "  seed=%d p%d eval=%d | %s | steps=%d checks=%d nodes=%d",
        approach_seed,
        passages,
        eval_seed,
        status,
        total_steps,
        num_collision_checks,
        num_nodes_extended,
    )

    return {
        "approach_seed": approach_seed,
        "passages": passages,
        "eval_seed": eval_seed,
        "goal_reached": goal_reached,
        "total_steps": total_steps,
        "total_reward": total_reward,
        "num_collision_checks": num_collision_checks,
        "num_nodes_extended": num_nodes_extended,
    }


# --- Result writing ----------------------------------------------------------


def write_seed_results(
    approach_seed: int,
    results: list[dict],
    best_code: str,
    all_codes: list[str],
    all_scores: list,
) -> None:
    """Write per-seed results and all candidate policies to a text file."""
    path = RESULTS_DIR / f"seed_{approach_seed}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Agentic Integrated Motion2D — Seed {approach_seed}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {MODEL}\n")
        f.write(f"Training: p={TRAIN_PASSAGES}, seed={TRAIN_SEED}\n")
        f.write(f"Eval passages: {EVAL_PASSAGES}\n")
        f.write(f"Eval seeds: {EVAL_SEEDS}\n")
        f.write(f"Max steps: {MAX_STEPS}\n")
        f.write(f"Candidates: {NUM_CANDIDATES}\n\n")

        # All candidate policies with scores
        f.write("ALL CANDIDATE POLICIES\n")
        f.write("-" * 80 + "\n")
        for i, code_str in enumerate(all_codes):
            score = all_scores[i] if i < len(all_scores) else None
            chosen = " << CHOSEN" if code_str == best_code else ""
            if score is not None:
                goal_reached, neg_cost = score
                f.write(
                    f"\n--- Policy {i} | goal_reached={goal_reached}, "
                    f"collision_checks={-neg_cost}{chosen} ---\n"
                )
            else:
                f.write(f"\n--- Policy {i} | FAILED{chosen} ---\n")
            f.write(code_str + "\n")

        # Evaluation results
        f.write("\n" + "=" * 80 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"{'Passages':<10} {'Eval Seed':<12} {'Reached':<10} "
            f"{'Steps':<8} {'Checks':<10} {'Nodes':<8}\n"
        )
        f.write("-" * 58 + "\n")
        for r in results:
            f.write(
                f"p{r['passages']:<9} {r['eval_seed']:<12} "
                f"{str(r['goal_reached']):<10} {r['total_steps']:<8} "
                f"{r['num_collision_checks']:<10} "
                f"{r['num_nodes_extended']:<8}\n"
            )
        successes = sum(1 for r in results if r["goal_reached"])
        f.write(f"\nSuccess: {successes}/{len(results)}\n")
    logging.info("Seed %d results written to: %s", approach_seed, path)


def write_aggregate_results(
    all_results: dict[int, list[dict]],
    best_codes: dict[int, str],
) -> None:
    """Write aggregate results across all seeds to a text file."""
    flat = [r for rs in all_results.values() for r in rs]
    path = RESULTS_DIR / "aggregate_results.txt"
    with open(path, "w", encoding="utf-8") as f:
        n_seeds = len(APPROACH_SEEDS)
        f.write(f"Agentic Integrated Motion2D — Aggregate Results ({n_seeds} seeds)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {MODEL}\n")
        f.write(f"Training: p={TRAIN_PASSAGES}, seed={TRAIN_SEED}\n")
        f.write(f"Approach seeds: {APPROACH_SEEDS}\n")
        f.write(f"Eval passages: {EVAL_PASSAGES}\n")
        f.write(f"Eval seeds: {EVAL_SEEDS}\n")
        f.write(f"Max steps: {MAX_STEPS}\n")
        f.write(f"Candidates per seed: {NUM_CANDIDATES}\n\n")

        # Per-seed summary
        f.write("PER-SEED SUMMARY\n")
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
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Passages':<10} {'Solved':<12} {'Success%':<12} "
            f"{'Avg Steps':<12} {'Avg Checks':<14} {'Avg Nodes':<10}\n"
        )
        f.write("-" * 70 + "\n")
        for p in EVAL_PASSAGES:
            p_results = [r for r in flat if r["passages"] == p]
            solved = sum(1 for r in p_results if r["goal_reached"])
            avg_steps = np.mean([r["total_steps"] for r in p_results])
            avg_checks = np.mean([r["num_collision_checks"] for r in p_results])
            avg_nodes = np.mean([r["num_nodes_extended"] for r in p_results])
            f.write(
                f"p{p:<9} {solved}/{len(p_results):<9} "
                f"{solved / len(p_results) * 100:<12.1f} "
                f"{avg_steps:<12.1f} {avg_checks:<14.1f} {avg_nodes:<10.1f}\n"
            )

        total_solved = sum(1 for r in flat if r["goal_reached"])
        f.write(
            f"\nOverall: {total_solved}/{len(flat)} "
            f"({total_solved / len(flat) * 100:.1f}%)\n"
        )

        # Chosen policies
        for seed in APPROACH_SEEDS:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"CHOSEN POLICY (seed {seed})\n")
            f.write(f"{'=' * 80}\n")
            f.write(best_codes[seed] + "\n")

    logging.info("Aggregate results written to: %s", path)


# --- Main --------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    """Synthesize policies and evaluate them across passages and seeds."""
    global MAX_STEPS  # pylint: disable=global-statement
    MAX_STEPS = args.max_steps

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Set up birrt metrics path
    metrics_file = Path(tempfile.NamedTemporaryFile(suffix=".json", delete=False).name)
    os.environ["birrt_metrics_path"] = str(metrics_file)

    cache = SQLite3PretrainedLargeModelCache(Path("llm_cache.db"))
    llm = OpenAIModel(MODEL, cache)

    all_results: dict[int, list[dict]] = {}
    best_codes: dict[int, str] = {}
    all_candidate_codes: dict[int, list[str]] = {}
    all_candidate_scores: dict[int, list] = {}

    for approach_seed in APPROACH_SEEDS:
        logging.info("%s", "=" * 60)
        logging.info("APPROACH SEED %d — synthesizing policies", approach_seed)
        logging.info("%s", "=" * 60)

        # Create training env (p=2)
        _register_env(TRAIN_PASSAGES)
        train_env = kinder.make(_env_id(TRAIN_PASSAGES), allow_state_access=True)
        train_inner = train_env.unwrapped
        train_obs, train_info = train_env.reset(seed=TRAIN_SEED)

        # Build planner context for training env
        robot = _get_robot(train_inner, train_obs)
        planner_context = {
            "get_object_centric_state": lambda o, _i=train_inner: _get_oc_state(_i, o),
            "robot": robot,
            "action_space": train_env.action_space,
        }

        score_fn = _make_score_fn(TRAIN_PASSAGES, TRAIN_SEED)

        approach = AgenticIntegratedApproach(
            MOTION2D_ENVIRONMENT_DESCRIPTION,
            train_env.observation_space,
            train_env.action_space,
            seed=approach_seed,
            llm=llm,
            planner_context=planner_context,
            planner_doc=BIRRT_PLANNER_DOC,
            init_doc=BIRRT_INIT_DOC,
            score_fn=score_fn,
            num_candidates=NUM_CANDIDATES,
            scoring_max_timesteps=MAX_STEPS,
        )
        approach.reset(train_obs, train_info)
        train_env.close()  # type: ignore[no-untyped-call]

        # pylint: disable=protected-access
        best_codes[approach_seed] = approach._best_code
        all_candidate_codes[approach_seed] = approach._all_candidate_codes
        all_candidate_scores[approach_seed] = approach._all_candidate_scores
        logging.info(
            "Chosen policy for seed %d:\n%s",
            approach_seed,
            approach._best_code,
        )

        # Evaluate on all passage counts
        seed_results = []
        for passages, eval_seed in zip(EVAL_PASSAGES, EVAL_SEEDS):
            result = run_eval_trial(
                approach=approach,
                passages=passages,
                eval_seed=eval_seed,
                approach_seed=approach_seed,
                max_steps=MAX_STEPS,
                save_video=args.video,
            )
            seed_results.append(result)

        all_results[approach_seed] = seed_results
        write_seed_results(
            approach_seed,
            seed_results,
            best_codes[approach_seed],
            all_candidate_codes[approach_seed],
            all_candidate_scores[approach_seed],
        )

    write_aggregate_results(all_results, best_codes)

    # Console summary
    flat = [r for rs in all_results.values() for r in rs]
    logging.info("%s", "\n" + "=" * 60)
    logging.info("SUMMARY")
    logging.info("%s", "=" * 60)
    logging.info(
        "%-10s %-12s %-12s %-14s %-10s",
        "Passages",
        "Solved",
        "Success%",
        "Avg Checks",
        "Avg Nodes",
    )
    logging.info("-" * 58)
    for p in EVAL_PASSAGES:
        p_results = [r for r in flat if r["passages"] == p]
        solved = sum(1 for r in p_results if r["goal_reached"])
        avg_checks = np.mean([r["num_collision_checks"] for r in p_results])
        avg_nodes = np.mean([r["num_nodes_extended"] for r in p_results])
        logging.info(
            "p%-9d %d/%-9d %-12.1f %-14.1f %-10.1f",
            p,
            solved,
            len(p_results),
            solved / len(p_results) * 100,
            avg_checks,
            avg_nodes,
        )
    total_solved = sum(1 for r in flat if r["goal_reached"])
    logging.info(
        "Overall: %d/%d (%.1f%%)",
        total_solved,
        len(flat),
        total_solved / len(flat) * 100,
    )

    # Clean up
    metrics_file.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Run AgenticIntegratedApproach on Motion2D")
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
        help="Save an MP4 for each eval trial to results/agentic/videos/",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
