"""Run AgenticIntegratedApproach on MazeEnv for all mazes in data/mazes/."""

import glob
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel

from programmatic_policy_learning.approaches.agentic_integrated_approach import (
    ASTAR_INIT_DOC,
    ASTAR_PLANNER_DOC,
    AgenticIntegratedApproach,
    score_policy_maze,
)
from programmatic_policy_learning.envs.providers.maze_provider import MazeEnv

logging.basicConfig(level=logging.INFO)

ENVIRONMENT_DESCRIPTION = """
    A maze environment with an outer void, a wall border, and an inner maze.

    COORDINATE SYSTEM:
    - Observation is (row, col).
    - Rows increase going South, columns increase going East.
    - The inner maze occupies rows [0 .. inner_height-1], cols [0 .. inner_width-1].
      The inner dimensions are not known in advance and vary between episodes.

    LAYOUT:
    - Outer void: an obstacle-free area surrounding the maze on all four
      sides. The agent starts here each episode. The void extends
      arbitrarily far in every direction.
    - Wall border: a one-cell-thick solid rectangle enclosing the inner
      maze with exactly ONE gap — the entrance at (-1, 0). Moving into
      any wall cell is blocked (get_next_state returns the same state).
    - Inner maze: a grid starting at (0, 0) with unknown internal walls.

    ACTIONS:
    - Action 0 (North): row -= 1.
    - Action 1 (South): row += 1.
    - Action 2 (East):  col += 1.
    - Action 3 (West):  col -= 1.

    WALL BORDER NAVIGATION:
    The wall border is a continuous barrier. You cannot move along or
    through it — only through the entrance at (-1, 0). Key consequences:
    - Row -1 is entirely wall EXCEPT col 0 (the entrance). If the agent
      reaches row -1 at any other column, it will be stuck — no East/West
      movement is possible along row -1.
    - Col -1 and col inner_width are entirely wall. No North/South
      movement is possible along those columns.
    - To navigate from the void to the entrance, the agent must stay clear
      of the wall border (row <= -2 or row >= inner_height+1 for horizontal
      movement, col <= -2 or col >= inner_width+1 for vertical movement)
      and approach the entrance at (-1, 0) from row -2, col 0.
    - From the entrance (-1, 0), moving South enters the maze at (0, 0).

    TASK:
    Navigate from the starting position in the outer void to the goal at the
    bottom-right corner of the inner maze. The goal position is provided in
    the info dict passed to reset(obs, info) as info["goal"].
"""


def run_evaluation(
    approach: AgenticIntegratedApproach,
    env: MazeEnv,
    maze_name: str,
    seed: int = 123,
    max_steps: int = 1000,
) -> dict:
    """Evaluate the approach on a single maze instance."""
    obs, info = env.reset(seed=seed)

    approach.update_planner_context(
        {
            "get_actions": env.get_actions,
            "get_next_state": env.get_next_state,
            "get_cost": env.get_cost,
            "check_goal": env.check_goal,
        }
    )

    # Clear metrics before evaluation (must be before reset, since some
    # policies plan inside reset).
    metrics_path_str = os.environ.get("astar_metrics_path")
    if metrics_path_str:
        Path(metrics_path_str).write_text("", encoding="utf-8")

    approach.reset(obs, info)

    goal_reached = False
    total_steps = 0

    for step in range(max_steps):
        action = approach.step()
        assert env.action_space.contains(action)
        obs, reward, terminated, _, info = env.step(action)
        approach.update(obs, float(reward), terminated, info)
        total_steps = step + 1

        if terminated:
            goal_reached = True
            break

    # Read search metrics
    total_evals = 0
    total_expansions = 0
    if metrics_path_str:
        metrics_path = Path(metrics_path_str)
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        total_evals += entry["num_evals"]
                        total_expansions += entry["num_expansions"]

    result = {
        "maze_name": maze_name,
        "goal_reached": goal_reached,
        "total_steps": total_steps,
        "num_evals": total_evals,
        "num_expansions": total_expansions,
        "goal_pos": env.goal_pos,
    }

    status = "REACHED" if goal_reached else "FAILED"
    logging.info(
        "%s | %s | steps=%d | expansions=%d",
        maze_name,
        status,
        total_steps,
        total_expansions,
    )
    return result


# Baseline results from compare_maze_approaches.py (outer_margin=10, seed=123).
# Keys are maze names; values are (search_expansions, oracle_expansions).
BASELINE_RESULTS = {
    "maze1_10x10": (259, 46),
    "maze1_15x15": (215, 29),
    "maze2_10x10": (541, 318),
    "maze2_15x15": (1076, 907),
    "maze3_10x10": (471, 19),
    "maze3_15x15": (723, 293),
    "maze4_10x10": (781, 535),
    "maze4_15x15": (240, 53),
    "maze5_10x10": (235, 20),
    "maze5_15x15": (471, 84),
}


def main() -> None:
    """Run 5 seeds of the agentic approach, evaluate each on all mazes."""
    outer_margin = 10
    max_steps = 1000
    seeds = [123, 124, 125, 126, 127]
    env_seed = 123  # Fixed seed for all env-related operations

    # Set up astar metrics path
    metrics_path = Path(tempfile.NamedTemporaryFile(suffix=".json", delete=False).name)
    os.environ["astar_metrics_path"] = str(metrics_path)

    # Load all mazes
    maze_dir = Path("data/mazes")
    maze_files = sorted(glob.glob(str(maze_dir / "*.npy")))
    if not maze_files:
        raise FileNotFoundError(f"No .npy maze files found in {maze_dir}")
    logging.info("Found %d maze files in %s", len(maze_files), maze_dir)

    # Use the first maze for training
    train_maze_file = maze_files[0]
    train_maze = np.load(train_maze_file)
    train_name = Path(train_maze_file).stem

    # Set up LLM
    cache_path = Path("llm_cache.db")
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    llm = OpenAIModel("gpt-5.2", cache)

    # Output directory
    output_dir = Path("experiments/maze_experiments/agentic_maze")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results: seed -> list of per-maze result dicts
    all_seed_results: dict[int, list[dict]] = {}
    best_codes: dict[int, str] = {}
    all_candidate_codes: dict[int, list[str]] = {}
    all_candidate_scores: dict[int, list] = {}

    for seed in seeds:
        logging.info("%s", "\n" + "=" * 80)
        logging.info("SEED %d", seed)
        logging.info("%s", "=" * 80)

        # Create training env and get initial obs
        train_env = MazeEnv(
            inner_maze=train_maze, outer_margin=outer_margin, enable_render=False
        )
        train_env.action_space.seed(env_seed)
        obs, info = train_env.reset(seed=env_seed)

        logging.info(
            "Training on %s (%dx%d)",
            train_name,
            train_maze.shape[0],
            train_maze.shape[1],
        )

        # Build maze-specific planner context and scoring function
        maze_planner_context = {
            "get_actions": train_env.get_actions,
            "get_next_state": train_env.get_next_state,
            "get_cost": train_env.get_cost,
            "check_goal": train_env.check_goal,
        }

        def _maze_score_fn(  # type: ignore[no-untyped-def]
            policy,
            obs,
            info,
            max_ts,
            _gns=train_env.get_next_state,
            _cg=train_env.check_goal,
        ):
            return score_policy_maze(
                policy,
                obs,
                info,
                max_ts,
                get_next_state=_gns,
                check_goal=_cg,
            )

        # Create a fresh approach for this seed
        approach = AgenticIntegratedApproach(
            ENVIRONMENT_DESCRIPTION,
            train_env.observation_space,
            train_env.action_space,
            seed=seed,
            llm=llm,
            planner_context=maze_planner_context,
            planner_doc=ASTAR_PLANNER_DOC,
            init_doc=ASTAR_INIT_DOC,
            score_fn=_maze_score_fn,
        )
        approach.reset(obs, info)
        train_env.close()

        # pylint: disable=protected-access
        best_codes[seed] = approach._best_code
        all_candidate_codes[seed] = approach._all_candidate_codes
        all_candidate_scores[seed] = approach._all_candidate_scores
        logging.info("Chosen policy for seed %d:\n%s", seed, approach._best_code)

        # Evaluate on all mazes
        seed_results = []
        for maze_file in maze_files:
            maze_name = Path(maze_file).stem
            inner_maze = np.load(maze_file)
            env = MazeEnv(
                inner_maze=inner_maze, outer_margin=outer_margin, enable_render=False
            )
            result = run_evaluation(
                approach, env, maze_name, seed=123, max_steps=max_steps
            )
            seed_results.append(result)
            env.close()

        all_seed_results[seed] = seed_results

        # Write per-seed results file
        seed_file = output_dir / f"seed_{seed}.txt"
        with open(seed_file, "w", encoding="utf-8") as f:
            f.write(f"Agentic Integrated Approach - Seed {seed}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Max steps per maze: {max_steps}\n")
            f.write(f"Outer margin: {outer_margin}\n")
            f.write(f"Training maze: {train_name}\n\n")

            # All candidate policies with scores
            f.write("ALL CANDIDATE POLICIES\n")
            f.write("-" * 80 + "\n")
            for i, code_str in enumerate(all_candidate_codes[seed]):
                score = all_candidate_scores[seed][i]
                chosen = " << CHOSEN" if code_str == best_codes[seed] else ""
                if score is not None:
                    goal_reached, neg_exp = score
                    f.write(
                        f"\n--- Policy {i} | goal_reached={goal_reached}, "
                        f"expansions={-neg_exp}{chosen} ---\n"
                    )
                else:
                    f.write(f"\n--- Policy {i} | FAILED{chosen} ---\n")
                f.write(code_str + "\n")

            # Evaluation results
            f.write("\n" + "=" * 80 + "\n")
            f.write("EVALUATION RESULTS\n")
            f.write("=" * 80 + "\n")
            f.write(
                f"{'Maze':<16} {'Reached':<9} {'Steps':<7} " f"{'Expansions':<12}\n"
            )
            f.write("-" * 44 + "\n")
            for r in seed_results:
                f.write(
                    f"{r['maze_name']:<16} {str(r['goal_reached']):<9} "
                    f"{r['total_steps']:<7} {r['num_expansions']:<12}\n"
                )
            successes = sum(1 for r in seed_results if r["goal_reached"])
            f.write(f"\nSuccess: {successes}/{len(seed_results)}\n")
        logging.info("Seed %d results written to: %s", seed, seed_file)

    # ── Summary across all seeds ──────────────────────────────────────────
    logging.info("%s", "\n" + "=" * 80)
    logging.info("AGGREGATE RESULTS (5 seeds x %d mazes)", len(maze_files))
    logging.info("%s", "=" * 80)

    # Per-seed summary
    for seed in seeds:
        results = all_seed_results[seed]
        successes = sum(1 for r in results if r["goal_reached"])
        solved_exp = [r["num_expansions"] for r in results if r["goal_reached"]]
        avg_exp = np.mean(solved_exp) if solved_exp else float("nan")
        logging.info(
            "Seed %d: %d/%d solved (%.1f%%), avg expansions %.1f",
            seed,
            successes,
            len(results),
            successes / len(results) * 100,
            avg_exp,
        )

    # Per-maze summary (averaged across seeds)
    logging.info("")
    header = (
        f"{'Maze':<16} {'Solved':<8} {'Avg Steps':<11} "
        f"{'Avg Exp':<10} {'Search':<10} {'Oracle':<10}"
    )
    logging.info(header)
    logging.info("-" * 65)
    maze_names = [Path(f).stem for f in maze_files]
    for i, maze_name in enumerate(maze_names):
        maze_results = [all_seed_results[s][i] for s in seeds]
        solved = sum(1 for r in maze_results if r["goal_reached"])
        avg_steps = np.mean([r["total_steps"] for r in maze_results])
        solved_exp = [r["num_expansions"] for r in maze_results if r["goal_reached"]]
        avg_exp = np.mean(solved_exp) if solved_exp else float("nan")
        search_exp, oracle_exp = BASELINE_RESULTS.get(maze_name, (0, 0))
        tag = " *" if maze_name == train_name else ""
        logging.info(
            "%-16s %d/%-5d %-11.1f %-10.1f %-10d %-10d%s",
            maze_name,
            solved,
            len(seeds),
            avg_steps,
            avg_exp,
            search_exp,
            oracle_exp,
            tag,
        )

    all_results_flat = [r for rs in all_seed_results.values() for r in rs]
    total_solved = sum(1 for r in all_results_flat if r["goal_reached"])
    total_count = len(all_results_flat)
    solved_exp_flat = [
        r["num_expansions"] for r in all_results_flat if r["goal_reached"]
    ]
    overall_avg_exp = np.mean(solved_exp_flat) if solved_exp_flat else float("nan")
    logging.info("-" * 65)
    logging.info(
        "Overall: %d/%d (%.1f%%)  |  Avg expansions: %.1f",
        total_solved,
        total_count,
        total_solved / total_count * 100,
        overall_avg_exp,
    )

    # Write aggregate results file
    output_file = output_dir / "aggregate_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Agentic Integrated Approach - Maze Results (5 seeds)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Max steps per maze: {max_steps}\n")
        f.write(f"Outer margin: {outer_margin}\n")
        f.write(f"Training maze: {train_name}\n")
        f.write("* = training maze\n\n")

        # Per-seed detail
        for seed in seeds:
            f.write(f"\n--- Seed {seed} ---\n")
            results = all_seed_results[seed]
            f.write(
                f"{'Maze':<16} {'Reached':<9} {'Steps':<7} " f"{'Expansions':<12}\n"
            )
            f.write("-" * 44 + "\n")
            for r in results:
                f.write(
                    f"{r['maze_name']:<16} {str(r['goal_reached']):<9} "
                    f"{r['total_steps']:<7} {r['num_expansions']:<12}\n"
                )
            successes = sum(1 for r in results if r["goal_reached"])
            f.write(f"Success: {successes}/{len(results)}\n")

        # Aggregate table
        f.write("\n" + "=" * 80 + "\n")
        f.write("AGGREGATE (averaged across seeds)\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"{'Maze':<16} {'Solved':<8} {'Avg Steps':<11} "
            f"{'Avg Exp':<10} {'Search':<10} {'Oracle':<10}\n"
        )
        f.write("-" * 65 + "\n")
        for i, maze_name in enumerate(maze_names):
            maze_results = [all_seed_results[s][i] for s in seeds]
            solved = sum(1 for r in maze_results if r["goal_reached"])
            avg_steps = np.mean([r["total_steps"] for r in maze_results])
            solved_exp = [
                r["num_expansions"] for r in maze_results if r["goal_reached"]
            ]
            avg_exp = np.mean(solved_exp) if solved_exp else float("nan")
            search_exp, oracle_exp = BASELINE_RESULTS.get(maze_name, (0, 0))
            tag = " *" if maze_name == train_name else ""
            f.write(
                f"{maze_name:<16} {solved}/{len(seeds):<5} {avg_steps:<11.1f} "
                f"{avg_exp:<10.1f} {search_exp:<10} {oracle_exp:<10}{tag}\n"
            )
        f.write(
            f"\nOverall: {total_solved}/{total_count} "
            f"({total_solved / total_count * 100:.1f}%)\n"
        )
        avg_search = sum(v[0] for v in BASELINE_RESULTS.values()) / len(
            BASELINE_RESULTS
        )
        avg_oracle = sum(v[1] for v in BASELINE_RESULTS.values()) / len(
            BASELINE_RESULTS
        )
        f.write(
            f"Avg expansions — Agentic: {overall_avg_exp:.1f}  "
            f"Search: {avg_search:.1f}  "
            f"Oracle: {avg_oracle:.1f}\n"
        )

        # Chosen policies
        for seed in seeds:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"CHOSEN POLICY (seed {seed})\n")
            f.write(f"{'=' * 80}\n")
            f.write(best_codes[seed] + "\n")

    logging.info("Results written to: %s", output_file)

    # Clean up
    metrics_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
