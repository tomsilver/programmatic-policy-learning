"""Run LLM PPL baseline on MazeEnv for multiple trials."""

import glob
import logging
import tempfile
from pathlib import Path

import numpy as np
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel

from programmatic_policy_learning.approaches.llm_ppl_approach import LLMPPLApproach
from programmatic_policy_learning.envs.providers.maze_provider import MazeEnv

logging.basicConfig(level=logging.INFO)


def create_environment_description(env: MazeEnv, description_type: str) -> str:
    """Create environment description based on the type of maze.

    Args:
        env: The MazeEnv instance
        description_type: One of "maze_with_obstacles", "empty_inner", "completely_empty"
    """
    base_description = f"""
        Navigate a RxC grid to reach the goal.

        COORDINATE SYSTEM:
        - Your observation is (row, col)
        - Row 0 is at the top, row R is at the bottom
        - Col 0 is at the left, col C is at the right
        - Larger row values mean further down, larger col values mean further right

        ACTIONS:
        - 0 = North: row decreases by 1
        - 1 = South: row increases by 1
        - 2 = East: col increases by 1
        - 3 = West: col decreases by 1

        TASK:
        - The agent begins at a random position in the outer void region surrounding the maze.
        - The goal is at position {env.goal_pos}.
    """

    if description_type == "maze_with_obstacles":
        return base_description + f"""
        - There is a {env.inner_h}x{env.inner_w} inner maze with
          walls/obstacles (1s) that block movement.
        - The inner maze is surrounded by a wall border with a single entrance at the north.
        - The outer region (void) has no obstacles.
        - You must find the entrance, navigate through the maze avoiding walls, and reach the goal.

        Write a policy function that takes (row, col) and returns an action (0, 1, 2, or 3).
    """
    if description_type == "empty_inner":
        return base_description + f"""
        - There is a {env.inner_h}x{env.inner_w} inner area that is
          completely open (no obstacles inside).
        - The inner area is surrounded by a wall border with a single entrance at the north.
        - The outer region (void) has no obstacles.
        - You must find the entrance, enter the inner area, and navigate to the goal.

        Write a policy function that takes (row, col) and returns an action (0, 1, 2, or 3).
    """
    # completely_empty
    return base_description + """
        - The grid is completely open with no obstacles.
        - Navigate from start to goal.

        Write a policy function that takes (row, col) and returns an action (0, 1, 2, or 3).
    """


def run_trial(
    trial_num: int,
    env: MazeEnv,
    environment_description: str,
    max_steps: int = 200,
    enable_render: bool = False,
) -> dict:
    """Run a single trial of the LLM PPL approach on MazeEnv."""
    logging.info(f"=== Trial {trial_num + 1} ===")

    # Reset to get randomized start position
    obs, info = env.reset(seed=trial_num)

    # Create fresh cache for each trial to ensure different LLM responses
    cache_path = Path(tempfile.NamedTemporaryFile(suffix=".db", delete=False).name)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    llm = OpenAIModel("gpt-4o-mini", cache)

    # Create approach with trial-specific seed
    approach = LLMPPLApproach(
        environment_description,
        env.observation_space,
        env.action_space,
        seed=trial_num,
        llm=llm,
    )

    approach.reset(obs, info)

    # Log the generated policy
    logging.info(f"Start: {obs}, Goal: {env.goal_pos}")
    generated_policy = getattr(approach, "policy", None)
    if generated_policy is None:
        generated_policy = vars(approach).get("_policy")
    logging.info("Generated policy:\n%s", generated_policy)

    # Run the episode
    goal_reached = False
    total_steps = 0
    total_reward = 0.0

    for step in range(max_steps):
        action = approach.step()
        obs, reward, terminated, _, info = env.step(action)
        total_reward += float(reward)
        total_steps = step + 1

        if enable_render:
            env.render()

        approach.update(obs, float(reward), terminated, info)

        if terminated:
            goal_reached = True
            logging.info(f"Goal reached in {total_steps} steps!")
            break

    if not goal_reached:
        logging.info(f"Failed to reach goal in {max_steps} steps")

    return {
        "trial": trial_num,
        "start_pos": obs,
        "goal_pos": env.goal_pos,
        "goal_reached": goal_reached,
        "total_steps": total_steps,
        "total_reward": total_reward,
    }


def run_experiment_set(
    experiment_name: str,
    envs_and_names: list[tuple[MazeEnv, str]],
    description_type: str,
    num_trials: int,
    max_steps: int,
    enable_render: bool,
) -> list[dict]:
    """Run a set of experiments and return results."""
    logging.info("=" * 80)
    logging.info(f"EXPERIMENT: {experiment_name}")
    logging.info("=" * 80)

    all_results = []

    for env, name in envs_and_names:
        logging.info(f"\n--- {name} ---")
        env_description = create_environment_description(env, description_type)

        for trial in range(num_trials):
            result = run_trial(
                trial_num=trial,
                env=env,
                environment_description=env_description,
                max_steps=max_steps,
                enable_render=enable_render,
            )
            result["maze_name"] = name
            result["experiment"] = experiment_name
            all_results.append(result)
            logging.info("-" * 40)

        env.close()

    return all_results


def main() -> None:
    """Run multiple experiment sets and report success rates."""
    num_trials = 10
    max_steps = 150
    enable_render = False
    outer_margin = 10

    all_results = []

    # =========================================================================
    # EXPERIMENT 1: Run on all mazes from data/mazes/ (like compare_maze_approaches.py)
    # =========================================================================
    maze_dir = Path("data/mazes")
    maze_files = sorted(glob.glob(str(maze_dir / "*.npy")))
    logging.info(f"Found {len(maze_files)} maze files in {maze_dir}")

    envs_and_names = []
    for maze_file in maze_files:
        inner_maze = np.load(maze_file)
        env = MazeEnv(
            inner_maze=inner_maze,
            outer_margin=outer_margin,
            enable_render=enable_render,
        )
        envs_and_names.append((env, Path(maze_file).stem))

    results_1 = run_experiment_set(
        experiment_name="Mazes with Obstacles (from data/mazes/)",
        envs_and_names=envs_and_names,
        description_type="maze_with_obstacles",
        num_trials=1,  # 1 trial per maze
        max_steps=max_steps,
        enable_render=enable_render,
    )
    all_results.extend(results_1)

    # =========================================================================
    # EXPERIMENT 2: 10 trials with empty inner_maze (all 0's)
    # =========================================================================
    inner_maze_empty = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    env_empty_inner = MazeEnv(
        inner_maze=inner_maze_empty,
        outer_margin=outer_margin,
        enable_render=enable_render,
    )

    results_2 = run_experiment_set(
        experiment_name="Empty Inner Maze (7x7 all 0s, with wall border)",
        envs_and_names=[(env_empty_inner, "empty_inner_7x7")],
        description_type="empty_inner",
        num_trials=num_trials,
        max_steps=max_steps,
        enable_render=enable_render,
    )
    all_results.extend(results_2)

    # =========================================================================
    # EXPERIMENT 3: Completely empty grid with no obstacles
    # Using the same empty inner_maze but removing the wall border
    # =========================================================================
    inner_maze_empty_2 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    env_completely_empty = MazeEnv(
        inner_maze=inner_maze_empty_2,
        outer_margin=outer_margin,
        enable_render=enable_render,
    )

    # Remove all walls from the grid to make it completely empty
    env_completely_empty.grid[:, :] = 0

    results_3 = run_experiment_set(
        experiment_name="Completely Empty Grid (no obstacles)",
        envs_and_names=[(env_completely_empty, "completely_empty")],
        description_type="completely_empty",
        num_trials=num_trials,
        max_steps=max_steps,
        enable_render=enable_render,
    )
    all_results.extend(results_3)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    logging.info("%s", "\n" + "=" * 80)
    logging.info("OVERALL SUMMARY")
    logging.info("=" * 80)

    # Group results by experiment
    experiments: dict[str, list[dict]] = {}
    for r in all_results:
        exp_name = r["experiment"]
        if exp_name not in experiments:
            experiments[exp_name] = []
        experiments[exp_name].append(r)

    for exp_name, exp_results in experiments.items():
        successes = sum(1 for r in exp_results if r["goal_reached"])
        total = len(exp_results)
        success_rate = successes / total * 100
        logging.info(f"\n{exp_name}:")
        logging.info(f"  Total trials: {total}")
        logging.info(f"  Successes: {successes}")
        logging.info(f"  Success rate: {success_rate:.1f}%")

    # Write results to file
    output_file = Path("llm_ppl_maze_results.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("LLM PPL MazeEnv Baseline Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Max steps per trial: {max_steps}\n")
        f.write(f"Outer margin: {outer_margin}\n\n")

        for exp_name, exp_results in experiments.items():
            successes = sum(1 for r in exp_results if r["goal_reached"])
            total = len(exp_results)
            success_rate = successes / total * 100

            f.write("=" * 80 + "\n")
            f.write(f"EXPERIMENT: {exp_name}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Success rate: {success_rate:.1f}% ({successes}/{total})\n\n")

            f.write("Detailed Results:\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"{'Trial':<8} {'Maze':<20} {'Goal':<15} {'Reached':<10} {'Steps':<8}\n"
            )
            f.write("-" * 80 + "\n")
            for r in exp_results:
                f.write(
                    f"{r['trial'] + 1:<8} {r['maze_name']:<20} {str(r['goal_pos']):<15} "
                    f"{str(r['goal_reached']):<10} {r['total_steps']:<8}\n"
                )
            f.write("\n")

    logging.info(f"\nResults written to: {output_file}")


if __name__ == "__main__":
    main()
