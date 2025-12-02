"""Hydra-compatible script to compare SearchApproach and ExpertMazeApproach on
all mazes."""

import glob
import logging
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from programmatic_policy_learning.approaches.expert_approach import ExpertApproach
from programmatic_policy_learning.approaches.experts.maze_experts import (
    create_expert_maze_with_outer_world_policy,
)
from programmatic_policy_learning.approaches.search_approach import SearchApproach
from programmatic_policy_learning.envs.providers.maze_provider import MazeEnv


def run_approach_on_maze(
    approach: Any, env: MazeEnv, max_steps: int = 1000
) -> dict[str, Any]:
    """Run an approach on a maze and return metrics."""
    obs, info = env.reset(seed=123)
    approach.reset(obs, info)

    goal_reached = False
    for _ in range(max_steps):
        action = approach.step()
        obs, rew, done, _, info = env.step(action)
        reward = float(rew)
        approach.update(obs, reward, done, info)
        if done:
            goal_reached = True
            break

    # Get metrics
    if hasattr(approach, "metrics"):
        metrics = approach.metrics
    elif hasattr(approach, "metrics"):
        metrics = approach.metrics
    else:
        raise AttributeError("Approach has no metrics")

    return {
        "goal_reached": goal_reached,
        "num_evals": metrics.num_evals,
        "num_expansions": metrics.num_expansions,
    }


@hydra.main(version_base=None, config_name="compare_config", config_path="conf/")
def main(cfg: DictConfig) -> None:
    """Run experiments on all mazes."""
    logging.info("Starting maze comparison experiments")

    # Find all maze files
    maze_dir = Path(cfg.maze_dir)
    maze_files = sorted(glob.glob(str(maze_dir / "*.npy")))

    logging.info(f"Found {len(maze_files)} maze files")

    # Storage for results
    search_results = []
    oracle_results = []

    for maze_file in maze_files:
        maze_name = Path(maze_file).stem
        logging.info(f"Running {maze_name}...")

        # Load maze
        inner_maze = np.load(maze_file)

        # Create environment
        env = MazeEnv(
            inner_maze=inner_maze,
            outer_margin=cfg.outer_margin,
            enable_render=cfg.enable_render,
        )

        # Run SearchApproach
        search_approach = SearchApproach(
            environment_description="Maze environment",
            observation_space=env.observation_space,
            action_space=env.action_space,
            seed=cfg.seed,
            get_actions=env.get_actions,
            get_next_state=env.get_next_state,
            get_cost=env.get_cost,
            check_goal=env.check_goal,
        )
        search_metrics = run_approach_on_maze(search_approach, env, cfg.max_steps)
        search_results.append(search_metrics)
        logging.info(
            f"  SearchApproach: Goal={search_metrics['goal_reached']}, "
            f"Evals={search_metrics['num_evals']}, "
            f"Expansions={search_metrics['num_expansions']}"
        )

        # Reset environment for next approach
        env = MazeEnv(
            inner_maze=inner_maze,
            outer_margin=cfg.outer_margin,
            enable_render=cfg.enable_render,
        )

        # Run ExpertMazeApproach
        expert_policy = create_expert_maze_with_outer_world_policy(
            grid=env.grid.copy(),
            goal=env.goal_pos,
            inner_h=env.inner_h,
            inner_w=env.inner_w,
            get_actions=env.get_actions,
            get_next_state=env.get_next_state,
            get_cost=env.get_cost,
            check_goal=env.check_goal,
        )
        oracle_approach: ExpertApproach = ExpertApproach(
            environment_description="Maze environment",
            observation_space=env.observation_space,
            action_space=env.action_space,
            seed=cfg.seed,
            expert_fn=expert_policy,
        )
        oracle_metrics = run_approach_on_maze(oracle_approach, env, cfg.max_steps)
        oracle_results.append(oracle_metrics)
        logging.info(
            f"  Oracle Approach: Goal={oracle_metrics['goal_reached']}, "
            f"Evals={oracle_metrics['num_evals']}, "
            f"Expansions={oracle_metrics['num_expansions']}"
        )

    # Write results to file
    output_file = Path(cfg.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MAZE SEARCH COMPARISON RESULTS\n")
        f.write("=" * 80 + "\n\n")

        # Individual maze results
        for i, maze_file in enumerate(maze_files):
            maze_name = Path(maze_file).stem
            f.write(f"Maze: {maze_name}\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"{'Approach':<25} {'Goal Reached':<15} "
                + f"{'Num Evals':<15} {'Num Expansions':<15}\n"
            )
            f.write("-" * 80 + "\n")

            # Search Approach
            sr = search_results[i]
            f.write(
                f"{'Search Approach':<25} {str(sr['goal_reached']):<15} "
                f"{sr['num_evals']:<15} {sr['num_expansions']:<15}\n"
            )

            # Oracle Approach
            or_ = oracle_results[i]
            f.write(
                f"{'Oracle Approach':<25} {str(or_['goal_reached']):<15} "
                f"{or_['num_evals']:<15} {or_['num_expansions']:<15}\n"
            )
            f.write("\n")

        # Summary statistics
        f.write("=" * 80 + "\n")
        f.write("TOTAL (AVERAGES)\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"{'Approach':<25} {'Goal Reached':<15} "
            + f"{'Num Evals':<15} {'Num Expansions':<15}\n"
        )
        f.write("-" * 80 + "\n")

        # Calculate averages for SearchApproach
        search_goal_rate = (
            sum(r["goal_reached"] for r in search_results) / len(search_results) * 100
        )
        search_avg_evals = np.mean([r["num_evals"] for r in search_results])
        search_avg_expansions = np.mean([r["num_expansions"] for r in search_results])

        f.write(
            f"{'Search Approach':<25} {f'{search_goal_rate:.1f}%':<15} "
            f"{search_avg_evals:<15.1f} {search_avg_expansions:<15.1f}\n"
        )

        # Calculate averages for Oracle Approach
        oracle_goal_rate = (
            sum(r["goal_reached"] for r in oracle_results) / len(oracle_results) * 100
        )
        oracle_avg_evals = np.mean([r["num_evals"] for r in oracle_results])
        oracle_avg_expansions = np.mean([r["num_expansions"] for r in oracle_results])

        f.write(
            f"{'Oracle Approach':<25} {f'{oracle_goal_rate:.1f}%':<15} "
            f"{oracle_avg_evals:<15.1f} {oracle_avg_expansions:<15.1f}\n"
        )

    logging.info(f"Results written to: {output_file}")
    logging.info(
        f"Search Approach - Goal Rate: {search_goal_rate:.1f}%, "
        f"Avg Evals: {search_avg_evals:.1f}, Avg Expansions: {search_avg_expansions:.1f}"
    )
    logging.info(
        f"Oracle Approach - Goal Rate: {oracle_goal_rate:.1f}%, "
        f"Avg Evals: {oracle_avg_evals:.1f}, Avg Expansions: {oracle_avg_expansions:.1f}"
    )

    # Generate visualizations
    logging.info("Generating visualizations...")
    plot_dir = Path(cfg.output_file).parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Extract maze names
    maze_names = [Path(f).stem for f in maze_files]

    # Plot 1: Bar chart comparing expansions for each maze
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(maze_names) + 1)  # +1 for average
    width = 0.35

    search_expansions = [r["num_expansions"] for r in search_results] + [
        search_avg_expansions
    ]
    oracle_expansions = [r["num_expansions"] for r in oracle_results] + [
        oracle_avg_expansions
    ]

    bars1 = ax.bar(
        x - width / 2, search_expansions, width, label="Search Approach", color="steelblue"
    )
    bars2 = ax.bar(
        x + width / 2, oracle_expansions, width, label="Oracle Approach", color="coral"
    )

    ax.set_xlabel("Maze", fontsize=12)
    ax.set_ylabel("Number of Expansions", fontsize=12)
    ax.set_title(
        f"Search Expansions Comparison (Outer Margin={cfg.outer_margin})", fontsize=14
    )
    ax.set_xticks(x)
    ax.set_xticklabels(maze_names + ["Average"], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    expansions_plot = plot_dir / "expansions_comparison.png"
    plt.savefig(expansions_plot, dpi=300, bbox_inches="tight")
    logging.info(f"Saved expansions bar chart to: {expansions_plot}")
    plt.close()

    # Plot 2: Bar chart comparing evaluations for each maze
    fig, ax = plt.subplots(figsize=(14, 6))

    search_evals = [r["num_evals"] for r in search_results] + [search_avg_evals]
    oracle_evals = [r["num_evals"] for r in oracle_results] + [oracle_avg_evals]

    bars1 = ax.bar(
        x - width / 2, search_evals, width, label="Search Approach", color="steelblue"
    )
    bars2 = ax.bar(
        x + width / 2, oracle_evals, width, label="Oracle Approach", color="coral"
    )

    ax.set_xlabel("Maze", fontsize=12)
    ax.set_ylabel("Number of Evaluations", fontsize=12)
    ax.set_title(
        f"State Evaluations Comparison (Outer Margin={cfg.outer_margin})", fontsize=14
    )
    ax.set_xticks(x)
    ax.set_xticklabels(maze_names + ["Average"], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    evals_plot = plot_dir / "evaluations_comparison.png"
    plt.savefig(evals_plot, dpi=300, bbox_inches="tight")
    logging.info(f"Saved evaluations bar chart to: {evals_plot}")
    plt.close()

    # Plot 3: Side-by-side comparison (both metrics)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Expansions
    bars1 = ax1.bar(
        x - width / 2, search_expansions, width, label="Search Approach", color="steelblue"
    )
    bars2 = ax1.bar(
        x + width / 2, oracle_expansions, width, label="Oracle Approach", color="coral"
    )
    ax1.set_xlabel("Maze", fontsize=12)
    ax1.set_ylabel("Number of Expansions", fontsize=12)
    ax1.set_title("Node Expansions", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(maze_names + ["Avg"], rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Evaluations
    bars1 = ax2.bar(
        x - width / 2, search_evals, width, label="Search Approach", color="steelblue"
    )
    bars2 = ax2.bar(
        x + width / 2, oracle_evals, width, label="Oracle Approach", color="coral"
    )
    ax2.set_xlabel("Maze", fontsize=12)
    ax2.set_ylabel("Number of Evaluations", fontsize=12)
    ax2.set_title("State Evaluations", fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(maze_names + ["Avg"], rotation=45, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Search Metrics Comparison (Outer Margin={cfg.outer_margin})", fontsize=16
    )
    plt.tight_layout()
    combined_plot = plot_dir / "combined_comparison.png"
    plt.savefig(combined_plot, dpi=300, bbox_inches="tight")
    logging.info(f"Saved combined comparison to: {combined_plot}")
    plt.close()

    logging.info("All visualizations generated successfully!")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
