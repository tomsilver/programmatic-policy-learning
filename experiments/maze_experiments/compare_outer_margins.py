"""Hydra-compatible script to compare approaches across different outer
margins."""

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


@hydra.main(version_base=None, config_name="outer_margin_config", config_path="conf/")
def main(cfg: DictConfig) -> None:
    """Run experiments across different outer margins."""
    logging.info("Starting outer margin comparison experiments")

    # Find all maze files
    maze_dir = Path(cfg.maze_dir)
    maze_files = sorted(glob.glob(str(maze_dir / "*.npy")))

    logging.info(f"Found {len(maze_files)} maze files")
    logging.info(f"Testing outer margins: {cfg.outer_margins}")

    # Storage for results: {outer_margin: {maze_name: {approach: metrics}}}
    results_by_margin: dict[int, dict[str, dict[str, dict]]] = {}

    for outer_margin in cfg.outer_margins:
        logging.info(f"\n{'='*80}")
        logging.info(f"Testing outer_margin = {outer_margin}")
        logging.info(f"{'='*80}")

        results_by_margin[outer_margin] = {}

        for maze_file in maze_files:
            maze_name = Path(maze_file).stem
            logging.info(f"  Running {maze_name} with outer_margin={outer_margin}...")

            # Load maze
            inner_maze = np.load(maze_file)

            results_by_margin[outer_margin][maze_name] = {}

            # Run SearchApproach
            env = MazeEnv(
                inner_maze=inner_maze,
                outer_margin=outer_margin,
                enable_render=cfg.enable_render,
            )
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
            results_by_margin[outer_margin][maze_name]["search"] = search_metrics
            logging.info(
                f"    SearchApproach: Goal={search_metrics['goal_reached']}, "
                f"Evals={search_metrics['num_evals']}, "
                f"Expansions={search_metrics['num_expansions']}"
            )

            # Run ExpertMazeApproach
            env = MazeEnv(
                inner_maze=inner_maze,
                outer_margin=outer_margin,
                enable_render=cfg.enable_render,
            )
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
            results_by_margin[outer_margin][maze_name]["oracle"] = oracle_metrics
            logging.info(
                f"    Oracle Approach: Goal={oracle_metrics['goal_reached']}, "
                f"Evals={oracle_metrics['num_evals']}, "
                f"Expansions={oracle_metrics['num_expansions']}"
            )

    # Write results to file
    output_file = Path(cfg.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("OUTER MARGIN COMPARISON RESULTS\n")
        f.write("=" * 100 + "\n\n")

        # Table header
        f.write(
            f"{'Outer Margin':<15} {'Approach':<20} {'Goal Rate':<15} "
            f"{'Avg Evals':<20} {'Avg Expansions':<20}\n"
        )
        f.write("-" * 100 + "\n")

        # Results for each outer margin
        for outer_margin in cfg.outer_margins:
            margin_results = results_by_margin[outer_margin]

            # Calculate averages for SearchApproach
            search_results = [
                maze_data["search"] for maze_data in margin_results.values()
            ]
            search_goal_rate = (
                sum(r["goal_reached"] for r in search_results)
                / len(search_results)
                * 100
            )
            search_avg_evals = np.mean([r["num_evals"] for r in search_results])
            search_avg_expansions = np.mean(
                [r["num_expansions"] for r in search_results]
            )

            # Calculate averages for Oracle Approach
            oracle_results = [
                maze_data["oracle"] for maze_data in margin_results.values()
            ]
            oracle_goal_rate = (
                sum(r["goal_reached"] for r in oracle_results)
                / len(oracle_results)
                * 100
            )
            oracle_avg_evals = np.mean([r["num_evals"] for r in oracle_results])
            oracle_avg_expansions = np.mean(
                [r["num_expansions"] for r in oracle_results]
            )

            # Write search approach row
            f.write(
                f"{outer_margin:<15} {'Search Approach':<20} "
                f"{f'{search_goal_rate:.1f}%':<15} "
                f"{search_avg_evals:<20.1f} {search_avg_expansions:<20.1f}\n"
            )

            # Write oracle approach row
            f.write(
                f"{'':<15} {'Oracle Approach':<20} "
                f"{f'{oracle_goal_rate:.1f}%':<15} "
                f"{oracle_avg_evals:<20.1f} {oracle_avg_expansions:<20.1f}\n"
            )
            f.write("\n")

        f.write("=" * 100 + "\n\n")

        # Detailed results by maze
        f.write("DETAILED RESULTS BY MAZE\n")
        f.write("=" * 100 + "\n\n")

        for maze_name in sorted(results_by_margin[cfg.outer_margins[0]].keys()):
            f.write(f"Maze: {maze_name}\n")
            f.write("-" * 100 + "\n")
            f.write(
                f"{'Outer Margin':<15} {'Approach':<20} {'Goal Reached':<15} "
                f"{'Num Evals':<20} {'Num Expansions':<20}\n"
            )
            f.write("-" * 100 + "\n")

            for outer_margin in cfg.outer_margins:
                search_res = results_by_margin[outer_margin][maze_name]["search"]
                oracle_res = results_by_margin[outer_margin][maze_name]["oracle"]

                f.write(
                    f"{outer_margin:<15} {'Search Approach':<20} "
                    f"{str(search_res['goal_reached']):<15} "
                    f"{search_res['num_evals']:<20} "
                    f"{search_res['num_expansions']:<20}\n"
                )
                f.write(
                    f"{'':<15} {'Oracle Approach':<20} "
                    f"{str(oracle_res['goal_reached']):<15} "
                    f"{oracle_res['num_evals']:<20} "
                    f"{oracle_res['num_expansions']:<20}\n"
                )

            f.write("\n")

    logging.info(f"\nResults written to: {output_file}")

    # Generate visualizations
    logging.info("Generating visualizations...")
    plot_dir = Path(cfg.output_file).parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data for plotting
    outer_margins_list = list(cfg.outer_margins)
    search_avg_expansions_list = []
    search_avg_evals_list = []
    oracle_avg_expansions_list = []
    oracle_avg_evals_list = []

    for outer_margin in outer_margins_list:
        margin_results = results_by_margin[outer_margin]
        search_results_margin = [
            maze_data["search"] for maze_data in margin_results.values()
        ]
        oracle_results_margin = [
            maze_data["oracle"] for maze_data in margin_results.values()
        ]

        search_avg_expansions_list.append(
            np.mean([r["num_expansions"] for r in search_results_margin])
        )
        search_avg_evals_list.append(
            np.mean([r["num_evals"] for r in search_results_margin])
        )
        oracle_avg_expansions_list.append(
            np.mean([r["num_expansions"] for r in oracle_results_margin])
        )
        oracle_avg_evals_list.append(
            np.mean([r["num_evals"] for r in oracle_results_margin])
        )

    # Plot 1: Line plot for expansions vs outer margin
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(
        outer_margins_list,
        search_avg_expansions_list,
        marker="o",
        linewidth=2,
        markersize=8,
        label="Search Approach",
        color="steelblue",
    )
    ax.plot(
        outer_margins_list,
        oracle_avg_expansions_list,
        marker="s",
        linewidth=2,
        markersize=8,
        label="Oracle Approach",
        color="coral",
    )

    ax.set_xlabel("Outer Margin", fontsize=13)
    ax.set_ylabel("Average Number of Expansions", fontsize=13)
    ax.set_title("Node Expansions vs Outer Margin Size", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(outer_margins_list)

    # Add value annotations above each point with offset
    y_range = max(search_avg_expansions_list) - min(oracle_avg_expansions_list)
    offset = y_range * 0.03  # 3% of y-range for offset

    for i, margin in enumerate(outer_margins_list):
        # Search approach values (above points with offset)
        ax.text(
            margin,
            search_avg_expansions_list[i] + offset,
            f"{int(search_avg_expansions_list[i])}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="steelblue",
            fontweight="bold",
        )
        # Oracle approach values (below points with offset)
        ax.text(
            margin,
            oracle_avg_expansions_list[i] - offset,
            f"{int(oracle_avg_expansions_list[i])}",
            ha="center",
            va="top",
            fontsize=9,
            color="coral",
            fontweight="bold",
        )

    plt.tight_layout()
    expansions_line_plot = plot_dir / "outer_margin_expansions.png"
    plt.savefig(expansions_line_plot, dpi=300, bbox_inches="tight")
    logging.info(f"Saved expansions line plot to: {expansions_line_plot}")
    plt.close()

    # Plot 2: Line plot for evaluations vs outer margin
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(
        outer_margins_list,
        search_avg_evals_list,
        marker="o",
        linewidth=2,
        markersize=8,
        label="Search Approach",
        color="steelblue",
    )
    ax.plot(
        outer_margins_list,
        oracle_avg_evals_list,
        marker="s",
        linewidth=2,
        markersize=8,
        label="Oracle Approach",
        color="coral",
    )

    ax.set_xlabel("Outer Margin", fontsize=13)
    ax.set_ylabel("Average Number of Evaluations", fontsize=13)
    ax.set_title("State Evaluations vs Outer Margin Size", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(outer_margins_list)

    # Add value annotations above each point with offset
    y_range = max(search_avg_evals_list) - min(oracle_avg_evals_list)
    offset = y_range * 0.03  # 3% of y-range for offset

    for i, margin in enumerate(outer_margins_list):
        # Search approach values (above points with offset)
        ax.text(
            margin,
            search_avg_evals_list[i] + offset,
            f"{int(search_avg_evals_list[i])}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="steelblue",
            fontweight="bold",
        )
        # Oracle approach values (below points with offset)
        ax.text(
            margin,
            oracle_avg_evals_list[i] - offset,
            f"{int(oracle_avg_evals_list[i])}",
            ha="center",
            va="top",
            fontsize=9,
            color="coral",
            fontweight="bold",
        )

    plt.tight_layout()
    evals_line_plot = plot_dir / "outer_margin_evaluations.png"
    plt.savefig(evals_line_plot, dpi=300, bbox_inches="tight")
    logging.info(f"Saved evaluations line plot to: {evals_line_plot}")
    plt.close()

    # Plot 3: Combined subplot with both metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Expansions
    ax1.plot(
        outer_margins_list,
        search_avg_expansions_list,
        marker="o",
        linewidth=2,
        markersize=8,
        label="Search Approach",
        color="steelblue",
    )
    ax1.plot(
        outer_margins_list,
        oracle_avg_expansions_list,
        marker="s",
        linewidth=2,
        markersize=8,
        label="Oracle Approach",
        color="coral",
    )
    ax1.set_xlabel("Outer Margin", fontsize=12)
    ax1.set_ylabel("Average Number of Expansions", fontsize=12)
    ax1.set_title("Node Expansions", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(outer_margins_list)

    # Add value annotations for expansions
    y_range1 = max(search_avg_expansions_list) - min(oracle_avg_expansions_list)
    offset1 = y_range1 * 0.03

    for i, margin in enumerate(outer_margins_list):
        ax1.text(
            margin,
            search_avg_expansions_list[i] + offset1,
            f"{int(search_avg_expansions_list[i])}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="steelblue",
            fontweight="bold",
        )
        ax1.text(
            margin,
            oracle_avg_expansions_list[i] - offset1,
            f"{int(oracle_avg_expansions_list[i])}",
            ha="center",
            va="top",
            fontsize=8,
            color="coral",
            fontweight="bold",
        )

    # Evaluations
    ax2.plot(
        outer_margins_list,
        search_avg_evals_list,
        marker="o",
        linewidth=2,
        markersize=8,
        label="Search Approach",
        color="steelblue",
    )
    ax2.plot(
        outer_margins_list,
        oracle_avg_evals_list,
        marker="s",
        linewidth=2,
        markersize=8,
        label="Oracle Approach",
        color="coral",
    )
    ax2.set_xlabel("Outer Margin", fontsize=12)
    ax2.set_ylabel("Average Number of Evaluations", fontsize=12)
    ax2.set_title("State Evaluations", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(outer_margins_list)

    # Add value annotations for evaluations
    y_range2 = max(search_avg_evals_list) - min(oracle_avg_evals_list)
    offset2 = y_range2 * 0.03

    for i, margin in enumerate(outer_margins_list):
        ax2.text(
            margin,
            search_avg_evals_list[i] + offset2,
            f"{int(search_avg_evals_list[i])}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="steelblue",
            fontweight="bold",
        )
        ax2.text(
            margin,
            oracle_avg_evals_list[i] - offset2,
            f"{int(oracle_avg_evals_list[i])}",
            ha="center",
            va="top",
            fontsize=8,
            color="coral",
            fontweight="bold",
        )

    fig.suptitle("Search Metrics vs Outer Margin Size", fontsize=16)
    plt.tight_layout()
    combined_line_plot = plot_dir / "outer_margin_combined.png"
    plt.savefig(combined_line_plot, dpi=300, bbox_inches="tight")
    logging.info(f"Saved combined line plot to: {combined_line_plot}")
    plt.close()

    logging.info("All visualizations generated successfully!")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
