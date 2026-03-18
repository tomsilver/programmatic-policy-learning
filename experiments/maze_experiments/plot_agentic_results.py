"""Plot results for the Agentic Integrated Approach.

Reads the aggregate_results.txt file and generates:
1. Bar chart of avg expansions per maze (agentic only, with std-dev error bars).
2. Grouped bar chart comparing Search, Oracle, and Agentic Integrated approaches.

Usage:
    python experiments/maze_experiments/plot_agentic_results.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Data paths ────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("experiments/maze_experiments/maze_results/agentic_maze_final")
PLOT_DIR = Path("experiments/maze_experiments/maze_results/graphs")

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

SEEDS = [123, 124, 125, 126, 127]


def _parse_seed_results(results_dir: Path) -> dict[int, dict[str, int]]:
    """Parse per-seed result files.

    Returns {seed: {maze_name: expansions}}.
    """
    seed_data: dict[int, dict[str, int]] = {}
    for seed in SEEDS:
        seed_file = results_dir / f"seed_{seed}.txt"
        if not seed_file.exists():
            raise FileNotFoundError(f"Missing seed file: {seed_file}")
        text = seed_file.read_text(encoding="utf-8")

        # Find EVALUATION RESULTS section
        eval_section = text.split("EVALUATION RESULTS")[1]
        maze_expansions: dict[str, int] = {}
        for line in eval_section.strip().splitlines():
            # Lines look like: maze1_10x10      True      35      64
            match = re.match(
                r"(maze\d+_\d+x\d+)\s+(True|False)\s+(\d+)\s+(\d+)", line.strip()
            )
            if match:
                maze_name = match.group(1)
                expansions = int(match.group(4))
                maze_expansions[maze_name] = expansions
        seed_data[seed] = maze_expansions
    return seed_data


def plot_agentic_expansions(
    seed_data: dict[int, dict[str, int]],
    maze_names: list[str],
    plot_dir: Path,
) -> None:
    """Bar chart: average number of expansions per maze across seeds."""
    avg_expansions: list[float] = []
    std_expansions: list[float] = []
    for maze in maze_names:
        exps = [seed_data[s][maze] for s in SEEDS]
        avg_expansions.append(float(np.mean(exps)))
        std_expansions.append(float(np.std(exps)))

    _, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(maze_names))
    bars = ax.bar(
        x,
        avg_expansions,
        yerr=std_expansions,
        capsize=4,
        color="mediumseagreen",
        edgecolor="white",
        label="Agentic Integrated",
    )

    # Value labels — placed above the error bar cap
    for rect, val, std in zip(bars, avg_expansions, std_expansions):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            val + std + 5,
            f"{int(val)}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Maze", fontsize=12)
    ax.set_ylabel("Avg Expansions (5 seeds)", fontsize=12)
    ax.set_title("Agentic Integrated Approach — Node Expansions per Maze", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(maze_names, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = plot_dir / "agentic_expansions.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_all_approaches_comparison(
    seed_data: dict[int, dict[str, int]],
    maze_names: list[str],
    plot_dir: Path,
) -> None:
    """Grouped bar chart comparing Search, Oracle, and Agentic Integrated."""
    # Per-maze averages for integrated approach
    integrated_avg: list[float] = []
    for maze in maze_names:
        exps = [seed_data[s][maze] for s in SEEDS]
        integrated_avg.append(float(np.mean(exps)))

    search_vals: list[float] = [float(BASELINE_RESULTS[m][0]) for m in maze_names]
    oracle_vals: list[float] = [float(BASELINE_RESULTS[m][1]) for m in maze_names]

    # Append overall averages
    labels = maze_names + ["Average"]
    search_vals_plot = search_vals + [float(np.mean(search_vals))]
    oracle_vals_plot = oracle_vals + [float(np.mean(oracle_vals))]
    integrated_plot = integrated_avg + [float(np.mean(integrated_avg))]

    _, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(labels))
    width = 0.25

    bars_search = ax.bar(
        x - width, search_vals_plot, width, label="Search (A*)", color="steelblue"
    )
    bars_oracle = ax.bar(
        x, oracle_vals_plot, width, label="Oracle (Expert)", color="coral"
    )
    bars_integrated = ax.bar(
        x + width,
        integrated_plot,
        width,
        label="Agentic Integrated",
        color="mediumseagreen",
    )

    # Value labels
    for bar_group in [bars_search, bars_oracle, bars_integrated]:
        for rect in bar_group:
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xlabel("Maze", fontsize=12)
    ax.set_ylabel("Number of Expansions", fontsize=12)
    ax.set_title("Node Expansions — All Approaches Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = plot_dir / "all_approaches_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main() -> None:
    """Generate plots from agentic integrated approach results."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    seed_data = _parse_seed_results(RESULTS_DIR)
    maze_names = sorted(BASELINE_RESULTS.keys())

    plot_agentic_expansions(seed_data, maze_names, PLOT_DIR)
    plot_all_approaches_comparison(seed_data, maze_names, PLOT_DIR)


if __name__ == "__main__":
    main()
