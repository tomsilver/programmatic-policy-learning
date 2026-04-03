"""Plot results for Search vs Oracle comparison and outer margin sweep.

Reads comparison_results.txt and outer_margin_results.txt from maze_results/
and generates plots without re-running experiments.

Usage::

    python experiments/maze_experiments/plot_comparison_results.py
"""

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Global style (matches Motion2D plots)
# ---------------------------------------------------------------------------
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

RESULTS_DIR = Path("experiments/maze_experiments/maze_results")
PLOTS_DIR = Path("experiments/maze_experiments/maze_results/graphs")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

_COLORS = {
    "search": "steelblue",
    "oracle": "coral",
}


def _pretty_maze_name(raw: str) -> str:
    """Convert 'maze1_10x10' to 'M1 (10x10)' with proper multiply sign."""
    m = re.match(r"maze(\d+)_(\d+)x(\d+)", raw)
    if m:
        return f"M{m.group(1)} ({m.group(2)}\u00d7{m.group(3)})"
    return raw


# ---------------------------------------------------------------------------
# Parsing — comparison_results.txt
# ---------------------------------------------------------------------------


def _parse_comparison_results() -> (
    tuple[list[str], list[dict[str, int]], list[dict[str, int]]]
):
    """Parse comparison_results.txt.

    Returns (maze_names, search_results, oracle_results) where each result
    dict has keys: num_evals, num_expansions.
    """
    path = RESULTS_DIR / "comparison_results.txt"
    text = path.read_text(encoding="utf-8")

    maze_names: list[str] = []
    search_results: list[dict[str, int]] = []
    oracle_results: list[dict[str, int]] = []

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        maze_match = re.match(r"^Maze:\s+(maze\S+)", line)
        if maze_match:
            maze_names.append(maze_match.group(1))
            # Skip header lines, then read Search and Oracle rows
            i += 4  # skip "---", header, "---"
            search_m = re.match(
                r"Search Approach\s+(True|False)\s+(\d+)\s+(\d+)", lines[i].strip()
            )
            if search_m:
                search_results.append(
                    {
                        "num_evals": int(search_m.group(2)),
                        "num_expansions": int(search_m.group(3)),
                    }
                )
            i += 1
            oracle_m = re.match(
                r"Oracle Approach\s+(True|False)\s+(\d+)\s+(\d+)", lines[i].strip()
            )
            if oracle_m:
                oracle_results.append(
                    {
                        "num_evals": int(oracle_m.group(2)),
                        "num_expansions": int(oracle_m.group(3)),
                    }
                )
        i += 1

    return maze_names, search_results, oracle_results


# ---------------------------------------------------------------------------
# Parsing — outer_margin_results.txt
# ---------------------------------------------------------------------------


def _parse_outer_margin_results() -> (
    tuple[list[int], list[float], list[float], list[float], list[float]]
):
    """Parse outer_margin_results.txt summary table.

    Returns (margins, search_avg_exp, oracle_avg_exp, search_avg_eval,
    oracle_avg_eval).
    """
    path = RESULTS_DIR / "outer_margin_results.txt"
    text = path.read_text(encoding="utf-8")

    margins: list[int] = []
    search_exp: list[float] = []
    oracle_exp: list[float] = []
    search_eval: list[float] = []
    oracle_eval: list[float] = []

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Search row: "10              Search Approach      100.0%  607.7  567.9"
        search_m = re.match(
            r"^(\d+)\s+Search Approach\s+[\d.]+%\s+([\d.]+)\s+([\d.]+)", line
        )
        if search_m:
            margins.append(int(search_m.group(1)))
            search_eval.append(float(search_m.group(2)))
            search_exp.append(float(search_m.group(3)))
            # Next line is Oracle for same margin
            i += 1
            oracle_m = re.match(
                r"\s*Oracle Approach\s+[\d.]+%\s+([\d.]+)\s+([\d.]+)",
                lines[i].strip(),
            )
            if oracle_m:
                oracle_eval.append(float(oracle_m.group(1)))
                oracle_exp.append(float(oracle_m.group(2)))
        i += 1

    return margins, search_exp, oracle_exp, search_eval, oracle_eval


# ---------------------------------------------------------------------------
# Plot 1 — Search vs Oracle expansions (bar chart)
# ---------------------------------------------------------------------------


def plot_expansions_comparison(
    maze_names: list[str],
    search_results: list[dict[str, int]],
    oracle_results: list[dict[str, int]],
) -> None:
    """Grouped bar chart: Search vs Oracle expansions per maze."""
    search_exp = [r["num_expansions"] for r in search_results]
    oracle_exp = [r["num_expansions"] for r in oracle_results]

    pretty = [_pretty_maze_name(m) for m in maze_names]
    labels = pretty + ["Average"]
    search_plot = search_exp + [float(np.mean(search_exp))]
    oracle_plot = oracle_exp + [float(np.mean(oracle_exp))]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(labels))
    width = 0.35

    bars_s = ax.bar(
        x - width / 2,
        search_plot,
        width,
        label="Pure Planning (A* Search)",
        color=_COLORS["search"],
        edgecolor="white",
    )
    bars_o = ax.bar(
        x + width / 2,
        oracle_plot,
        width,
        label="Oracle (Human Expert)",
        color=_COLORS["oracle"],
        edgecolor="white",
    )

    top = max(*search_plot, *oracle_plot)
    ax.set_ylim(0, top * 1.2)

    for bar_group in [bars_s, bars_o]:
        for rect in bar_group:
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                height + top * 0.01,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Maze")
    ax.set_ylabel("Number of Expansions")
    ax.set_title("Maze Environment \u2014 Node Expansions by Approach")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    fig.tight_layout()
    out = PLOTS_DIR / "expansions_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2 — Search vs Oracle evaluations (bar chart)
# ---------------------------------------------------------------------------


def plot_evaluations_comparison(
    maze_names: list[str],
    search_results: list[dict[str, int]],
    oracle_results: list[dict[str, int]],
) -> None:
    """Grouped bar chart: Search vs Oracle evaluations per maze."""
    search_eval = [r["num_evals"] for r in search_results]
    oracle_eval = [r["num_evals"] for r in oracle_results]

    pretty = [_pretty_maze_name(m) for m in maze_names]
    labels = pretty + ["Average"]
    search_plot = search_eval + [float(np.mean(search_eval))]
    oracle_plot = oracle_eval + [float(np.mean(oracle_eval))]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(labels))
    width = 0.35

    bars_s = ax.bar(
        x - width / 2,
        search_plot,
        width,
        label="Pure Planning (A* Search)",
        color=_COLORS["search"],
        edgecolor="white",
    )
    bars_o = ax.bar(
        x + width / 2,
        oracle_plot,
        width,
        label="Oracle (Human Expert)",
        color=_COLORS["oracle"],
        edgecolor="white",
    )

    top = max(*search_plot, *oracle_plot)
    ax.set_ylim(0, top * 1.2)

    for bar_group in [bars_s, bars_o]:
        for rect in bar_group:
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                height + top * 0.01,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Maze")
    ax.set_ylabel("Number of Evaluations")
    ax.set_title("Maze Environment \u2014 State Evaluations by Approach")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    fig.tight_layout()
    out = PLOTS_DIR / "evaluations_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3 — Combined side-by-side (expansions + evaluations)
# ---------------------------------------------------------------------------


def plot_combined_comparison(
    maze_names: list[str],
    search_results: list[dict[str, int]],
    oracle_results: list[dict[str, int]],
) -> None:
    """Side-by-side: expansions and evaluations."""
    pretty = [_pretty_maze_name(m) for m in maze_names]
    labels = pretty + ["Avg"]
    width = 0.35
    x = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    for ax, key, title in [
        (ax1, "num_expansions", "Node Expansions"),
        (ax2, "num_evals", "State Evaluations"),
    ]:
        s_vals = [r[key] for r in search_results]
        o_vals = [r[key] for r in oracle_results]
        s_plot = s_vals + [float(np.mean(s_vals))]
        o_plot = o_vals + [float(np.mean(o_vals))]

        ax.bar(
            x - width / 2,
            s_plot,
            width,
            label="Pure Planning",
            color=_COLORS["search"],
            edgecolor="white",
        )
        ax.bar(
            x + width / 2,
            o_plot,
            width,
            label="Oracle",
            color=_COLORS["oracle"],
            edgecolor="white",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Maze")
        ax.set_ylabel(f"Number of {title.split()[1]}")
        ax.set_title(title)
        ax.legend(framealpha=0.9)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

    fig.tight_layout()
    out = PLOTS_DIR / "combined_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 4 — Outer margin: expansions line plot
# ---------------------------------------------------------------------------


def plot_outer_margin_expansions(
    margins: list[int],
    search_exp: list[float],
    oracle_exp: list[float],
) -> None:
    """Line plot: avg expansions vs outer margin."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        margins,
        search_exp,
        marker="o",
        linewidth=2,
        markersize=6,
        label="Pure Planning (A* Search)",
        color=_COLORS["search"],
    )
    ax.plot(
        margins,
        oracle_exp,
        marker="s",
        linewidth=2,
        markersize=6,
        label="Oracle (Human Expert)",
        color=_COLORS["oracle"],
    )

    ax.set_xlabel("Outer Margin")
    ax.set_ylabel("Avg Number of Expansions")
    ax.set_title("Maze Environment \u2014 Node Expansions vs Outer Margin Size")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_xticks(margins)

    fig.tight_layout()
    out = PLOTS_DIR / "outer_margin_expansions.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 5 — Outer margin: evaluations line plot
# ---------------------------------------------------------------------------


def plot_outer_margin_evaluations(
    margins: list[int],
    search_eval: list[float],
    oracle_eval: list[float],
) -> None:
    """Line plot: avg evaluations vs outer margin."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        margins,
        search_eval,
        marker="o",
        linewidth=2,
        markersize=6,
        label="Pure Planning (A* Search)",
        color=_COLORS["search"],
    )
    ax.plot(
        margins,
        oracle_eval,
        marker="s",
        linewidth=2,
        markersize=6,
        label="Oracle (Human Expert)",
        color=_COLORS["oracle"],
    )

    ax.set_xlabel("Outer Margin")
    ax.set_ylabel("Avg Number of Evaluations")
    ax.set_title("Maze Environment \u2014 State Evaluations vs Outer Margin Size")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_xticks(margins)

    fig.tight_layout()
    out = PLOTS_DIR / "outer_margin_evaluations.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 6 — Outer margin: combined side-by-side
# ---------------------------------------------------------------------------


def plot_outer_margin_combined(
    margins: list[int],
    search_exp: list[float],
    oracle_exp: list[float],
    search_eval: list[float],
    oracle_eval: list[float],
) -> None:
    """Side-by-side line plots: expansions and evaluations vs margin."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, s_vals, o_vals, ylabel, title in [
        (ax1, search_exp, oracle_exp, "Avg Expansions", "Node Expansions"),
        (ax2, search_eval, oracle_eval, "Avg Evaluations", "State Evaluations"),
    ]:
        ax.plot(
            margins,
            s_vals,
            marker="o",
            linewidth=2,
            markersize=6,
            label="Pure Planning",
            color=_COLORS["search"],
        )
        ax.plot(
            margins,
            o_vals,
            marker="s",
            linewidth=2,
            markersize=6,
            label="Oracle",
            color=_COLORS["oracle"],
        )
        ax.set_xlabel("Outer Margin")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(framealpha=0.9)
        ax.grid(alpha=0.25, linestyle="--")
        ax.set_xticks(margins)

    fig.tight_layout()
    out = PLOTS_DIR / "outer_margin_combined.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Load results from text files and generate all plots."""
    # Comparison plots (Search vs Oracle)
    comp_path = RESULTS_DIR / "comparison_results.txt"
    if comp_path.exists():
        maze_names, search_res, oracle_res = _parse_comparison_results()
        plot_expansions_comparison(maze_names, search_res, oracle_res)
        plot_evaluations_comparison(maze_names, search_res, oracle_res)
        plot_combined_comparison(maze_names, search_res, oracle_res)
    else:
        print(f"Skipping comparison plots — {comp_path} not found")

    # Outer margin plots
    margin_path = RESULTS_DIR / "outer_margin_results.txt"
    if margin_path.exists():
        margins, s_exp, o_exp, s_eval, o_eval = _parse_outer_margin_results()
        plot_outer_margin_expansions(margins, s_exp, o_exp)
        plot_outer_margin_evaluations(margins, s_eval, o_eval)
        plot_outer_margin_combined(margins, s_exp, o_exp, s_eval, o_eval)
    else:
        print(f"Skipping outer margin plots — {margin_path} not found")

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
