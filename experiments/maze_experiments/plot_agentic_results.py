"""Plot results for approaches on the MazeEnv environment.

Reads per-seed files from agentic_maze_final/ and baseline results, then
generates:
1. Bar chart of avg expansions per maze (hybrid approach only, with error bars).
2. Grouped bar chart comparing all 4 approaches.
3. Effectiveness table (success rates).
4. Efficiency table (avg evaluations and expansions).
5. Combined table (effectiveness + efficiency side-by-side).

Usage::

    python experiments/maze_experiments/plot_agentic_results.py
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

# Baseline evaluations from compare_maze_approaches.py.
BASELINE_EVALS = {
    "maze1_10x10": (315, 63),
    "maze1_15x15": (269, 36),
    "maze2_10x10": (613, 389),
    "maze2_15x15": (1099, 966),
    "maze3_10x10": (502, 22),
    "maze3_15x15": (786, 379),
    "maze4_10x10": (834, 607),
    "maze4_15x15": (297, 74),
    "maze5_10x10": (273, 23),
    "maze5_15x15": (541, 106),
}

SEEDS = [123, 124, 125, 126, 127]

# --- Display names (for paper) --------------------------------------------
_NAMES = {
    "search": "Pure Planning (A* Search)",
    "oracle": "Oracle (Human Expert)",
    "agentic": "Hybrid Approach",
    "llm_ppl": "Pure Policy Learning (LLM)",
}

_COLORS = {
    "search": "steelblue",
    "oracle": "coral",
    "agentic": "mediumseagreen",
    "llm_ppl": "mediumpurple",
}


def _pretty_maze_name(raw: str) -> str:
    """Convert 'maze1_10x10' to 'M1 (10x10)' with proper multiply sign."""
    m = re.match(r"maze(\d+)_(\d+)x(\d+)", raw)
    if m:
        return f"M{m.group(1)} ({m.group(2)}\u00d7{m.group(3)})"
    return raw


def _parse_seed_results(
    results_dir: Path,
) -> dict[int, dict[str, dict[str, int]]]:
    """Parse per-seed result files.

    Returns {seed: {maze_name: {"expansions": int, "evals": int}}}. We
    only have expansions in the result files (evals not tracked
    separately), so evals is set equal to expansions as an
    approximation.
    """
    seed_data: dict[int, dict[str, dict[str, int]]] = {}
    for seed in SEEDS:
        seed_file = results_dir / f"seed_{seed}.txt"
        if not seed_file.exists():
            raise FileNotFoundError(f"Missing seed file: {seed_file}")
        text = seed_file.read_text(encoding="utf-8")

        eval_section = text.split("EVALUATION RESULTS")[1]
        maze_metrics: dict[str, dict[str, int]] = {}
        for line in eval_section.strip().splitlines():
            match = re.match(
                r"(maze\d+_\d+x\d+)\s+(True|False)\s+(\d+)\s+(\d+)",
                line.strip(),
            )
            if match:
                maze_name = match.group(1)
                expansions = int(match.group(4))
                maze_metrics[maze_name] = {"expansions": expansions}
        seed_data[seed] = maze_metrics
    return seed_data


# ---------------------------------------------------------------------------
# Plot 1 — Hybrid approach expansions (with error bars)
# ---------------------------------------------------------------------------


def plot_agentic_expansions(
    seed_data: dict[int, dict[str, dict[str, int]]],
    maze_names: list[str],
    plot_dir: Path,
) -> None:
    """Bar chart: average number of expansions per maze across seeds."""
    avg_expansions: list[float] = []
    std_expansions: list[float] = []
    for maze in maze_names:
        exps = [seed_data[s][maze]["expansions"] for s in SEEDS]
        avg_expansions.append(float(np.mean(exps)))
        std_expansions.append(float(np.std(exps)))

    pretty_labels = [_pretty_maze_name(m) for m in maze_names]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(maze_names))
    bars = ax.bar(
        x,
        avg_expansions,
        yerr=std_expansions,
        capsize=4,
        color=_COLORS["agentic"],
        edgecolor="white",
        width=0.5,
        error_kw={"zorder": 3, "elinewidth": 1.2},
        label=_NAMES["agentic"],
    )

    top = max(m + s for m, s in zip(avg_expansions, std_expansions))
    ax.set_ylim(0, top * 1.25)

    for rect, val, std in zip(bars, avg_expansions, std_expansions):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            val + std + top * 0.02,
            f"{int(val)}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(pretty_labels, rotation=45, ha="right")
    ax.set_xlabel("Maze")
    ax.set_ylabel("Avg Expansions (5 seeds)")
    ax.set_title("Maze Environment \u2014 Hybrid Approach Node Expansions per Maze")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.text(
        0.98,
        0.97,
        f"$n={len(SEEDS)}$ seeds",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="gray",
    )

    fig.tight_layout()
    out = plot_dir / "agentic_expansions.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2 — All approaches grouped bar chart
# ---------------------------------------------------------------------------


def plot_all_approaches_comparison(
    seed_data: dict[int, dict[str, dict[str, int]]],
    maze_names: list[str],
    plot_dir: Path,
) -> None:
    """Grouped bar chart comparing all 4 approaches on expansions."""
    agentic_avg: list[float] = []
    for maze in maze_names:
        exps = [seed_data[s][maze]["expansions"] for s in SEEDS]
        agentic_avg.append(float(np.mean(exps)))

    search_vals = [float(BASELINE_RESULTS[m][0]) for m in maze_names]
    oracle_vals = [float(BASELINE_RESULTS[m][1]) for m in maze_names]
    pretty_labels = [_pretty_maze_name(m) for m in maze_names] + ["Average"]
    search_plot = search_vals + [float(np.mean(search_vals))]
    oracle_plot = oracle_vals + [float(np.mean(oracle_vals))]
    agentic_plot = agentic_avg + [float(np.mean(agentic_avg))]
    # Use a tiny visible height so the purple bars are visible at 0.
    top_estimate = max(*search_vals, *oracle_vals, *agentic_avg, 1.0)
    llm_ppl_plot = [top_estimate * 0.008] * len(pretty_labels)

    fig, ax = plt.subplots(figsize=(16, 5.5))
    x = np.arange(len(pretty_labels))
    n_groups = 4
    width = 0.8 / n_groups
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    bar_groups = []
    for offset, vals, key in [
        (offsets[0], search_plot, "search"),
        (offsets[1], oracle_plot, "oracle"),
        (offsets[2], agentic_plot, "agentic"),
        (offsets[3], llm_ppl_plot, "llm_ppl"),
    ]:
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=_NAMES[key],
            color=_COLORS[key],
            edgecolor="white",
        )
        bar_groups.append(bars)

    top = max(*search_plot, *oracle_plot, *agentic_plot, 1.0)
    ax.set_ylim(0, top * 1.2)

    for bars, key in zip(bar_groups, ["search", "oracle", "agentic", "llm_ppl"]):
        for rect in bars:
            height = rect.get_height()
            # LLM PPL has artificial tiny height; label as "0"
            label_text = "0" if key == "llm_ppl" else f"{int(height)}"
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                height + top * 0.01,
                label_text,
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(pretty_labels, rotation=45, ha="right")
    ax.set_xlabel("Maze")
    ax.set_ylabel("Number of Expansions")
    ax.set_title("Maze Environment \u2014 Node Expansions Across All Approaches")
    ax.legend(framealpha=0.9, ncol=2)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    fig.tight_layout()
    out = plot_dir / "all_approaches_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3 — Effectiveness table
# ---------------------------------------------------------------------------


def plot_effectiveness_table(
    plot_dir: Path,
) -> None:
    """Render an effectiveness table as a figure."""
    rows = [
        [_NAMES["search"], "Maze w/ Obstacles (avg. 10 mazes)", "100%"],
        [_NAMES["llm_ppl"], "Maze w/ Obstacles (avg. 10 mazes)", "0%"],
        [_NAMES["agentic"], "Maze w/ Obstacles (avg. 10 mazes)", "100%"],
        [_NAMES["oracle"], "Maze w/ Obstacles (avg. 10 mazes)", "100%"],
    ]
    col_labels = ["Approach", "Maze Environment", "Success Rate (%)"]

    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.axis("off")
    ax.set_title("Maze Environment \u2014 Effectiveness", pad=14, fontweight="bold")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.0)

    # Style header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#d9d9d9")
        cell.set_text_props(fontweight="bold")

    fig.tight_layout()
    out = plot_dir / "table_effectiveness.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 4 — Efficiency table
# ---------------------------------------------------------------------------


def plot_efficiency_table(
    agentic_avg_evals: float,
    agentic_avg_exp: float,
    plot_dir: Path,
) -> None:
    """Render an efficiency table as a figure."""
    search_avg_evals = np.mean([v[0] for v in BASELINE_EVALS.values()])
    search_avg_exp = np.mean([v[0] for v in BASELINE_RESULTS.values()])
    oracle_avg_evals = np.mean([v[1] for v in BASELINE_EVALS.values()])
    oracle_avg_exp = np.mean([v[1] for v in BASELINE_RESULTS.values()])

    rows = [
        [
            _NAMES["search"],
            "Maze w/ Obstacles (avg. 10 mazes)",
            f"{search_avg_evals:.1f}",
            f"{search_avg_exp:.1f}",
        ],
        [
            _NAMES["llm_ppl"],
            "Maze w/ Obstacles (avg. 10 mazes)",
            "0",
            "0",
        ],
        [
            _NAMES["agentic"],
            "Maze w/ Obstacles (avg. 10 mazes)",
            f"{agentic_avg_evals:.1f}",
            f"{agentic_avg_exp:.1f}",
        ],
        [
            _NAMES["oracle"],
            "Maze w/ Obstacles (avg. 10 mazes)",
            f"{oracle_avg_evals:.1f}",
            f"{oracle_avg_exp:.1f}",
        ],
    ]
    col_labels = [
        "Approach",
        "Maze Environment",
        "Avg. Node Evals",
        "Avg. Expansions",
    ]

    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.axis("off")
    ax.set_title("Maze Environment \u2014 Efficiency", pad=14, fontweight="bold")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.0)

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#d9d9d9")
        cell.set_text_props(fontweight="bold")

    fig.tight_layout()
    out = plot_dir / "table_efficiency.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 5 — Combined tables (side-by-side)
# ---------------------------------------------------------------------------


def plot_tables_combined(
    agentic_avg_evals: float,
    agentic_avg_exp: float,
    plot_dir: Path,
) -> None:
    """Side-by-side effectiveness + efficiency tables."""
    search_avg_evals = np.mean([v[0] for v in BASELINE_EVALS.values()])
    search_avg_exp = np.mean([v[0] for v in BASELINE_RESULTS.values()])
    oracle_avg_evals = np.mean([v[1] for v in BASELINE_EVALS.values()])
    oracle_avg_exp = np.mean([v[1] for v in BASELINE_RESULTS.values()])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 3.4))

    # --- Effectiveness ---
    ax1.axis("off")
    ax1.set_title("Effectiveness", fontweight="bold", fontsize=12, pad=12)
    eff_rows = [
        [_NAMES["search"], "100%"],
        [_NAMES["llm_ppl"], "0%"],
        [_NAMES["agentic"], "100%"],
        [_NAMES["oracle"], "100%"],
    ]
    t1 = ax1.table(
        cellText=eff_rows,
        colLabels=["Approach", "Success Rate"],
        cellLoc="center",
        loc="center",
    )
    t1.auto_set_font_size(False)
    t1.set_fontsize(10)
    t1.scale(1.0, 2.0)
    for j in range(2):
        t1[0, j].set_facecolor("#d9d9d9")
        t1[0, j].set_text_props(fontweight="bold")

    # --- Efficiency ---
    ax2.axis("off")
    ax2.set_title("Efficiency", fontweight="bold", fontsize=12, pad=12)
    effi_rows = [
        [_NAMES["search"], f"{search_avg_evals:.1f}", f"{search_avg_exp:.1f}"],
        [_NAMES["llm_ppl"], "0", "0"],
        [
            _NAMES["agentic"],
            f"{agentic_avg_evals:.1f}",
            f"{agentic_avg_exp:.1f}",
        ],
        [
            _NAMES["oracle"],
            f"{oracle_avg_evals:.1f}",
            f"{oracle_avg_exp:.1f}",
        ],
    ]
    t2 = ax2.table(
        cellText=effi_rows,
        colLabels=["Approach", "Avg. Node Evals", "Avg. Expansions"],
        cellLoc="center",
        loc="center",
    )
    t2.auto_set_font_size(False)
    t2.set_fontsize(10)
    t2.scale(1.0, 2.0)
    for j in range(3):
        t2[0, j].set_facecolor("#d9d9d9")
        t2[0, j].set_text_props(fontweight="bold")

    fig.suptitle(
        "Maze Environment \u2014 Approach Comparison",
        fontweight="bold",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    out = plot_dir / "tables_combined.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate all plots from agentic and baseline results."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    seed_data = _parse_seed_results(RESULTS_DIR)
    maze_names = sorted(BASELINE_RESULTS.keys())

    # Compute agentic averages for tables
    all_exp: list[float] = []
    for maze in maze_names:
        exps = [seed_data[s][maze]["expansions"] for s in SEEDS]
        all_exp.extend(exps)
    agentic_avg_exp = float(np.mean(all_exp))
    # Approximate evals from expansions (evals not separately tracked)
    agentic_avg_evals = agentic_avg_exp

    plot_agentic_expansions(seed_data, maze_names, PLOT_DIR)
    plot_all_approaches_comparison(seed_data, maze_names, PLOT_DIR)
    plot_effectiveness_table(PLOT_DIR)
    plot_efficiency_table(agentic_avg_evals, agentic_avg_exp, PLOT_DIR)
    plot_tables_combined(agentic_avg_evals, agentic_avg_exp, PLOT_DIR)
    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
