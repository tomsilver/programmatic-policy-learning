"""Plot results for the Motion2D BiRRT approach.

Reads aggregate_results.txt and per-seed files from results/ and generates:
1. Success rate bar chart across passage counts.
2. Average steps bar chart (solved trials only) across passage counts.
3. BiRRT planning cost (collision checks and nodes extended) across passage counts.
4. Per-seed success heatmap.

Usage::

    python experiments/motion2d_experiments/plot_birrt_motion2d.py
"""

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Global style
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

RESULTS_DIR = Path("experiments/motion2d_experiments/results/birrt")
PLOTS_DIR = Path("experiments/motion2d_experiments/plots/birrt")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(10))
PASSAGES = [1, 3, 5, 7]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_seed_file(seed: int) -> list[dict]:
    """Parse one seed file into a list of result dicts."""
    path = RESULTS_DIR / f"seed_{seed}.txt"
    text = path.read_text(encoding="utf-8")
    results = []
    for line in text.splitlines():
        # e.g.: p1         True      87      87      1234         456
        m = re.match(
            r"p(\d+)\s+(True|False)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", line.strip()
        )
        if m:
            results.append(
                {
                    "passages": int(m.group(1)),
                    "goal_reached": m.group(2) == "True",
                    "total_steps": int(m.group(3)),
                    "plan_length": int(m.group(4)),
                    "num_collision_checks": int(m.group(5)),
                    "num_nodes_extended": int(m.group(6)),
                }
            )
    return results


def load_all_results() -> tuple[dict[int, list[dict]], list[int]]:
    """Load all per-seed result files and return results with inferred
    passages."""
    available_seeds = sorted(
        int(p.stem.split("_")[1]) for p in RESULTS_DIR.glob("seed_*.txt")
    )
    all_results = {seed: _parse_seed_file(seed) for seed in available_seeds}
    # Infer passage counts from what's actually in the data.
    passages = sorted({r["passages"] for rs in all_results.values() for r in rs})
    return all_results, passages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COLORS = ["steelblue", "mediumseagreen", "coral", "mediumpurple", "goldenrod", "teal"]


def _per_passage(
    all_results: dict[int, list[dict]],
    passages: list[int],
    key: str,
    only_solved: bool = False,
) -> tuple[list[float], list[float]]:
    """Return (means, stds) of *key* for each passage count across seeds."""
    means, stds = [], []
    flat = [r for rs in all_results.values() for r in rs]
    for p in passages:
        vals = [
            r[key]
            for r in flat
            if r["passages"] == p and (not only_solved or r["goal_reached"])
        ]
        means.append(float(np.mean(vals)) if vals else 0.0)
        stds.append(float(np.std(vals)) if vals else 0.0)
    return means, stds


def _success_rate(
    all_results: dict[int, list[dict]], passages: list[int]
) -> list[float]:
    flat = [r for rs in all_results.values() for r in rs]
    rates = []
    for p in passages:
        p_results = [r for r in flat if r["passages"] == p]
        rates.append(sum(r["goal_reached"] for r in p_results) / len(p_results) * 100)
    return rates


# ---------------------------------------------------------------------------
# Plot 1 — Success rate
# ---------------------------------------------------------------------------


def plot_success_rate(all_results: dict[int, list[dict]], passages: list[int]) -> None:
    """Plot success rate bar chart across passage counts."""
    rates = _success_rate(all_results, passages)
    labels = [str(p) for p in passages]
    colors = [_COLORS[i % len(_COLORS)] for i in range(len(passages))]
    fig, ax = plt.subplots(figsize=(max(5, len(passages) * 1.2), 4))
    bars = ax.bar(labels, rates, color=colors, edgecolor="white", width=0.5)
    for rect, val in zip(bars, rates):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            val + 1.5,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    n_seeds = len(all_results)
    ax.set_ylim(0, 118)
    ax.set_xlabel("Number of Passages")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("BiRRT Planning Success Rate on Motion2D Environment")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.text(
        0.98,
        0.97,
        f"$n={n_seeds}$ seeds",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="gray",
    )
    fig.tight_layout()
    out = PLOTS_DIR / "success_rate.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2 — Average steps (solved only)
# ---------------------------------------------------------------------------


def plot_avg_steps(all_results: dict[int, list[dict]], passages: list[int]) -> None:
    """Plot average steps bar chart (solved trials only) across passage
    counts."""
    means, stds = _per_passage(all_results, passages, "total_steps", only_solved=True)
    labels = [str(p) for p in passages]
    colors = [_COLORS[i % len(_COLORS)] for i in range(len(passages))]
    fig, ax = plt.subplots(figsize=(max(5, len(passages) * 1.2), 4))
    x = np.arange(len(passages))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=colors,
        edgecolor="white",
        width=0.5,
        error_kw={"zorder": 3, "elinewidth": 1.2},
    )
    top = max(m + s for m, s in zip(means, stds)) if means else 1
    ax.set_ylim(0, top * 1.25)
    for rect, val, std in zip(bars, means, stds):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            val + std + top * 0.03,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    n_seeds = len(all_results)
    ax.set_xlabel("Number of Passages")
    ax.set_ylabel("Steps to Goal (solved trials only)")
    ax.set_title("BiRRT Steps to Goal on Motion2D Environment")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.text(
        0.98,
        0.97,
        f"$n={n_seeds}$ seeds",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="gray",
    )
    fig.tight_layout()
    out = PLOTS_DIR / "avg_steps.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3 — Planning cost (collision checks + nodes extended)
# ---------------------------------------------------------------------------


def plot_planning_cost(all_results: dict[int, list[dict]], passages: list[int]) -> None:
    """Plot BiRRT planning cost (collision checks and nodes) across passage
    counts."""
    coll_means, coll_stds = _per_passage(all_results, passages, "num_collision_checks")
    node_means, node_stds = _per_passage(all_results, passages, "num_nodes_extended")
    labels = [str(p) for p in passages]

    fig, ax = plt.subplots(figsize=(max(6, len(passages) * 1.4), 4))
    x = np.arange(len(passages))
    width = 0.35

    err_kw = {"zorder": 3, "elinewidth": 1.2}
    bars1 = ax.bar(
        x - width / 2,
        coll_means,
        width,
        yerr=coll_stds,
        capsize=4,
        label="Collision Checks",
        color="steelblue",
        error_kw=err_kw,
    )
    bars2 = ax.bar(
        x + width / 2,
        node_means,
        width,
        yerr=node_stds,
        capsize=4,
        label="Nodes Extended",
        color="coral",
        error_kw=err_kw,
    )

    all_tops = [m + s for m, s in zip(coll_means, coll_stds)] + [
        m + s for m, s in zip(node_means, node_stds)
    ]
    top = max(all_tops) if all_tops else 1
    ax.set_ylim(0, top * 1.25)
    label_offset = top * 0.02

    for ms, ss, bars in [
        (coll_means, coll_stds, bars1),
        (node_means, node_stds, bars2),
    ]:
        for rect, val, std in zip(bars, ms, ss):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                val + std + label_offset,
                f"{val:,.0f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    n_seeds = len(all_results)
    ax.set_xlabel("Number of Passages")
    ax.set_ylabel("Count (mean $\\pm$ std across seeds)")
    ax.set_title("BiRRT Planning Cost on Motion2D Environment")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.text(
        0.98,
        0.97,
        f"$n={n_seeds}$ seeds",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="gray",
    )
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.tight_layout()
    out = PLOTS_DIR / "planning_cost.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 4 — Per-seed success heatmap
# ---------------------------------------------------------------------------


def plot_seed_heatmap(all_results: dict[int, list[dict]], passages: list[int]) -> None:
    """Heatmap: rows = seeds, columns = passage counts, value = success (0/1)."""
    seeds = sorted(all_results.keys())
    labels = [str(p) for p in passages]
    matrix = np.zeros((len(seeds), len(passages)), dtype=int)
    for si, seed in enumerate(seeds):
        for pi, p in enumerate(passages):
            row = next((r for r in all_results[seed] if r["passages"] == p), None)
            if row and row["goal_reached"]:
                matrix[si, pi] = 1

    fig, ax = plt.subplots(
        figsize=(max(4, len(passages) * 0.9), max(3, len(seeds) * 0.6))
    )
    ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(passages)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(seeds)))
    ax.set_yticklabels([f"Seed {s}" for s in seeds])
    ax.set_xlabel("Number of Passages")
    ax.set_title("BiRRT Goal-Reaching on Motion2D Environment")

    for si in range(len(seeds)):
        for pi in range(len(passages)):
            label = "Y" if matrix[si, pi] else "N"
            ax.text(
                pi,
                si,
                label,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="black",
            )

    fig.tight_layout()
    out = PLOTS_DIR / "seed_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Load BiRRT results and generate all summary plots."""
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(
            f"Results directory not found: {RESULTS_DIR}\n"
            "Run run_birrt_motion2d.py first."
        )
    all_results, passages = load_all_results()

    plot_success_rate(all_results, passages)
    plot_avg_steps(all_results, passages)
    plot_planning_cost(all_results, passages)
    plot_seed_heatmap(all_results, passages)
    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
