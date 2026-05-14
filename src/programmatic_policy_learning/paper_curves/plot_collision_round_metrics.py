"""Plot collision-pair trajectories from LPP collision-round artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
import fnmatch
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value.strip()).strip("-").lower()
    return cleaned or "item"


def _apply_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def _load_collision_round_records(
    results_dir: Path, run_patterns: list[str] | None = None
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for metrics_path in sorted(results_dir.glob("runs/*/collision_round_metrics.json")):
        run_dir = metrics_path.parent
        # If run_patterns provided, only include matching run directories.
        if run_patterns:
            matched = False
            for pat in run_patterns:
                if fnmatch.fnmatch(run_dir.name, pat):
                    matched = True
                    break
            if not matched:
                continue
        metrics_payload = _read_json(metrics_path)
        result_path = run_dir / "result.json"
        result_payload = _read_json(result_path) if result_path.exists() else {}
        method_name = str(
            result_payload.get(
                "method_name",
                run_dir.name.split("__")[1] if "__" in run_dir.name else run_dir.name,
            )
        )
        method_label = str(
            result_payload.get("method_display_name", method_name.replace("_", " "))
        )
        environment_key = str(
            result_payload.get(
                "environment_key",
                result_payload.get("backend_environment", "unknown"),
            )
        )
        environment_name = str(result_payload.get("environment", environment_key))
        demo_count = result_payload.get("demo_count")
        seed = result_payload.get("seed")
        for metric in metrics_payload.get("round_metrics", []):
            record = {
                "run_id": run_dir.name,
                "environment_key": environment_key,
                "environment": environment_name,
                "method_name": method_name,
                "method_label": method_label,
                "demo_count": int(demo_count) if demo_count is not None else None,
                "seed": int(seed) if seed is not None else None,
                "round": int(metric["round"]),
                "stage": str(metric.get("stage", "")),
                "approx_pairs": int(metric.get("approx_pairs", 0)),
                "mixed_buckets": int(metric.get("mixed_buckets", 0)),
                "collided_rows": int(metric.get("collided_rows", 0)),
                "num_rows": int(metric.get("num_rows", 0)),
                "num_features": int(metric.get("num_features", 0)),
                "generated_feature_count": int(
                    metric.get("generated_feature_count", 0)
                ),
            }
            records.append(record)
    return records


def _sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def _write_records_csv(records: list[dict[str, Any]], csv_path: Path) -> None:
    if not records:
        return
    fieldnames = [
        "run_id",
        "environment_key",
        "environment",
        "method_name",
        "method_label",
        "demo_count",
        "seed",
        "round",
        "stage",
        "approx_pairs",
        "mixed_buckets",
        "collided_rows",
        "num_rows",
        "num_features",
        "generated_feature_count",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _save_collision_pairs_plots(
    records: list[dict[str, Any]], plots_dir: Path
) -> list[Path]:
    if not records:
        return []
    _apply_plot_style()
    _ensure_dir(plots_dir)
    saved_paths: list[Path] = []
    records_by_env: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_env[str(record["environment_key"])].append(record)
    for env_key in sorted(records_by_env):
        env_records = records_by_env[env_key]
        fig, ax = plt.subplots(figsize=(6.4, 4.1))
        grouped_values: dict[tuple[str, str, int, int], list[float]] = defaultdict(list)
        for record in env_records:
            key = (
                str(record["method_name"]),
                str(record["method_label"]),
                int(record["demo_count"]),
                int(record["round"]),
            )
            grouped_values[key].append(float(record["approx_pairs"]))
        series_keys = sorted(
            {(method, label, demo_count) for method, label, demo_count, _ in grouped_values}
        )
        for method_name, method_label, demo_count in series_keys:
            points: list[tuple[int, float, float]] = []
            for key, values in grouped_values.items():
                if key[:3] != (method_name, method_label, demo_count):
                    continue
                round_idx = key[3]
                points.append((round_idx, statistics.mean(values), _sem(values)))
            points.sort(key=lambda item: item[0])
            label = f"{method_label} d={demo_count}"
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            ax.plot(x_values, y_values, marker="o", linewidth=2.0, label=label)
            sem_values = [point[2] for point in points]
            if any(value > 0 for value in sem_values):
                ax.fill_between(
                    x_values,
                    [y - sem for y, sem in zip(y_values, sem_values)],
                    [y + sem for y, sem in zip(y_values, sem_values)],
                    alpha=0.18,
                )
        env_name = str(env_records[0]["environment"])
        ax.set_xlabel("Collision feedback round")
        ax.set_ylabel("Approx. collision pairs")
        ax.set_title(f"{env_name}: Feature collision pairs by round")
        ax.set_xticks(sorted({int(record["round"]) for record in env_records}))
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False)
        ax.set_axisbelow(True)
        fig.tight_layout()
        out_path = plots_dir / f"{_slugify(str(env_key))}_collision_pairs_by_round.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)
    return saved_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        required=True,
        type=Path,
        help="Paper-curves results directory containing runs/* artifacts.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <results-dir>/plots.",
    )
    parser.add_argument(
        "--run-pattern",
        action="append",
        default=None,
        help=(
            "Only include runs whose run directory name matches any of these shell-style "
            "patterns (can be provided multiple times). Example: 'nim__our-main__d10__*'"
        ),
    )
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    plots_dir = (
        args.plots_dir.resolve()
        if args.plots_dir is not None
        else results_dir / "plots"
    )
    records = _load_collision_round_records(results_dir, run_patterns=args.run_pattern)
    if not records:
        print(f"No collision_round_metrics.json files found under {results_dir}.")
        return 1
    _ensure_dir(plots_dir)
    csv_path = plots_dir / "collision_round_metrics.csv"
    _write_records_csv(records, csv_path)
    saved_paths = _save_collision_pairs_plots(records, plots_dir)
    print(f"Wrote {csv_path}")
    for path in saved_paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
