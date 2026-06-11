"""Run and plot fixed-demo collision-round ablations for LPP."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import logging
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from programmatic_policy_learning.paper_curves.common import (
    ensure_dir,
    load_yaml_config,
    read_json,
    setup_logging,
    shared_sqlite_cache_dir,
    slugify,
    utc_timestamp,
    write_json,
)
from programmatic_policy_learning.paper_curves.driver import _run_jobs


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


def _default_output_dir(config: dict[str, Any]) -> Path:
    output_root = Path(str(config.get("output_root", "results/paper_curves")))
    experiment_name = config.get("experiment_name", "collision_round_ablation")
    return output_root / slugify(str(experiment_name))


def _round_override_method(base_method: dict[str, Any], round_budget: int) -> dict[str, Any]:
    method = dict(base_method)
    overrides = [str(override) for override in method.get("overrides", [])]
    filtered: list[str] = []
    round_override_prefixes = (
        "approach.collision_feedback_enabled=",
        "approach.collision_feedback_max_rounds=",
    )
    for override in overrides:
        if any(override.startswith(prefix) for prefix in round_override_prefixes):
            continue
        filtered.append(override)
    if round_budget <= 0:
        filtered.append("approach.collision_feedback_enabled=false")
    else:
        filtered.append("approach.collision_feedback_enabled=true")
        filtered.append(f"approach.collision_feedback_max_rounds={int(round_budget)}")
    method["overrides"] = filtered
    base_name = str(base_method.get("name", "lpp"))
    base_label = str(base_method.get("display_name", base_name))
    method["name"] = f"{base_name}__r{int(round_budget)}"
    method["display_name"] = f"{base_label} (r={int(round_budget)})"
    method["shared_cache_base_name"] = base_name
    method["collision_round_budget"] = int(round_budget)
    return method


def _shared_run_cache_dir(
    *,
    results_dir: Path,
    env_key: str,
    method_cfg: dict[str, Any],
    demo_ids: list[int],
    seed: int,
) -> str | None:
    if not bool(method_cfg.get("shared_run_cache", False)):
        return None
    signature_overrides = [
        str(override)
        for override in method_cfg.get("overrides", [])
        if not str(override).startswith("approach.collision_feedback_enabled=")
        and not str(override).startswith("approach.collision_feedback_max_rounds=")
    ]
    signature_payload = {
        "method_name": str(
            method_cfg.get("shared_cache_base_name", method_cfg.get("name", ""))
        ),
        "demo_ids": [int(each) for each in demo_ids],
        "signature_overrides": sorted(signature_overrides),
        "seed": int(seed),
        "kind": "collision_round_run_cache",
    }
    signature = hashlib.sha1(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    cache_dir = ensure_dir(
        results_dir / "shared_caches" / slugify(env_key) / "collision_round_run_cache"
    )
    return str(
        (
            cache_dir
            / (
                f"{slugify(str(method_cfg.get('shared_cache_base_name', method_cfg.get('name', 'method'))))}"
                f"__seed{int(seed)}__{signature}"
            )
        ).resolve()
    )


def _build_jobs(
    config: dict[str, Any],
    *,
    results_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    environments = list(config.get("environments", []))
    if not environments:
        raise ValueError("Config must define at least one environment.")
    base_method = dict(config.get("method", {}))
    if str(base_method.get("backend", "lpp")).lower() != "lpp":
        raise ValueError("collision_round_ablation only supports backend: lpp.")
    demo_count = int(config["demo_count"])
    demo_id_pool = [int(each) for each in config.get("demo_id_pool", list(range(10)))]
    if demo_count > len(demo_id_pool):
        raise ValueError(
            f"demo_count={demo_count} exceeds demo_id_pool size {len(demo_id_pool)}."
        )
    demo_ids = [int(each) for each in demo_id_pool[:demo_count]]
    round_budgets = [int(each) for each in config["round_budgets"]]
    global_seeds = [int(each) for each in config["seeds"]]
    test_env_nums = [int(each) for each in config.get("test_env_nums", list(range(10, 20)))]
    backend_cfg = dict(config.get("codebases", {}).get("lpp", {}))
    repo_root = Path(str(backend_cfg.get("root_dir", "."))).resolve()
    backend_python = str(backend_cfg.get("python_executable", "python"))

    methods = [_round_override_method(base_method, round_budget) for round_budget in round_budgets]
    jobs: list[dict[str, Any]] = []
    for env_cfg in environments:
        env_key = str(env_cfg.get("key", env_cfg["name"]))
        for method_cfg in methods:
            round_budget = int(method_cfg["collision_round_budget"])
            method_seeds = [int(each) for each in method_cfg.get("seeds", global_seeds)]
            for seed in method_seeds:
                run_id = (
                    f"{slugify(env_key)}__{slugify(str(base_method.get('name', 'lpp')))}"
                    f"__d{demo_count}__r{round_budget}__s{seed}"
                )
                artifact_dir = results_dir / "runs" / run_id
                result_path = artifact_dir / "result.json"
                shared_run_cache_dir = _shared_run_cache_dir(
                    results_dir=results_dir,
                    env_key=env_key,
                    method_cfg=method_cfg,
                    demo_ids=demo_ids,
                    seed=seed,
                )
                jobs.append(
                    {
                        "run_id": run_id,
                        "repo_root": str(repo_root),
                        "backend_python": backend_python,
                        "environment": env_cfg,
                        "method": method_cfg,
                        "seed": int(seed),
                        "demo_count": int(demo_count),
                        "demo_ids": demo_ids,
                        "train_env_nums": list(demo_ids),
                        "test_env_nums": test_env_nums,
                        "shared_sqlite_cache_dir": str(
                            shared_sqlite_cache_dir(results_dir, "lpp").resolve()
                        ),
                        "shared_run_cache_dir": shared_run_cache_dir,
                        "eval_max_steps": int(config.get("eval_max_steps", 100)),
                        "artifact_dir": str(artifact_dir.resolve()),
                        "result_path": str(result_path.resolve()),
                    }
                )
    return jobs, environments, methods


def _mean_std_sem(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    mean = float(statistics.mean(values))
    std = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    sem = float(std / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, std, sem


def _load_final_records(
    results_dir: Path,
    run_patterns: list[str] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result_path in sorted(results_dir.glob("runs/*/result.json")):
        run_dir = result_path.parent
        if run_patterns and not any(fnmatch.fnmatch(run_dir.name, pat) for pat in run_patterns):
            continue
        result_payload = read_json(result_path)
        if result_payload.get("status") != "success":
            continue
        metrics_path = run_dir / "collision_round_metrics.json"
        if not metrics_path.exists():
            continue
        metrics_payload = read_json(metrics_path)
        round_metrics = list(metrics_payload.get("round_metrics", []))
        if not round_metrics:
            continue
        final_metric = dict(round_metrics[-1])
        config_fields = dict(result_payload.get("config_fields", {}))
        enabled = bool(config_fields.get("collision_feedback_enabled", False))
        requested_round_budget = (
            int(config_fields.get("collision_feedback_max_rounds", 0)) if enabled else 0
        )
        records.append(
            {
                "run_id": run_dir.name,
                "environment": str(result_payload.get("environment", "unknown")),
                "environment_key": str(
                    result_payload.get(
                        "environment_key",
                        result_payload.get("backend_environment", "unknown"),
                    )
                ),
                "seed": int(result_payload["seed"]),
                "demo_count": int(result_payload["demo_count"]),
                "requested_round_budget": requested_round_budget,
                "actual_round": int(final_metric.get("round", 0)),
                "train_success_rate": float(result_payload["train_success_rate"]),
                "test_success_rate": float(result_payload["test_success_rate"]),
                "approx_pairs": int(final_metric.get("approx_pairs", 0)),
                "lower_bound_error_count": int(
                    final_metric.get("lower_bound_error_count", 0)
                ),
                "mixed_buckets": int(final_metric.get("mixed_buckets", 0)),
                "collided_rows": int(final_metric.get("collided_rows", 0)),
                "num_rows": int(final_metric.get("num_rows", 0)),
                "num_features": int(final_metric.get("num_features", 0)),
            }
        )
    return records


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        key = (
            str(record["environment_key"]),
            str(record["environment"]),
            int(record["requested_round_budget"]),
        )
        for metric_name in (
            "actual_round",
            "train_success_rate",
            "test_success_rate",
            "approx_pairs",
            "lower_bound_error_count",
            "mixed_buckets",
            "collided_rows",
            "num_rows",
            "num_features",
        ):
            grouped[key][metric_name].append(float(record[metric_name]))

    summary_rows: list[dict[str, Any]] = []
    for (env_key, env_name, requested_round_budget), metric_lists in sorted(grouped.items()):
        row: dict[str, Any] = {
            "environment_key": env_key,
            "environment": env_name,
            "requested_round_budget": requested_round_budget,
            "num_seed_runs": len(metric_lists["train_success_rate"]),
        }
        for metric_name, values in metric_lists.items():
            mean, std, sem = _mean_std_sem(values)
            row[f"{metric_name}_mean"] = mean
            row[f"{metric_name}_std"] = std
            row[f"{metric_name}_sem"] = sem
        summary_rows.append(row)
    return summary_rows


def _plot_env_collision_metrics(
    env_cfg: dict[str, Any],
    env_summary_rows: list[dict[str, Any]],
    plots_dir: Path,
) -> list[Path]:
    _apply_plot_style()
    ensure_dir(plots_dir)
    env_key = str(env_cfg.get("key", env_cfg["name"]))
    env_title = str(env_cfg.get("plot_title", env_cfg["name"]))
    caption = str(env_cfg.get("plot_caption", "")).strip()
    env_rows = sorted(env_summary_rows, key=lambda row: int(row["requested_round_budget"]))
    x_values = [int(row["requested_round_budget"]) for row in env_rows]
    approx_means = [float(row.get("approx_pairs_mean", 0.0) or 0.0) for row in env_rows]
    approx_sems = [float(row.get("approx_pairs_sem", 0.0) or 0.0) for row in env_rows]
    lower_means = [
        float(row.get("lower_bound_error_count_mean", 0.0) or 0.0) for row in env_rows
    ]
    lower_sems = [
        float(row.get("lower_bound_error_count_sem", 0.0) or 0.0) for row in env_rows
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharex=True)
    specs = (
        (axes[0], approx_means, approx_sems, "Approx. collision pairs"),
        (axes[1], lower_means, lower_sems, "Collision lower-bound error"),
    )
    for ax, means, sems, ylabel in specs:
        ax.plot(x_values, means, marker="o", linewidth=2.0)
        if any(value > 0 for value in sems):
            ax.fill_between(
                x_values,
                [mean - sem for mean, sem in zip(means, sems)],
                [mean + sem for mean, sem in zip(means, sems)],
                alpha=0.18,
            )
        ax.set_xlabel("Collision-feedback round budget")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.set_axisbelow(True)
    fig.suptitle(f"{env_title}: Final collisions vs round budget")
    if caption:
        fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=9, wrap=True)
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    else:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    stem = slugify(env_key)
    png_path = plots_dir / f"{stem}_collision_metrics_vs_round.png"
    pdf_path = plots_dir / f"{stem}_collision_metrics_vs_round.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def _plot_env_accuracy_vs_round(
    env_cfg: dict[str, Any],
    env_summary_rows: list[dict[str, Any]],
    plots_dir: Path,
) -> list[Path]:
    _apply_plot_style()
    ensure_dir(plots_dir)
    env_key = str(env_cfg.get("key", env_cfg["name"]))
    env_title = str(env_cfg.get("plot_title", env_cfg["name"]))
    caption = str(env_cfg.get("plot_caption", "")).strip()
    env_rows = sorted(env_summary_rows, key=lambda row: int(row["requested_round_budget"]))
    x_values = [int(row["requested_round_budget"]) for row in env_rows]
    train_means = [
        float(row.get("train_success_rate_mean", 0.0) or 0.0) for row in env_rows
    ]
    train_sems = [float(row.get("train_success_rate_sem", 0.0) or 0.0) for row in env_rows]
    test_means = [float(row.get("test_success_rate_mean", 0.0) or 0.0) for row in env_rows]
    test_sems = [float(row.get("test_success_rate_sem", 0.0) or 0.0) for row in env_rows]

    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    ax.plot(x_values, train_means, marker="o", linewidth=2.0, label="Train accuracy")
    ax.plot(
        x_values,
        test_means,
        marker="s",
        linewidth=2.0,
        linestyle="--",
        label="Test accuracy",
    )
    if any(value > 0 for value in train_sems):
        ax.fill_between(
            x_values,
            [mean - sem for mean, sem in zip(train_means, train_sems)],
            [mean + sem for mean, sem in zip(train_means, train_sems)],
            alpha=0.18,
        )
    if any(value > 0 for value in test_sems):
        ax.fill_between(
            x_values,
            [mean - sem for mean, sem in zip(test_means, test_sems)],
            [mean + sem for mean, sem in zip(test_means, test_sems)],
            alpha=0.12,
        )
    ax.set_xlabel("Collision-feedback round budget")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    ax.set_title(f"{env_title}: Accuracy vs round budget")
    if caption:
        fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=9, wrap=True)
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    else:
        fig.tight_layout()
    stem = slugify(env_key)
    png_path = plots_dir / f"{stem}_accuracy_vs_round.png"
    pdf_path = plots_dir / f"{stem}_accuracy_vs_round.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def _plot_combined_accuracy_vs_round(
    summary_rows: list[dict[str, Any]],
    environments: list[dict[str, Any]],
    plots_dir: Path,
) -> list[Path]:
    """Plot train/test accuracy curves for every environment on one chart."""
    _apply_plot_style()
    ensure_dir(plots_dir)
    env_titles = {
        str(env_cfg.get("key", env_cfg["name"])): str(
            env_cfg.get("plot_title", env_cfg["name"])
        )
        for env_cfg in environments
    }
    summary_by_env: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        summary_by_env[str(row["environment_key"])].append(row)
        env_titles.setdefault(str(row["environment_key"]), str(row["environment"]))

    if len(summary_by_env) < 2:
        return []

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for index, (env_key, env_rows_unsorted) in enumerate(sorted(summary_by_env.items())):
        env_rows = sorted(
            env_rows_unsorted, key=lambda row: int(row["requested_round_budget"])
        )
        x_values = [int(row["requested_round_budget"]) for row in env_rows]
        train_means = [
            float(row.get("train_success_rate_mean", 0.0) or 0.0) for row in env_rows
        ]
        train_sems = [
            float(row.get("train_success_rate_sem", 0.0) or 0.0) for row in env_rows
        ]
        test_means = [
            float(row.get("test_success_rate_mean", 0.0) or 0.0) for row in env_rows
        ]
        test_sems = [
            float(row.get("test_success_rate_sem", 0.0) or 0.0) for row in env_rows
        ]
        color = colors[index % len(colors)] if colors else None
        env_label = env_titles.get(env_key, env_key)
        ax.plot(
            x_values,
            train_means,
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"{env_label} train",
        )
        ax.plot(
            x_values,
            test_means,
            marker="s",
            linewidth=2.0,
            linestyle="--",
            color=color,
            label=f"{env_label} test",
        )
        if any(value > 0 for value in train_sems):
            ax.fill_between(
                x_values,
                [mean - sem for mean, sem in zip(train_means, train_sems)],
                [mean + sem for mean, sem in zip(train_means, train_sems)],
                color=color,
                alpha=0.14,
            )
        if any(value > 0 for value in test_sems):
            ax.fill_between(
                x_values,
                [mean - sem for mean, sem in zip(test_means, test_sems)],
                [mean + sem for mean, sem in zip(test_means, test_sems)],
                color=color,
                alpha=0.08,
            )

    ax.set_xlabel("Collision-feedback round budget")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2)
    ax.set_title("Accuracy vs collision-feedback round budget")
    fig.tight_layout()
    png_path = plots_dir / "combined_accuracy_vs_round.png"
    pdf_path = plots_dir / "combined_accuracy_vs_round.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def _plot_env_train_accuracy_vs_collision(
    env_cfg: dict[str, Any],
    env_summary_rows: list[dict[str, Any]],
    plots_dir: Path,
) -> list[Path]:
    _apply_plot_style()
    ensure_dir(plots_dir)
    env_key = str(env_cfg.get("key", env_cfg["name"]))
    env_title = str(env_cfg.get("plot_title", env_cfg["name"]))
    caption = str(env_cfg.get("plot_caption", "")).strip()
    env_rows = sorted(env_summary_rows, key=lambda row: int(row["requested_round_budget"]))
    x_values = [
        float(row.get("lower_bound_error_count_mean", 0.0) or 0.0) for row in env_rows
    ]
    x_err = [
        float(row.get("lower_bound_error_count_sem", 0.0) or 0.0) for row in env_rows
    ]
    y_values = [float(row.get("train_success_rate_mean", 0.0) or 0.0) for row in env_rows]
    y_err = [float(row.get("train_success_rate_sem", 0.0) or 0.0) for row in env_rows]
    labels = [int(row["requested_round_budget"]) for row in env_rows]

    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    ax.plot(x_values, y_values, linewidth=1.5, alpha=0.8)
    ax.errorbar(
        x_values,
        y_values,
        xerr=x_err,
        yerr=y_err,
        fmt="o",
        capsize=3,
        linewidth=1.3,
    )
    for label, x_val, y_val in zip(labels, x_values, y_values):
        ax.annotate(f"r={label}", (x_val, y_val), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Final collision lower-bound error")
    ax.set_ylabel("Train accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(left=0.0)
    ax.set_axisbelow(True)
    ax.set_title(f"{env_title}: Train accuracy vs collisions")
    if caption:
        fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=9, wrap=True)
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    else:
        fig.tight_layout()
    stem = slugify(env_key)
    png_path = plots_dir / f"{stem}_train_accuracy_vs_collision.png"
    pdf_path = plots_dir / f"{stem}_train_accuracy_vs_collision.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def _write_outputs(
    results_dir: Path,
    config: dict[str, Any],
    environments: list[dict[str, Any]],
    run_patterns: list[str] | None = None,
) -> list[Path]:
    records = _load_final_records(results_dir, run_patterns=run_patterns)
    if not records:
        logging.warning("No successful collision-round ablation records found under %s.", results_dir)
        return []
    plots_dir = ensure_dir(results_dir / "plots")
    raw_csv_path = results_dir / "collision_round_ablation_records.csv"
    raw_fields = [
        "run_id",
        "environment",
        "environment_key",
        "seed",
        "demo_count",
        "requested_round_budget",
        "actual_round",
        "train_success_rate",
        "test_success_rate",
        "approx_pairs",
        "lower_bound_error_count",
        "mixed_buckets",
        "collided_rows",
        "num_rows",
        "num_features",
    ]
    _write_csv(raw_csv_path, records, raw_fields)

    summary_rows = _summarize_records(records)
    summary_csv_path = results_dir / "collision_round_ablation_summary.csv"
    summary_fields = sorted(summary_rows[0].keys()) if summary_rows else []
    _write_csv(summary_csv_path, summary_rows, summary_fields)

    saved_paths: list[Path] = [raw_csv_path, summary_csv_path]
    summary_by_env: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        summary_by_env[str(row["environment_key"])].append(row)
    saved_paths.extend(
        _plot_combined_accuracy_vs_round(summary_rows, environments, plots_dir)
    )
    for env_cfg in environments:
        env_key = str(env_cfg.get("key", env_cfg["name"]))
        env_rows = summary_by_env.get(env_key, [])
        if not env_rows:
            continue
        saved_paths.extend(_plot_env_collision_metrics(env_cfg, env_rows, plots_dir))
        saved_paths.extend(_plot_env_accuracy_vs_round(env_cfg, env_rows, plots_dir))
        saved_paths.extend(_plot_env_train_accuracy_vs_collision(env_cfg, env_rows, plots_dir))
    return saved_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="YAML config path.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Existing or target results directory. Defaults to results/paper_curves/<experiment-name>/.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip launching runs and regenerate CSVs/plots from existing artifacts.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-run jobs even if result.json already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print jobs without launching them.",
    )
    parser.add_argument(
        "--run-pattern",
        action="append",
        default=None,
        help="Only aggregate runs whose directory matches this shell-style pattern. Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_yaml_config(args.config.resolve())
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir is not None
        else _default_output_dir(config).resolve()
    )
    ensure_dir(results_dir)
    setup_logging(results_dir / "driver.log")
    write_json(results_dir / "config_snapshot.json", config)
    logging.info("Results directory: %s", results_dir)

    jobs, environments, _methods = _build_jobs(config, results_dir=results_dir)
    plan_payload = {
        "generated_at": utc_timestamp(),
        "num_jobs": len(jobs),
        "jobs": jobs,
    }
    write_json(results_dir / "planned_jobs.json", plan_payload)

    if not args.plot_only:
        _run_jobs(
            jobs,
            results_dir=results_dir,
            skip_existing=not args.no_skip_existing,
            continue_on_error=bool(config.get("continue_on_error", True)),
            dry_run=bool(args.dry_run),
        )
    if args.dry_run:
        logging.info("Dry run completed; skipping aggregation.")
        return 0

    saved_paths = _write_outputs(
        results_dir,
        config,
        environments,
        run_patterns=args.run_pattern,
    )
    if not saved_paths:
        return 1
    for path in saved_paths:
        logging.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
