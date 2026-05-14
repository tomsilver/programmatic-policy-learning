"""Aggregation helpers for paper-curve runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from programmatic_policy_learning.paper_curves.common import (
    find_result_files,
    read_json,
)


def load_results_dataframe(results_dir: Path) -> pd.DataFrame:
    """Load normalized result JSON files into a dataframe."""
    rows: list[dict[str, Any]] = []
    for result_path in find_result_files(results_dir):
        payload = read_json(result_path)
        row = dict(payload)
        row["result_path"] = str(result_path.resolve())
        for key, value in list(row.items()):
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value, sort_keys=True)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_summary(
    results_df: pd.DataFrame,
    *,
    x_key: str = "demo_count",
) -> pd.DataFrame:
    """Aggregate per-seed train/test metrics into summary curves."""
    if results_df.empty:
        return pd.DataFrame()

    success_df = results_df[results_df["status"] == "success"].copy()
    if success_df.empty:
        return pd.DataFrame()
    if x_key not in success_df.columns:
        raise ValueError(f"Column '{x_key}' not found in results dataframe.")

    metric_frames: list[pd.DataFrame] = []
    metric_specs = (
        ("train", "train_success_rate", "Train success rate"),
        ("test", "test_success_rate", "Test success rate"),
    )
    group_cols = [
        "environment",
        "environment_key",
        "method_name",
        "method_display_name",
        x_key,
    ]

    for split_name, metric_col, metric_label in metric_specs:
        if metric_col not in success_df.columns:
            continue
        split_df = success_df[success_df[metric_col].notna()].copy()
        if split_df.empty:
            continue
        grouped = (
            split_df.groupby(group_cols, dropna=False)[metric_col]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        grouped["std"] = grouped["std"].fillna(0.0)
        grouped["sem"] = grouped.apply(
            lambda row: (
                float(row["std"]) / float(row["count"]) ** 0.5
                if float(row["count"]) > 0
                else 0.0
            ),
            axis=1,
        )
        grouped = grouped.rename(
            columns={
                "mean": "mean_success_rate",
                "std": "std_success_rate",
                "count": "num_seeds",
            }
        )
        grouped["eval_split"] = split_name
        grouped["metric_label"] = metric_label
        metric_frames.append(grouped)

    if not metric_frames:
        # Backward-compatible fallback for legacy results.
        grouped = (
            success_df.groupby(group_cols, dropna=False)["success_rate"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        grouped["std"] = grouped["std"].fillna(0.0)
        grouped["sem"] = grouped.apply(
            lambda row: (
                float(row["std"]) / float(row["count"]) ** 0.5
                if float(row["count"]) > 0
                else 0.0
            ),
            axis=1,
        )
        grouped = grouped.rename(
            columns={
                "mean": "mean_success_rate",
                "std": "std_success_rate",
                "count": "num_seeds",
            }
        )
        grouped["eval_split"] = "test"
        grouped["metric_label"] = "Test success rate"
        metric_frames.append(grouped)

    combined = pd.concat(metric_frames, ignore_index=True)
    return combined.sort_values(
        by=["environment", "method_name", "eval_split", x_key],
        ignore_index=True,
    )
