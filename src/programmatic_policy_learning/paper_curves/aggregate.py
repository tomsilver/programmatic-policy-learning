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


def compute_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed results into mean/std/sem curves."""
    if results_df.empty:
        return pd.DataFrame()

    success_df = results_df[results_df["status"] == "success"].copy()
    if success_df.empty:
        return pd.DataFrame()

    grouped = (
        success_df.groupby(
            [
                "environment",
                "environment_key",
                "method_name",
                "method_display_name",
                "demo_count",
            ],
            dropna=False,
        )["success_rate"]
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
    return grouped.sort_values(
        by=["environment", "method_name", "demo_count"], ignore_index=True
    )
