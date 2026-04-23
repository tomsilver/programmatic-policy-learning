"""Cross-demo feature scoring utilities for LPP."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from programmatic_policy_learning.dsl.state_action_program import StateActionProgram


def _column_sums(X: Any, row_mask: np.ndarray | None = None) -> np.ndarray:
    X_part = X[row_mask] if row_mask is not None else X
    if X_part.shape[0] == 0:
        return np.zeros(X.shape[1], dtype=float)
    return np.asarray(X_part.sum(axis=0)).ravel().astype(float)


def score_cross_demo_features(
    X: Any,
    y: np.ndarray,
    row_demo_ids: np.ndarray,
    programs_sa: list[StateActionProgram],
    *,
    consistency_tau: float = 0.05,
) -> list[dict[str, Any]]:
    """Score feature support, contrast, and per-demo stability.

    Rows are state-action examples. Positive rows are expert actions; negative
    rows are candidate actions. ``row_demo_ids`` must align one-to-one with rows
    in ``X`` and ``y``.
    """
    y_flat = y.astype(bool).flatten()
    row_demo_ids = np.asarray(row_demo_ids).reshape(-1)
    if X.shape[0] != y_flat.shape[0] or X.shape[0] != row_demo_ids.shape[0]:
        raise ValueError(
            "Feature scoring inputs must align: "
            f"X rows={X.shape[0]} y={y_flat.shape[0]} demo_ids={row_demo_ids.shape[0]}"
        )
    if X.shape[1] != len(programs_sa):
        raise ValueError(
            "Feature scoring programs must align with columns: "
            f"X cols={X.shape[1]} programs={len(programs_sa)}"
        )

    n_rows = int(X.shape[0])
    pos_mask = y_flat
    neg_mask = ~y_flat
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    unique_demo_ids = sorted(int(demo_id) for demo_id in np.unique(row_demo_ids))
    n_demos = len(unique_demo_ids)

    total_fire = _column_sums(X)
    pos_fire = _column_sums(X, pos_mask)
    neg_fire = _column_sums(X, neg_mask)
    p_pos = pos_fire / max(1, n_pos)
    p_neg = neg_fire / max(1, n_neg)
    signed_contrast = p_pos - p_neg
    abs_contrast = np.abs(signed_contrast)
    fire_rate = total_fire / max(1, n_rows)

    pos_demo_support = np.zeros(X.shape[1], dtype=int)
    neg_demo_support = np.zeros(X.shape[1], dtype=int)
    per_demo_contrasts: list[list[float]] = [[] for _ in range(X.shape[1])]
    per_demo_abs_contrasts: list[list[float]] = [[] for _ in range(X.shape[1])]
    per_demo_valid_counts = np.zeros(X.shape[1], dtype=int)
    consistency_counts = np.zeros(X.shape[1], dtype=int)

    for demo_id in unique_demo_ids:
        demo_mask = row_demo_ids == demo_id
        demo_pos_mask = demo_mask & pos_mask
        demo_neg_mask = demo_mask & neg_mask
        demo_n_pos = int(demo_pos_mask.sum())
        demo_n_neg = int(demo_neg_mask.sum())
        if demo_n_pos > 0:
            demo_pos_fire = _column_sums(X, demo_pos_mask)
            pos_demo_support += (demo_pos_fire > 0).astype(int)
        if demo_n_neg > 0:
            demo_neg_fire = _column_sums(X, demo_neg_mask)
            neg_demo_support += (demo_neg_fire > 0).astype(int)
        if demo_n_pos == 0 or demo_n_neg == 0:
            continue
        demo_pos_rate = _column_sums(X, demo_pos_mask) / demo_n_pos
        demo_neg_rate = _column_sums(X, demo_neg_mask) / demo_n_neg
        demo_contrast = demo_pos_rate - demo_neg_rate
        demo_abs_contrast = np.abs(demo_contrast)
        per_demo_valid_counts += 1
        consistency_counts += (demo_abs_contrast >= consistency_tau).astype(int)
        for col_idx in range(X.shape[1]):
            per_demo_contrasts[col_idx].append(float(demo_contrast[col_idx]))
            per_demo_abs_contrasts[col_idx].append(float(demo_abs_contrast[col_idx]))

    scores: list[dict[str, Any]] = []
    for col_idx, program in enumerate(programs_sa):
        valid_count = int(per_demo_valid_counts[col_idx])
        mean_demo_contrast = (
            float(np.mean(per_demo_contrasts[col_idx])) if valid_count else 0.0
        )
        mean_demo_abs_contrast = (
            float(np.mean(per_demo_abs_contrasts[col_idx])) if valid_count else 0.0
        )
        pos_support_count = int(pos_demo_support[col_idx])
        neg_support_count = int(neg_demo_support[col_idx])
        pos_support_frac = pos_support_count / max(1, n_demos)
        neg_support_frac = neg_support_count / max(1, n_demos)
        directional_support_frac = max(pos_support_frac, neg_support_frac)
        consistency_frac = int(consistency_counts[col_idx]) / max(1, valid_count)
        cross_demo_score = (
            directional_support_frac * abs(mean_demo_contrast) * consistency_frac
        )
        scores.append(
            {
                "feature_index": col_idx,
                "program": str(program),
                "total_fire_count": int(total_fire[col_idx]),
                "total_fire_rate": float(fire_rate[col_idx]),
                "positive_fire_count": int(pos_fire[col_idx]),
                "negative_fire_count": int(neg_fire[col_idx]),
                "p_pos": float(p_pos[col_idx]),
                "p_neg": float(p_neg[col_idx]),
                "signed_contrast": float(signed_contrast[col_idx]),
                "abs_contrast": float(abs_contrast[col_idx]),
                "pos_demo_support_count": pos_support_count,
                "pos_demo_support_frac": float(pos_support_frac),
                "neg_demo_support_count": neg_support_count,
                "neg_demo_support_frac": float(neg_support_frac),
                "directional_support_frac": float(directional_support_frac),
                "valid_contrast_demo_count": valid_count,
                "mean_demo_contrast": mean_demo_contrast,
                "mean_demo_abs_contrast": mean_demo_abs_contrast,
                "contrast_consistency_count": int(consistency_counts[col_idx]),
                "contrast_consistency_frac": float(consistency_frac),
                "cross_demo_score": float(cross_demo_score),
            }
        )
    return scores


def feature_score_keep_mask(
    scores: list[dict[str, Any]],
    *,
    min_pos_demo_support: int = 1,
    min_neg_demo_support: int | None = None,
    allow_negative_support: bool = True,
    min_abs_mean_demo_contrast: float = 0.0,
    min_consistency_frac: float = 0.0,
    min_total_fire_rate: float = 0.0,
    max_total_fire_rate: float = 1.0,
) -> np.ndarray:
    """Return a keep mask from cross-demo feature-score thresholds."""
    if min_neg_demo_support is None:
        min_neg_demo_support = min_pos_demo_support
    keep = []
    for score in scores:
        positive_support_ok = (
            int(score["pos_demo_support_count"]) >= min_pos_demo_support
        )
        negative_support_ok = allow_negative_support and (
            int(score.get("neg_demo_support_count", 0)) >= min_neg_demo_support
        )
        keep.append(
            (positive_support_ok or negative_support_ok)
            and float(score["mean_demo_abs_contrast"]) >= min_abs_mean_demo_contrast
            and float(score["contrast_consistency_frac"]) >= min_consistency_frac
            and float(score["total_fire_rate"]) >= min_total_fire_rate
            and float(score["total_fire_rate"]) <= max_total_fire_rate
        )
    return np.asarray(keep, dtype=bool)


def write_feature_scores(
    output_path: Path,
    scores: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
) -> None:
    """Write feature-score diagnostics as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "features": scores,
    }
    output_path.write_text(json.dumps(payload, indent=4), encoding="utf-8")
    logging.info("Cross-demo feature scores written to %s", output_path)
