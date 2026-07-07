"""Collision feedback helpers for LPP."""

import json
import logging
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
from scipy.sparse import hstack

from programmatic_policy_learning.approaches.lpp_utils.lpp_feature_source_utils import (
    _parse_py_feature_sources,
)
from programmatic_policy_learning.approaches.lpp_utils.utils import (
    log_feature_collisions,
    summarize_collision_groups,
)
from programmatic_policy_learning.data.dataset import run_programs_on_examples
from programmatic_policy_learning.dsl.state_action_program import StateActionProgram
from programmatic_policy_learning.learning.prior_calculation import (
    priors_from_features,
    priors_from_features_v2,
)

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")


def _feature_column_key(X: Any, col_idx: int) -> bytes:
    """Return a stable exact-column key for sparse or dense feature
    matrices."""
    if hasattr(X, "tocsc"):
        X_csc = X.tocsc()
        start = int(X_csc.indptr[col_idx])
        end = int(X_csc.indptr[col_idx + 1])
        indices = np.asarray(X_csc.indices[start:end], dtype=np.int32)
        data = np.asarray(X_csc.data[start:end])
        return indices.tobytes() + b"||" + data.tobytes()

    X_arr = np.asarray(X)
    col = np.ascontiguousarray(X_arr[:, col_idx])
    return col.tobytes()


def _collision_repair_feature_stats(
    *,
    round_idx: int,
    X_existing: Any,
    X_new: Any,
    new_feature_sources: list[str],
) -> dict[str, Any]:
    """Summarize repair-feature usefulness without filtering any columns."""
    n_rows = int(X_new.shape[0])
    existing_keys = {
        _feature_column_key(X_existing, col_idx)
        for col_idx in range(int(X_existing.shape[1]))
    }
    new_seen_keys: dict[bytes, int] = {}
    feature_records: list[dict[str, Any]] = []
    counts = {
        "constant": 0,
        "duplicate_existing": 0,
        "duplicate_within_repair_batch": 0,
        "new_active_distinction": 0,
    }
    new_col_nnz = np.asarray(X_new.getnnz(axis=0)).ravel()

    for local_idx, source in enumerate(new_feature_sources):
        key = _feature_column_key(X_new, local_idx)
        nnz = int(new_col_nnz[local_idx])
        is_constant = nnz == 0 or nnz == n_rows
        duplicate_existing = key in existing_keys
        duplicate_within_batch = key in new_seen_keys

        if is_constant:
            category = "constant"
        elif duplicate_existing:
            category = "duplicate_existing"
        elif duplicate_within_batch:
            category = "duplicate_within_repair_batch"
        else:
            category = "new_active_distinction"

        counts[category] += 1
        feature_records.append(
            {
                "local_feature_index": int(local_idx),
                "category": category,
                "nnz": nnz,
                "fire_rate": float(nnz / n_rows) if n_rows else 0.0,
                "duplicate_of_repair_local_index": (
                    int(new_seen_keys[key]) if duplicate_within_batch else None
                ),
                "source": source,
            }
        )
        new_seen_keys.setdefault(key, local_idx)

    return {
        "round": int(round_idx),
        "added_features": int(len(new_feature_sources)),
        **counts,
        "features": feature_records,
    }


def _write_collision_repair_feature_stats(
    output_path: Path,
    stats_history: list[dict[str, Any]],
) -> None:
    """Write a human-readable repair-feature stats sidecar."""
    lines: list[str] = []
    for stats in stats_history:
        round_idx = int(stats["round"])
        lines.extend(
            [
                (
                    f"round {round_idx}: added "
                    f"{int(stats['added_features'])} repair features"
                ),
                f"- {int(stats['constant'])} constant",
                f"- {int(stats['duplicate_existing'])} duplicate existing features",
                (
                    f"- {int(stats['duplicate_within_repair_batch'])} duplicate "
                    "within repair batch"
                ),
                f"- {int(stats['new_active_distinction'])} new active distinctions",
                "",
            ]
        )
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "collision_repair_feature_stats.txt").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )
    (output_path / "collision_repair_feature_stats.json").write_text(
        json.dumps({"rounds": stats_history}, indent=2),
        encoding="utf-8",
    )


def _collision_round_metric(
    *,
    round_idx: int,
    stage: str,
    collision_groups: list[dict[str, Any]],
    X: Any,
    generated_feature_count: int = 0,
) -> dict[str, Any]:
    summary = summarize_collision_groups(collision_groups)
    metric: dict[str, Any] = {
        "round": int(round_idx),
        "stage": stage,
        "generated_feature_count": int(generated_feature_count),
        "num_rows": int(X.shape[0]),
        "num_features": int(X.shape[1]),
    }
    metric.update(summary)
    return metric


def _collision_summary_is_flat(
    before_summary: dict[str, int],
    after_summary: dict[str, int],
) -> bool:
    """Return whether collision severity did not improve across a round."""
    return int(before_summary.get("lower_bound_error_count", 0)) == int(
        after_summary.get("lower_bound_error_count", 0)
    ) and int(before_summary.get("approx_pairs", 0)) == int(
        after_summary.get("approx_pairs", 0)
    )


def _append_new_features_from_sources(
    X: Any,
    programs_sa: list[StateActionProgram],
    program_prior_log_probs: list[float] | None,
    dsl_functions: dict[str, Any],
    new_feature_sources: list[str],
    examples: list[tuple[ObsT, ActT]],
    *,
    start_index: int,
    collision_loop_idx: int,
    prior_version: str = "v1",
    prior_beta: float = 1.0,
) -> tuple[Any, int]:
    """Append LLM-generated features to matrix/program metadata."""
    new_functions, new_feature_names = _parse_py_feature_sources(
        new_feature_sources, dsl_functions
    )
    dsl_functions.update(new_functions)
    new_programs = [f"{name}(s, a)" for name in new_feature_names]
    new_programs_sa = [StateActionProgram(p) for p in new_programs]
    X_new = run_programs_on_examples(
        new_programs_sa,
        examples,
        dsl_functions,
        feature_sources=new_feature_sources,
        collision_loop_idx=collision_loop_idx,
    )
    X = hstack([X, X_new]).tocsr()
    programs_sa.extend(new_programs_sa)
    if program_prior_log_probs is not None:
        if prior_version == "v2":
            new_priors = priors_from_features_v2(new_feature_sources, beta=prior_beta)[
                "beta_log_scores"
            ]
        elif prior_version in {"v1", "uniform"}:
            new_priors = priors_from_features(new_feature_sources)["logprobs"]
        else:
            raise ValueError(f"Unsupported prior_version: {prior_version}")
        program_prior_log_probs.extend(new_priors)
    return X, start_index + len(new_feature_names)


def run_collision_feedback_loop(
    *,
    collision_groups: list[dict[str, Any]],
    examples: list[tuple[ObsT, ActT]],
    max_rounds: int,
    target_collisions: int,
    start_index: int,
    program_prior_log_probs: list[float] | None,
    X: Any,
    y: np.ndarray | None,
    programs_sa: list[StateActionProgram],
    dsl_functions: dict[str, Any],
    generate_features: Callable[
        [str, int, int], tuple[list[str], dict[str, Any], Path]
    ],
    make_prompt: Callable[
        [list[dict[str, Any]], list[tuple[ObsT, ActT]], bool], str | None
    ],
    record_attempt_summary: (
        Callable[[int, dict[str, Any], int, int], None] | None
    ) = None,
    prior_version: str = "uniform",
    prior_beta: float = 1.0,
    collision_bucket_mode: str = "positive_anchor",
    worst_bucket_reprompt_enabled: bool = False,
) -> tuple[
    Any,
    list[StateActionProgram],
    list[float] | None,
    list[dict[str, Any]],
    Path | None,
    np.ndarray,
    list[dict[str, Any]],
]:
    """Run collision-repair rounds by generating and appending new features."""
    collision_payloads: list[dict[str, Any]] = []
    collision_round_metrics = [
        _collision_round_metric(
            round_idx=0,
            stage="post_initial_filter",
            collision_groups=collision_groups,
            X=X,
        )
    ]
    collision_output_path: Path | None = None
    repair_feature_stats_history: list[dict[str, Any]] = []
    col_nnz = np.asarray(X.getnnz(axis=0)).ravel()
    flat_round_streak = 0
    for round_idx in range(max_rounds):
        num_collisions = len(collision_groups) if collision_groups else 0
        if num_collisions <= target_collisions:
            logging.info(
                "Collision feedback stopping: %d <= target %d.",
                num_collisions,
                target_collisions,
            )
            break
        targeted_reprompt = worst_bucket_reprompt_enabled and flat_round_streak >= 2
        if targeted_reprompt:
            logging.info(
                "Collision repair switching to targeted worst-bucket reprompt "
                "after %d flat round(s).",
                flat_round_streak,
            )
        prompt = make_prompt(collision_groups, examples, targeted_reprompt)
        if prompt is None:
            break
        prompt = f"{prompt}\n\nCOLLISION_FEEDBACK_ROUND: {round_idx + 1}\n"
        new_feature_sources, collision_payload, output_path = generate_features(
            prompt, start_index, round_idx + 1
        )
        collision_payloads.append(collision_payload)
        collision_output_path = output_path

        if not new_feature_sources:
            logging.info("No new features generated; stopping feedback loop.")
            break
        X_before_append = X
        X, start_index = _append_new_features_from_sources(
            X,
            programs_sa,
            program_prior_log_probs,
            dsl_functions,
            new_feature_sources,
            examples,
            start_index=start_index,
            collision_loop_idx=round_idx + 1,
            prior_version=prior_version,
            prior_beta=prior_beta,
        )
        X_new = X[:, -len(new_feature_sources) :]
        repair_feature_stats = _collision_repair_feature_stats(
            round_idx=round_idx + 1,
            X_existing=X_before_append,
            X_new=X_new,
            new_feature_sources=new_feature_sources,
        )
        repair_feature_stats_history.append(repair_feature_stats)
        if collision_output_path is not None:
            _write_collision_repair_feature_stats(
                collision_output_path,
                repair_feature_stats_history,
            )
        logging.info(
            "Collision feedback appended %d features without redundant-feature "
            "filtering. Repair stats: constant=%d duplicate_existing=%d "
            "duplicate_within_batch=%d new_active_distinction=%d.",
            len(new_feature_sources),
            int(repair_feature_stats["constant"]),
            int(repair_feature_stats["duplicate_existing"]),
            int(repair_feature_stats["duplicate_within_repair_batch"]),
            int(repair_feature_stats["new_active_distinction"]),
        )
        col_nnz = np.asarray(X.getnnz(axis=0)).ravel()
        summary_before = summarize_collision_groups(collision_groups)
        collision_groups = log_feature_collisions(
            X,
            y,
            examples,
            bucket_mode=collision_bucket_mode,
        )
        summary_after = summarize_collision_groups(collision_groups)
        if _collision_summary_is_flat(summary_before, summary_after):
            flat_round_streak += 1
        else:
            flat_round_streak = 0
        collision_round_metrics.append(
            _collision_round_metric(
                round_idx=round_idx + 1,
                stage="post_feedback_filter",
                collision_groups=collision_groups,
                X=X,
                generated_feature_count=len(new_feature_sources),
            )
        )
        collision_round_metrics[-1]["repair_feature_stats"] = {
            key: repair_feature_stats[key]
            for key in (
                "added_features",
                "constant",
                "duplicate_existing",
                "duplicate_within_repair_batch",
                "new_active_distinction",
            )
        }
        num_collisions_after = len(collision_groups) if collision_groups else 0
        if record_attempt_summary is not None:
            record_attempt_summary(
                round_idx + 1,
                collision_payload,
                num_collisions,
                num_collisions_after,
            )
        logging.info(
            "Collision groups after feedback round %d: %d",
            round_idx + 1,
            num_collisions_after,
        )
    return (
        X,
        programs_sa,
        program_prior_log_probs,
        collision_payloads,
        collision_output_path,
        col_nnz,
        collision_round_metrics,
    )
