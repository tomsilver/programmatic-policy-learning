"""Collision feedback helpers for LPP."""

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.sparse import hstack

from programmatic_policy_learning.approaches.lpp_utils.lpp_feature_source_utils import (
    _parse_py_feature_sources,
)
from programmatic_policy_learning.approaches.lpp_utils.lpp_split_matrix_utils import (
    filter_constant_features,
)
from programmatic_policy_learning.approaches.lpp_utils.utils import (
    log_feature_collisions,
)
from programmatic_policy_learning.data.dataset import run_programs_on_examples
from programmatic_policy_learning.dsl.state_action_program import StateActionProgram
from programmatic_policy_learning.learning.prior_calculation import (
    priors_from_features,
    priors_from_features_v2,
)

_filter_constant_features = filter_constant_features


def _append_new_features_from_sources(
    X: Any,
    programs_sa: list[StateActionProgram],
    program_prior_log_probs: list[float] | None,
    dsl_functions: dict[str, Any],
    new_feature_sources: list[str],
    examples: list[tuple[np.ndarray, tuple[int, int]]],
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


def _run_collision_feedback_loop(
    *,
    collision_groups: list[dict[str, Any]],
    examples: list[tuple[np.ndarray, tuple[int, int]]],
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
        [list[dict[str, Any]], list[tuple[np.ndarray, tuple[int, int]]]], str | None
    ],
    prior_version: str = "uniform",
    prior_beta: float = 1.0,
) -> tuple[
    Any,
    list[StateActionProgram],
    list[float] | None,
    list[dict[str, Any]],
    Path | None,
    np.ndarray,
]:
    """Run collision-repair rounds by generating and appending new features."""
    collision_payloads: list[dict[str, Any]] = []
    collision_output_path: Path | None = None
    col_nnz = np.asarray(X.getnnz(axis=0)).ravel()
    for round_idx in range(max_rounds):
        num_collisions = len(collision_groups) if collision_groups else 0
        if num_collisions <= target_collisions:
            logging.info(
                "Collision feedback stopping: %d <= target %d.",
                num_collisions,
                target_collisions,
            )
            break
        prompt = make_prompt(collision_groups, examples)
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
        X, programs_sa, program_prior_log_probs, col_nnz = _filter_constant_features(
            X, programs_sa, program_prior_log_probs, round_idx=round_idx + 1
        )
        collision_groups = log_feature_collisions(X, y, examples)
        logging.info(
            "Collision groups after feedback round %d: %d",
            round_idx + 1,
            len(collision_groups) if collision_groups else 0,
        )
    return (
        X,
        programs_sa,
        program_prior_log_probs,
        collision_payloads,
        collision_output_path,
        col_nnz,
    )
