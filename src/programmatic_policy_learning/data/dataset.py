"""Dataset creation and processing utilities for programmatic policy
learning."""

import hashlib
import inspect
import logging
import multiprocessing
import os
import pickle
from importlib import import_module
from pathlib import Path
from typing import Any, Sequence, TypeVar

import cloudpickle
import numpy as np
from scipy.sparse import lil_matrix, vstack

from programmatic_policy_learning.data.demo_types import Trajectory
from programmatic_policy_learning.dsl.state_action_program import (
    StateActionProgram,
    set_dsl_functions,
)
from programmatic_policy_learning.utils.action_quantization import (
    Motion2DActionQuantizer,
)
from programmatic_policy_learning.utils.cache_utils import (
    cache_single_output,
    load_single_cache_output,
    manage_cache,
)
from programmatic_policy_learning.utils.grid_validation import require_grid_state_action


def allowed_cpus() -> int:
    """Determine the number of CPUs available for use."""
    # Use SLURM allocation if available
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        return int(slurm_cpus)
    # Otherwise fall back to system count
    return multiprocessing.cpu_count()


Coord = tuple[int, int]
ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")
CONTINUOUS_NEGATIVE_K = 10
CONTINUOUS_NEGATIVE_NOISE_SCALE = 0.2
GRID_NEGATIVE_K = 30
GRID_LOCAL_RADIUS = 2


def compute_cost_sensitive_bucket_weights(
    expert_bucket: Sequence[int],
    candidate_buckets: Sequence[Sequence[int]],
    *,
    beta_pos: float = 1.0,
    beta_neg: float = 1.0,
    alpha: float = 1.0,
    lambda_per_dim: Sequence[float] | None = None,
) -> np.ndarray:
    """Compute normalized cost-sensitive binary weights over quantized actions.

    Weights follow:
    - positive (expert bucket): beta_pos
    - negatives: beta_neg * g(cost) / sum(g(cost_negatives))

    with g(cost)=1+alpha*cost and
    cost = sum_d lambda_d * |i_d - i_d^*|.
    """
    expert = np.asarray(expert_bucket, dtype=int).reshape(-1)
    if expert.size == 0:
        raise ValueError("expert_bucket cannot be empty.")

    if beta_pos < 0.0 or beta_neg < 0.0:
        raise ValueError("beta_pos and beta_neg must be non-negative.")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")

    if lambda_per_dim is None:
        lambdas = np.ones_like(expert, dtype=float)
    else:
        lambdas = np.asarray(lambda_per_dim, dtype=float).reshape(-1)
        if lambdas.size != expert.size:
            raise ValueError(
                "lambda_per_dim length must match expert_bucket dimension: "
                f"got {lambdas.size}, expected {expert.size}."
            )
    if np.any(lambdas < 0.0):
        raise ValueError("All lambda_per_dim values must be non-negative.")

    if len(candidate_buckets) == 0:
        raise ValueError("candidate_buckets cannot be empty.")

    weights = np.zeros(len(candidate_buckets), dtype=float)
    neg_scores: list[float] = []
    neg_indices: list[int] = []
    positives = 0

    for idx, candidate_bucket in enumerate(candidate_buckets):
        cand = np.asarray(candidate_bucket, dtype=int).reshape(-1)
        if cand.size != expert.size:
            raise ValueError(
                "All candidate bucket dimensions must match expert_bucket: "
                f"got {cand.size}, expected {expert.size}."
            )

        if np.array_equal(cand, expert):
            weights[idx] = float(beta_pos)
            positives += 1
            continue

        dist = np.abs(cand - expert).astype(float)
        cost = float(np.dot(lambdas, dist))
        neg_scores.append(1.0 + alpha * cost)
        neg_indices.append(idx)

    if positives != 1:
        raise ValueError(
            "candidate_buckets must contain exactly one expert bucket; "
            f"found {positives}."
        )

    if neg_indices:
        denom = float(np.sum(neg_scores))
        if denom <= 0.0:
            raise ValueError("Invalid negative score normalization denominator.")
        for idx, score in zip(neg_indices, neg_scores):
            weights[idx] = float(beta_neg) * float(score) / denom

    return weights


def _coerce_action_like(template: Any, action_arr: np.ndarray) -> Any:
    """Convert sampled array back to the demonstrated action's container
    type."""
    if isinstance(template, np.ndarray):
        return action_arr.astype(template.dtype, copy=False)
    if isinstance(template, tuple):
        return tuple(float(x) for x in action_arr.tolist())
    if isinstance(template, list):
        return [float(x) for x in action_arr.tolist()]
    if isinstance(template, (int, float, np.integer, np.floating)):
        return float(action_arr.reshape(-1)[0])
    raise TypeError(
        "Unsupported continuous action type for sampling negatives: "
        f"{type(template)!r}"
    )


def sample_negative_actions_continuous(  # TODOO: can be simpler
    expert_action: ActT,
    *,
    K: int = 10,
    noise_scale: float = 0.2,
    action_low: Sequence[float] | np.ndarray | None = None,
    action_high: Sequence[float] | np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> list[ActT]:
    """Sample continuous negative actions by perturbing the expert action."""
    if K <= 0:
        return []
    if rng is None:
        rng = np.random.default_rng(0)
    base = np.asarray(expert_action, dtype=float)
    if base.ndim == 0:
        base = base.reshape(1)
    low_arr = np.asarray(action_low, dtype=float) if action_low is not None else None
    high_arr = np.asarray(action_high, dtype=float) if action_high is not None else None
    if (low_arr is None) != (high_arr is None):
        raise ValueError("action_low and action_high must be provided together.")
    if low_arr is not None and high_arr is not None:
        if low_arr.shape != base.shape or high_arr.shape != base.shape:
            raise ValueError(
                "continuous action bounds shape mismatch: "
                f"base={base.shape}, low={low_arr.shape}, high={high_arr.shape}"
            )
        if np.any(low_arr > high_arr):
            raise ValueError(
                "Each continuous action lower bound must be <= upper bound."
            )

    sampled: list[ActT] = []
    for _ in range(K * 5):
        candidate = base + rng.normal(0.0, noise_scale, size=base.shape)
        if low_arr is not None and high_arr is not None:
            candidate = np.clip(candidate, low_arr, high_arr)
        if np.allclose(candidate, base):
            continue
        sampled.append(_coerce_action_like(expert_action, candidate))
        if len(sampled) >= K:
            break

    return sampled


def sample_manual_negative_actions_continuous(
    expert_action: ActT,
    *,
    action_low: Sequence[float] | np.ndarray | None = None,
    action_high: Sequence[float] | np.ndarray | None = None,
) -> list[ActT]:
    """Build a fixed set of manual continuous negatives from the expert action.

    The generated variants are:
    1. wrong_x_sign
    2. wrong_y_sign
    3. wrong_both_signs
    4. correct_x_wrong_y
    5. wrong_x_correct_y
    6. x_only_partial_correction
    7. y_only_partial_correction
    8. too_small_same_direction
    9. too_large_same_direction
    10. misaligned_same_quadrant
    """
    base = np.asarray(expert_action, dtype=float)
    if base.ndim == 0:
        base = base.reshape(1)
    if base.shape[0] < 2:
        raise ValueError(
            "Manual continuous negative sampling requires at least 2 action dimensions."
        )

    low_arr = np.asarray(action_low, dtype=float) if action_low is not None else None
    high_arr = np.asarray(action_high, dtype=float) if action_high is not None else None
    if (low_arr is None) != (high_arr is None):
        raise ValueError("action_low and action_high must be provided together.")
    if low_arr is not None and high_arr is not None:
        if low_arr.shape != base.shape or high_arr.shape != base.shape:
            raise ValueError(
                "continuous action bounds shape mismatch: "
                f"base={base.shape}, low={low_arr.shape}, high={high_arr.shape}"
            )

    dx = float(base[0])
    dy = float(base[1])
    suffix = base[2:].copy()
    zero_suffix = np.zeros_like(suffix)
    half_dx = 0.5 * dx
    half_dy = 0.5 * dy
    large_dx = 1.5 * dx
    large_dy = 1.5 * dy

    candidates = [
        # wrong_x_sign
        np.concatenate([np.array([-dx, dy]), zero_suffix]),
        # wrong_y_sign
        np.concatenate([np.array([dx, -dy]), zero_suffix]),
        # wrong_both_signs
        np.concatenate([np.array([-dx, -dy]), zero_suffix]),
        # correct_x_wrong_y
        np.concatenate([np.array([dx, -half_dy]), zero_suffix]),
        # wrong_x_correct_y
        np.concatenate([np.array([-half_dx, dy]), zero_suffix]),
        # x_only_partial_correction
        np.concatenate([np.array([half_dx, 0.0]), zero_suffix]),
        # y_only_partial_correction
        np.concatenate([np.array([0.0, half_dy]), zero_suffix]),
        # too_small_same_direction
        np.concatenate([np.array([half_dx, half_dy]), zero_suffix]),
        # too_large_same_direction
        np.concatenate([np.array([large_dx, large_dy]), zero_suffix]),
        # misaligned_same_quadrant
        np.concatenate([np.array([dx, half_dy]), zero_suffix]),
    ]

    sampled: list[ActT] = []
    seen: set[tuple[float, ...]] = set()
    for candidate in candidates:
        if low_arr is not None and high_arr is not None:
            candidate = np.clip(candidate, low_arr, high_arr)
        key = tuple(float(x) for x in candidate.tolist())
        if key in seen or np.allclose(candidate, base):
            continue
        seen.add(key)
        sampled.append(_coerce_action_like(expert_action, candidate))
    return sampled


def _normalized_weights(*weights: float) -> list[float]:
    arr = [max(0.0, float(w)) for w in weights]
    total = sum(arr)
    if total <= 0.0:
        return [0.0 for _ in arr]
    return [w / total for w in arr]


def _allocate_counts(total: int, weights: list[float]) -> list[int]:
    if total <= 0 or len(weights) == 0:
        return [0 for _ in weights]
    raw = [total * w for w in weights]
    counts = [int(np.floor(x)) for x in raw]
    rem = total - sum(counts)
    if rem > 0:
        frac = sorted(
            ((raw[i] - counts[i], i) for i in range(len(weights))),
            reverse=True,
        )
        for _, idx in frac[:rem]:
            counts[idx] += 1
    return counts


def _sample_grid_mixture_negatives(
    state_grid: np.ndarray,
    expert_action: Coord,
    *,
    k_total: int,
    local_radius: int,
    w_local: float,
    w_struct: float,
    w_random: float,
    rng: np.random.Generator,
) -> list[Coord]:
    """Sample grid negatives from local/struct/random mixture."""
    if k_total <= 0:
        return []
    h = int(state_grid.shape[0])
    w = int(state_grid.shape[1])
    er, ec = expert_action

    all_coords = [(r, c) for r in range(h) for c in range(w) if (r, c) != (er, ec)]
    if not all_coords:
        return []

    token_expert = state_grid[er, ec]
    local_candidates = [
        (r, c)
        for r, c in all_coords
        if abs(r - er) + abs(c - ec) <= max(1, int(local_radius))
    ]
    struct_candidates = [
        (r, c) for r, c in all_coords if state_grid[r, c] == token_expert
    ]
    random_candidates = list(all_coords)

    weights = _normalized_weights(w_local, w_struct, w_random)
    if sum(weights) == 0.0:
        weights = [0.0, 0.0, 1.0]
    k_local, k_struct, k_random = _allocate_counts(k_total, weights)

    picked: list[Coord] = []
    picked_set: set[Coord] = set()

    def _take(candidates: list[Coord], k: int) -> None:
        if k <= 0:
            return
        available = [rc for rc in candidates if rc not in picked_set]
        if not available:
            return
        idx = rng.permutation(len(available))
        for j in idx[: min(k, len(available))]:
            rc = available[int(j)]
            picked.append(rc)
            picked_set.add(rc)

    _take(local_candidates, k_local)
    _take(struct_candidates, k_struct)
    _take(random_candidates, k_random)

    if len(picked) < k_total:
        remaining = [rc for rc in all_coords if rc not in picked_set]
        if remaining:
            idx = rng.permutation(len(remaining))
            for j in idx[: min(k_total - len(picked), len(remaining))]:
                picked.append(remaining[int(j)])

    return picked[:k_total]


def _sample_continuous_mixture_negatives(
    expert_action: ActT,
    *,
    k_total: int,
    local_noise_scale: float,
    action_low: Sequence[float] | np.ndarray | None,
    action_high: Sequence[float] | np.ndarray | None,
    w_local: float,
    w_uniform: float,
    w_traj: float,
    rng: np.random.Generator,
) -> list[ActT]:
    """Sample continuous negatives from local/uniform mixture.

    Note: trajectory-based negatives are not implemented yet; `w_traj` is
    redistributed over local/uniform.
    """
    _ = w_traj  # trajectory negatives are intentionally disabled for now
    if k_total <= 0:
        return []
    base = np.asarray(expert_action, dtype=float)
    if base.ndim == 0:
        base = base.reshape(1)

    low_arr = np.asarray(action_low, dtype=float) if action_low is not None else None
    high_arr = np.asarray(action_high, dtype=float) if action_high is not None else None
    has_bounds = low_arr is not None and high_arr is not None

    if has_bounds:
        assert low_arr is not None and high_arr is not None
        if low_arr.shape != base.shape or high_arr.shape != base.shape:
            raise ValueError(
                "continuous action bounds shape mismatch: "
                f"base={base.shape}, low={low_arr.shape}, high={high_arr.shape}"
            )
        if np.any(low_arr > high_arr):
            raise ValueError(
                "Each continuous action lower bound must be <= upper bound."
            )

    # traj component intentionally disabled for now.
    weights = _normalized_weights(w_local, w_uniform if has_bounds else 0.0, 0.0)
    if sum(weights) == 0.0:
        weights = [1.0, 0.0, 0.0]
    k_local, k_uniform, _ = _allocate_counts(k_total, weights)

    sampled: list[ActT] = []
    sampled_vecs: list[np.ndarray] = []

    def _add_candidate(candidate: np.ndarray) -> None:
        if np.allclose(candidate, base):
            return
        for prev in sampled_vecs:
            if np.allclose(candidate, prev):
                return
        sampled_vecs.append(candidate.copy())
        sampled.append(_coerce_action_like(expert_action, candidate))

    # Local perturbations around expert action.
    for _ in range(max(1, k_local) * 6):
        if len(sampled) >= k_local:
            break
        candidate = base + rng.normal(0.0, local_noise_scale, size=base.shape)
        if has_bounds:
            candidate = np.clip(candidate, low_arr, high_arr)
        _add_candidate(candidate)

    # Uniform negatives across valid action box.
    if has_bounds:
        assert low_arr is not None and high_arr is not None
        local_target = len(sampled)
        target_total = local_target + k_uniform
        for _ in range(max(1, k_uniform) * 6):
            if len(sampled) >= target_total:
                break
            candidate = rng.uniform(low_arr, high_arr)
            _add_candidate(candidate)

    # Backfill with local perturbations if needed.
    while len(sampled) < k_total:
        candidate = base + rng.normal(0.0, local_noise_scale, size=base.shape)
        if has_bounds:
            assert low_arr is not None and high_arr is not None
            candidate = np.clip(candidate, low_arr, high_arr)
        _add_candidate(candidate)
        if len(sampled_vecs) > (k_total * 20):
            break
    return sampled[:k_total]


def extract_examples_from_demonstration_item(
    demonstration_item: tuple[ObsT, ActT],
    *,
    negative_sampling: dict[str, Any] | None = None,
    action_mode: str = "discrete",
    compute_sample_weights: bool = False,
) -> tuple[
    list[tuple[ObsT, ActT]],
    list[tuple[ObsT, ActT]],
    np.ndarray,
]:
    """Convert a demonstrated (state, action) into positive and negative
    classification data.

    Parameters
    ----------
    demonstrations : (ObsT, ActT)
        A state, action pair.
    compute_sample_weights : bool
        If True, compute cost-sensitive sample weights for training.

    Returns
    -------
    positive_examples : [(ObsT, ActT)]
        A list with just the input state, action pair (for convenience).
    negative_examples : [(ObsT, ActT)]
        A list with negative examples of state, actions.
    sample_weights : np.ndarray
        Array of shape (len(positive_examples) + len(negative_examples),)
        with sample weights aligned to rows. All ones if compute_sample_weights=False.
    """
    state, action = demonstration_item

    positive_examples: list[tuple[ObsT, ActT]] = [(state, action)]
    negative_examples: list[tuple[ObsT, ActT]] = []
    sampling_cfg = dict(negative_sampling or {})
    sample_weights = np.ones(1, dtype=float)  # Default: single positive with weight=1

    if action_mode == "continuous":
        action_low = sampling_cfg.get("action_low")
        action_high = sampling_cfg.get("action_high")
        if action_low is None or action_high is None:
            raise ValueError(
                "Continuous action_mode requires action_low/action_high bounds for "
                "quantized expansion."
            )

        cfg_cont = dict(sampling_cfg.get("continuous", {}))
        bucket_counts_cfg = cfg_cont.get("bucket_counts", 5)

        base = np.asarray(action, dtype=float)
        if base.ndim == 0:
            base = base.reshape(1)
        if base.shape[0] < 2:
            raise ValueError(
                "Continuous quantized expansion requires at least 2 action "
                "dimensions (dx, dy)."
            )

        low_arr = np.asarray(action_low, dtype=float)
        high_arr = np.asarray(action_high, dtype=float)
        if low_arr.shape != base.shape or high_arr.shape != base.shape:
            raise ValueError(
                "continuous action bounds shape mismatch in quantized expansion: "
                f"base={base.shape}, low={low_arr.shape}, high={high_arr.shape}"
            )

        quantizer = Motion2DActionQuantizer.from_bounds(
            low_arr[
                :2
            ],  # TODO: currently only quantizing the first 2 dimensions for negative sampling; can be extended if needed
            high_arr[:2],
            bucket_counts=bucket_counts_cfg,
        )
        expert_bucket = quantizer.quantize(base[:2])

        quantized_expert = base.copy()
        quantized_expert[:2] = quantizer.dequantize(expert_bucket)
        if quantized_expert.shape[0] > 2:
            quantized_expert[2:] = 0.0
        quantized_expert = np.clip(quantized_expert, low_arr, high_arr)
        positive_examples = [(state, _coerce_action_like(action, quantized_expert))]

        neg_actions: list[ActT] = []
        all_buckets: list[tuple[int, ...]] = []
        for bucket in quantizer.all_bucket_indices():
            all_buckets.append(bucket)
            if bucket == expert_bucket:
                continue
            candidate = base.copy()
            candidate[:2] = quantizer.dequantize(bucket)
            if candidate.shape[0] > 2:
                candidate[2:] = 0.0
            candidate = np.clip(candidate, low_arr, high_arr)
            neg_actions.append(_coerce_action_like(action, candidate))
        for neg_action in neg_actions:
            negative_examples.append((state, neg_action))

        # Keep weight order aligned with row order: [positive] + [negatives].
        aligned_buckets = [expert_bucket] + [
            bucket for bucket in all_buckets if bucket != expert_bucket
        ]

        # Compute sample weights if requested
        if compute_sample_weights:
            weight_cfg = dict(cfg_cont.get("weight_config", {}))
            beta_pos = weight_cfg.get("beta_pos", 1.0)
            beta_neg = weight_cfg.get("beta_neg", 1.0)
            alpha = weight_cfg.get("alpha", 1.0)
            lambda_per_dim = weight_cfg.get("lambda_per_dim", None)
            sample_weights = compute_cost_sensitive_bucket_weights(
                expert_bucket,
                aligned_buckets,
                beta_pos=beta_pos,
                beta_neg=beta_neg,
                alpha=alpha,
                lambda_per_dim=lambda_per_dim,
            )
        else:
            # Default uniform weights: one for positive, one per negative
            sample_weights = np.ones(1 + len(neg_actions), dtype=float)

        return positive_examples, negative_examples, sample_weights
    if action_mode != "discrete":
        raise ValueError(f"Unknown action_mode: {action_mode!r}")

    state_grid, action_grid = require_grid_state_action(
        state,
        action,
        context="Discrete negative expansion",
    )

    # `negative_sampling.enabled` only controls discrete-mode subsampling.
    discrete_sampling_enabled = bool(sampling_cfg.get("enabled", False))
    if discrete_sampling_enabled:
        cfg_grid = dict(sampling_cfg.get("discrete", {}))
        k_total = int(cfg_grid.get("K", GRID_NEGATIVE_K))
        local_radius = int(cfg_grid.get("local_radius", GRID_LOCAL_RADIUS))
        w_local = float(cfg_grid.get("w_local", 0.5))
        w_struct = float(cfg_grid.get("w_struct", 0.3))
        w_random = float(cfg_grid.get("w_random", 0.2))
        rng_np = np.random.default_rng(0)
        neg_coords = _sample_grid_mixture_negatives(
            state_grid,
            action_grid,
            k_total=k_total,
            local_radius=local_radius,
            w_local=w_local,
            w_struct=w_struct,
            w_random=w_random,
            rng=rng_np,
        )
        for rc in neg_coords:
            negative_examples.append((state, rc))  # type: ignore[arg-type]
    else:
        for r in range(state_grid.shape[0]):
            for c in range(state_grid.shape[1]):
                if (r, c) == action_grid:
                    continue
                negative_examples.append((state, (r, c)))  # type: ignore[arg-type]

    # For discrete mode, return uniform weights (1.0 for each example)
    # TODOO: can also pass "balanced" to dt and ignore manual weight computation here, since all negatives are equally weighted anyway
    sample_weights = np.ones(1 + len(negative_examples), dtype=float)
    return positive_examples, negative_examples, sample_weights


def extract_examples_from_demonstration(
    demonstration: Trajectory[ObsT, ActT],
    *,
    negative_sampling: dict[str, Any] | None = None,
    action_mode: str = "discrete",
    compute_sample_weights: bool = False,
) -> tuple[list[tuple[ObsT, ActT]], list[tuple[ObsT, ActT]], np.ndarray]:
    """Convert demonstrated (state, action)s into positive and negative
    classification data.

    Parameters
    ----------
    demonstrations : [(ObsT, ActT)]
        State, action pairs from a trajectory.
    compute_sample_weights : bool
        If True, compute cost-sensitive sample weights. Only applies to continuous mode.

    Returns
    -------
    positive_examples : [(ObsT, ActT)]
        A list with the input state, action pairs.
    negative_examples : [(ObsT, ActT)]
        A list with negative examples of state, actions.
    sample_weights : np.ndarray
        Shape (total_examples,) where total_examples = len(pos) + len(neg).
        Weights aligned with rows: positives first, then negatives.
    """
    positive_examples: list[tuple[ObsT, ActT]] = []
    negative_examples: list[tuple[ObsT, ActT]] = []
    pos_weights: list[float] = []
    neg_weights: list[float] = []

    for demonstration_item in demonstration.steps:
        demo_positive_examples, demo_negative_examples, demo_weights = (
            extract_examples_from_demonstration_item(
                demonstration_item,
                negative_sampling=negative_sampling,
                action_mode=action_mode,
                compute_sample_weights=compute_sample_weights,
            )
        )
        positive_examples.extend(demo_positive_examples)
        negative_examples.extend(demo_negative_examples)
        if demo_weights.shape[0] != (
            len(demo_positive_examples) + len(demo_negative_examples)
        ):
            raise ValueError(
                "Per-item sample_weights length must match per-item examples."
            )
        pos_weights.extend(demo_weights[: len(demo_positive_examples)].tolist())
        neg_weights.extend(demo_weights[len(demo_positive_examples) :].tolist())

    combined_weights = np.asarray(pos_weights + neg_weights, dtype=float)
    return positive_examples, negative_examples, combined_weights


def _split_dsl(dsl: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Return (base, module_map).

    base is pickleable; module_map is name->import_path.
    """
    base, module_map = {}, {}
    for k, v in dsl.items():
        if inspect.ismodule(v):
            module_map[k] = v.__name__
        else:
            base[k] = v
    # Remove __builtins__ if present
    if "__builtins__" in base:
        del base["__builtins__"]
    return base, module_map


def eval_program_fn(s: np.ndarray, a: tuple[int, int], prog: str) -> bool | None:
    """Evaluate a program on a state-action pair."""
    try:
        result = eval("lambda s, a: " + prog, _WORKER_DSL)(s, a)
        logging.info(f"Program: {prog}, Input: (s={s}, a={a}), Result: {result}")
        return result
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.info(f"Program: {prog}, Input: (s={s}, a={a}), Exception: {e}")
        return None


# Global worker states
_WORKER_DSL = None
_WORKER_PROGRAMS = None

CACHE_DIR = "cache"


def _cache_key_run_all_programs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    cache_schema_version = "v4"
    base_class_name = str(args[0])
    demo_number = int(args[1])
    programs = args[2]
    program_count = len(programs) if programs is not None else 0
    seed = kwargs.get("seed")
    seed_tag = f"s{seed}" if seed is not None else "snone"
    demos_included = kwargs.get("demos_included")
    demos_tag = "none"
    if demos_included is not None:
        demos_list = list(demos_included)
        demos_tag = "d" + "-".join(str(d) for d in demos_list)
    negative_sampling = kwargs.get("negative_sampling")
    sampling_sig = "none"
    if negative_sampling:
        sampling_sig = hashlib.sha1(
            repr(negative_sampling).encode("utf-8")
        ).hexdigest()[:10]
    offline_path_name = kwargs.get("offline_path_name")
    offline_tag = "none"
    if offline_path_name:
        offline_tag = Path(str(offline_path_name)).name
    split_tag = kwargs.get("split_tag")
    split_part = "splitnone"
    if split_tag:
        split_part = f"split{split_tag}"
    return (
        f"{base_class_name}-demo{demo_number}-n{program_count}-"
        f"demos{demos_tag}-{seed_tag}-ns{sampling_sig}-offline{offline_tag}-"
        f"{split_part}-{cache_schema_version}"
    )


def worker_init(
    dsl_blob: bytes, module_map: dict[str, str], program_batch: list[str]
) -> None:
    """Set up the worker once.

    Loads the DSL, reimports modules, and compiles the given program
    batch. Runs only once per process before handling any examples.
    """
    base = cloudpickle.loads(dsl_blob)
    for name, modpath in module_map.items():
        base[name] = import_module(modpath)
    set_dsl_functions(base)

    from programmatic_policy_learning.dsl.state_action_program import (  # pylint: disable=import-outside-toplevel
        DSL_FUNCTIONS,
    )

    global _WORKER_DSL, _WORKER_PROGRAMS  # pylint: disable=global-statement
    _WORKER_DSL = DSL_FUNCTIONS
    _WORKER_PROGRAMS = [
        eval("lambda s, a: " + prog, DSL_FUNCTIONS) for prog in program_batch
    ]


def worker_eval_example(fn_input: tuple[ObsT, ActT]) -> list[bool]:
    """Run all precompiled programs on one (state, action) example.

    Uses the DSL and program_batch already set up by worker_init.
    """
    s, a = fn_input

    if _WORKER_PROGRAMS is None:
        raise RuntimeError("_WORKER_PROGRAMS is not initialized.\
            Ensure worker_init is called before using worker_eval_example.")

    results = []
    for f in _WORKER_PROGRAMS:
        try:
            results.append(f(s, a))
        except Exception as e:  # pylint: disable=broad-exception-caught
            results.append(None)
            logging.info(f"Error type: {type(e).__name__}\n" f"Error message: {e}")
    return results


@manage_cache(
    CACHE_DIR,
    [".npz", ".pkl", ".pkl", ".pkl"],
    key_fn=_cache_key_run_all_programs,
)
def run_all_programs_on_single_demonstration(
    base_class_name: str,
    demo_number: int,
    programs: list[StateActionProgram] | list[str],
    demo_traj: Trajectory[ObsT, ActT],
    dsl_functions: dict,
    *,
    negative_sampling: dict[str, Any] | None = None,
    return_examples: bool = False,
    offline_path_name: str | None = None,  # pylint: disable=unused-argument
    demos_included: Sequence[int] | None = None,  # pylint: disable=unused-argument
    split_tag: str | None = None,  # pylint: disable=unused-argument
    action_mode: str = "discrete",
    seed: int | None = None,  # pylint: disable=unused-argument
    program_interval: int = 1000,  # unused in this fast path; keep for compat  # pylint: disable=unused-argument
) -> tuple[Any, np.ndarray, list[tuple[ObsT, ActT]] | None, np.ndarray]:
    """Run all programs on a single demonstration and return feature matrix and
    labels."""
    logging.info(f"Running all programs on {base_class_name}, {demo_number}")
    positive_examples, negative_examples, sample_weights = (
        extract_examples_from_demonstration(
            demo_traj,
            negative_sampling=negative_sampling,
            action_mode=action_mode,
            compute_sample_weights=False,  # Step 2 prep: weights returned but not yet used
        )
    )

    fn_inputs = positive_examples + negative_examples
    y: list[int] = [1] * len(positive_examples) + [0] * len(negative_examples)
    base_dsl, module_map = _split_dsl(dsl_functions)

    try:
        dsl_blob = cloudpickle.dumps(base_dsl)
        cloudpickle.loads(dsl_blob)  # Test deserialization
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"Failed to serialize/deserialize DSL: {e}") from e

    # Extract program strings (don’t pickle heavy objects repeatedly)
    program_strs = [
        (p.program if isinstance(p, StateActionProgram) else str(p)) for p in programs
    ]

    num_data = len(fn_inputs)
    num_programs = len(program_strs)

    X = lil_matrix((num_data, num_programs), dtype=bool)

    # Combine the context initialization into a single block to avoid redefinition
    try:

        ctx = multiprocessing.get_context("spawn")  # type: ignore[assignment]
        # linux
    except (ValueError, RuntimeError):
        ctx = multiprocessing.get_context("fork")  # type: ignore[assignment]
        # macOS/Windows fallback (spawn)
    num_workers = allowed_cpus()
    num_workers = max(1, min(num_workers, len(fn_inputs)))
    for p_start in range(0, num_programs, program_interval):
        p_end = min(p_start + program_interval, num_programs)
        program_batch = program_strs[p_start:p_end]
        with ctx.Pool(
            processes=num_workers,
            initializer=worker_init,
            initargs=(dsl_blob, module_map, program_batch),
            maxtasksperchild=100,
        ) as pool:
            results_iter = pool.imap(worker_eval_example, fn_inputs, chunksize=64)
            batch_rows_list = list(results_iter)
        batch_matrix = np.array(batch_rows_list, dtype=bool)
        X[:, p_start:p_end] = batch_matrix
    examples = fn_inputs if return_examples else None
    return X.tocsr(), np.array(y, dtype=np.uint8), examples, sample_weights


def run_programs_on_examples(
    programs: list[StateActionProgram] | list[str],
    examples: list[tuple[ObsT, ActT]],
    dsl_functions: dict,
    *,
    program_interval: int = 1000,
    cache_dir: str | None = "cache",
    cache_key: str | None = None,
    feature_sources: list[str] | None = None,
    collision_loop_idx: int | None = None,
) -> Any:
    """Run programs on a fixed list of (state, action) examples.

    Returns a CSR sparse matrix with rows aligned to the given examples.
    """
    if not examples:
        return lil_matrix((0, 0), dtype=bool).tocsr()

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        if cache_key is None:
            hasher = hashlib.sha256()
            program_strs = [
                (p.program if isinstance(p, StateActionProgram) else str(p))
                for p in programs
            ]
            hasher.update("\n".join(program_strs).encode("utf-8"))
            hasher.update(pickle.dumps(examples, protocol=4))
            if feature_sources:
                hasher.update("\n".join(feature_sources).encode("utf-8"))
            if collision_loop_idx is not None:
                hasher.update(
                    f"collision_loop_idx={collision_loop_idx}".encode("utf-8")
                )
            cache_key = hasher.hexdigest()[:16]
        cache_file = os.path.join(
            cache_dir, f"run_programs_on_examples_{cache_key}_0.npz"
        )
        if os.path.isfile(cache_file):
            return load_single_cache_output(cache_file)

    base_dsl, module_map = _split_dsl(dsl_functions)

    try:
        dsl_blob = cloudpickle.dumps(base_dsl)
        cloudpickle.loads(dsl_blob)  # Test deserialization
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"Failed to serialize/deserialize DSL: {e}") from e

    program_strs = [
        (p.program if isinstance(p, StateActionProgram) else str(p)) for p in programs
    ]

    num_data = len(examples)
    num_programs = len(program_strs)
    X = lil_matrix((num_data, num_programs), dtype=bool)

    try:
        ctx = multiprocessing.get_context("spawn")  # type: ignore[assignment]
    except (ValueError, RuntimeError):
        ctx = multiprocessing.get_context("fork")  # type: ignore[assignment]
    num_workers = allowed_cpus()
    num_workers = max(1, min(num_workers, len(examples)))
    for p_start in range(0, num_programs, program_interval):
        p_end = min(p_start + program_interval, num_programs)
        program_batch = program_strs[p_start:p_end]
        with ctx.Pool(
            processes=num_workers,
            initializer=worker_init,
            initargs=(dsl_blob, module_map, program_batch),
            maxtasksperchild=100,
        ) as pool:
            results_iter = pool.imap(worker_eval_example, examples, chunksize=64)
            batch_rows_list = list(results_iter)
        batch_matrix = np.array(batch_rows_list, dtype=bool)
        X[:, p_start:p_end] = batch_matrix

    X_csr = X.tocsr()
    if cache_dir is not None:
        cache_single_output(X_csr, cache_file)
    return X_csr


def run_all_programs_on_demonstrations(
    base_class_name: str,
    demo_numbers: tuple[int, ...],
    programs: list,
    demo_dict: dict[int, Trajectory[ObsT, ActT]],
    dsl_functions: dict,
    *,
    negative_sampling: dict[str, Any] | None = None,
    return_examples: bool = False,
    offline_path_name: str | None = None,
    demos_included: Sequence[int] | None = None,
    split_tag: str | None = None,
    seed: int | None = None,
    action_mode: str = "discrete",
) -> tuple[
    Any | None, np.ndarray | None, list[tuple[ObsT, ActT]] | None, np.ndarray | None
]:
    """Run all programs on a set of demonstrations and aggregate results."""
    X, y = None, None
    examples_all: list[tuple[ObsT, ActT]] = []
    sample_weights_all: list[np.ndarray] = []
    for demo_number in demo_numbers:
        demo_X, demo_y, demo_examples, demo_sample_weights = (
            run_all_programs_on_single_demonstration(
                base_class_name,
                demo_number,
                programs,
                demo_dict[demo_number],
                dsl_functions,
                negative_sampling=negative_sampling,
                return_examples=return_examples,
                offline_path_name=offline_path_name,
                demos_included=demos_included,
                split_tag=split_tag,
                action_mode=action_mode,
                seed=seed,
            )
        )

        if X is None:
            X = demo_X
            y = demo_y
        else:
            X = vstack([X, demo_X])
            y = np.concatenate([y, demo_y])
        sample_weights_all.append(demo_sample_weights)
        if return_examples and demo_examples:
            examples_all.extend(demo_examples)
    sample_weights = (
        np.concatenate(sample_weights_all)
        if sample_weights_all
        else np.array([], dtype=float)
    )
    return X, y, (examples_all if return_examples else None), sample_weights
