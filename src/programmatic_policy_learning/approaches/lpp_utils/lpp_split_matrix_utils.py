"""Split and matrix utility helpers for the LPP approach."""

import hashlib
import logging
from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np

from programmatic_policy_learning.data.collect import get_demonstrations
from programmatic_policy_learning.data.dataset import (
    extract_examples_from_demonstration,
)
from programmatic_policy_learning.data.demo_types import Trajectory
from programmatic_policy_learning.dsl.state_action_program import StateActionProgram
ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")

def split_and_collect_demonstrations(
    *,
    env_factory: Any,
    expert: Any,
    demo_numbers: Sequence[int],
    val_frac: float | None,
    val_size: int | None,
    split_seed: int,
    split_strategy: str,
    preserve_ordering: bool,
    data_imbalance_cfg: dict[str, Any] | None,
    action_mode: str = "discrete",
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    Trajectory[ObsT, ActT],
    Trajectory[ObsT, ActT],
    dict[int, Trajectory[ObsT, ActT]],
    dict[int, Trajectory[ObsT, ActT]],
]:
    """Split demo IDs, collect train/val demonstrations, and log split
    stats."""
    train_demo_ids, val_demo_ids = split_dataset(
        demo_numbers,
        val_frac=val_frac,
        val_size=val_size,
        split_seed=split_seed,
        split_strategy=split_strategy,
        preserve_ordering=preserve_ordering,
    )

    if set(train_demo_ids).intersection(set(val_demo_ids)):
        raise AssertionError("train and val demo IDs are not disjoint.")

    demonstrations_train, demo_dict_train = get_demonstrations(
        env_factory,
        expert,
        demo_numbers=train_demo_ids,  # type: ignore[arg-type]
    )
    demonstrations_val = Trajectory[ObsT, ActT](steps=[])
    demo_dict_val: dict[int, Trajectory[ObsT, ActT]] = {}
    demo_dict_all = dict(demo_dict_train)
    if val_demo_ids:
        demonstrations_val, demo_dict_val = get_demonstrations(
            env_factory,
            expert,
            demo_numbers=val_demo_ids,  # type: ignore[arg-type]
        )
        demo_dict_all.update(demo_dict_val)

    # assert_state_disjointness(demo_dict_all, train_demo_ids, val_demo_ids)
    print(action_mode)
    train_states, train_expanded = count_states_and_expanded_examples(
        train_demo_ids,
        demo_dict_all,
        data_imbalance=data_imbalance_cfg,
        action_mode=action_mode,
    )
    val_states, val_expanded = count_states_and_expanded_examples(
        val_demo_ids,
        demo_dict_all,
        data_imbalance=data_imbalance_cfg,
        action_mode=action_mode,
    )
    logging.info(
        "Split stats | train: demos=%d states=%d expanded_examples=%d",
        len(train_demo_ids),
        train_states,
        train_expanded,
    )
    logging.info(
        "Split stats | val: demos=%d states=%d expanded_examples=%d",
        len(val_demo_ids),
        val_states,
        val_expanded,
    )
    return (
        train_demo_ids,
        val_demo_ids,
        demonstrations_train,
        demonstrations_val,
        demo_dict_train,
        demo_dict_val,
    )


def split_dataset(
    demo_numbers: Sequence[int],
    *,
    val_frac: float | None = None,
    val_size: int | None = None,
    split_seed: int = 0,
    split_strategy: str = "random",
    preserve_ordering: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split demo ids into train/val deterministically."""
    if split_strategy != "random":
        raise ValueError(f"Unsupported split_strategy: {split_strategy}")

    demo_ids: list[int] = []
    seen: set[int] = set()
    for d in demo_numbers:
        dd = int(d)
        if dd not in seen:
            seen.add(dd)
            demo_ids.append(dd)
    n = len(demo_ids)
    if n == 0:
        return tuple(), tuple()

    if val_size is not None:
        if val_size < 0 or val_size >= n:
            raise ValueError("val_size must be in [0, len(demo_numbers)-1].")
        n_val = int(val_size)
    else:
        frac = 0.0 if val_frac is None else float(val_frac)
        if frac < 0.0 or frac >= 1.0:
            raise ValueError("val_frac must be in [0.0, 1.0).")
        n_val = int(round(n * frac))

    if n_val <= 0:
        return tuple(demo_ids), tuple()

    work = list(demo_ids)
    if not preserve_ordering:
        rng = np.random.default_rng(split_seed)
        rng.shuffle(work)
    val_ids = tuple(work[:n_val])
    train_ids = tuple(work[n_val:])
    if len(train_ids) == 0:
        raise ValueError("Split produced empty train set.")
    return train_ids, val_ids


def _hash_state(obs: np.ndarray) -> str:
    arr = np.asarray(obs)
    hasher = hashlib.sha1()
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(str(arr.shape).encode("utf-8"))
    hasher.update(arr.tobytes())
    return hasher.hexdigest()


def count_states_and_expanded_examples(
    demo_ids: Sequence[int],
    demo_dict: dict[int, Any],
    *,
    data_imbalance: dict[str, Any] | None = None,
    action_mode: str = "discrete",
) -> tuple[int, int]:
    """Count states and expanded examples for a set of demo ids."""
    num_states = 0
    expanded = 0
    for demo_id in demo_ids:
        traj = demo_dict[int(demo_id)]
        num_states += len(traj.steps)
        pos, neg = extract_examples_from_demonstration(
            traj,
            data_imbalance=data_imbalance,
            action_mode=action_mode,
        )
        expanded += len(pos) + len(neg)
    return num_states, expanded


def assert_state_disjointness(
    demo_dict: dict[int, Any],
    train_ids: Sequence[int],
    val_ids: Sequence[int],
) -> None:
    """Ensure train/val do not share hashed states."""
    train_hashes: set[str] = set()
    val_hashes: set[str] = set()
    for demo_id in train_ids:
        for obs, _ in demo_dict[int(demo_id)].steps:
            train_hashes.add(_hash_state(obs))
    for demo_id in val_ids:
        for obs, _ in demo_dict[int(demo_id)].steps:
            val_hashes.add(_hash_state(obs))
    overlap = train_hashes.intersection(val_hashes)
    if overlap:
        raise AssertionError("train/val state leakage detected via state-hash overlap.")


def constant_feature_cols(X_csr: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return indices of all-zero/all-one columns and nnz per column."""
    n = X_csr.shape[0]
    col_nnz = np.asarray(X_csr.getnnz(axis=0)).ravel()
    all_zero = np.where(col_nnz == 0)[0]
    all_one = np.where(col_nnz == n)[0]
    return all_zero, all_one, col_nnz


def filter_constant_features(
    X: Any,
    programs_sa: list[StateActionProgram],
    program_prior_log_probs: list[float] | None,
    *,
    round_idx: int | None = None,
) -> tuple[Any, list[StateActionProgram], list[float] | None, np.ndarray]:
    """Drop all-zero/all-one feature columns and aligned metadata."""
    all_zero, all_one, col_nnz = constant_feature_cols(X)
    remove = np.unique(np.concatenate([all_zero, all_one]))
    if remove.size > 0:
        keep_mask = np.ones(X.shape[1], dtype=bool)
        keep_mask[remove] = False
        X = X[:, keep_mask]
        programs_sa = [p for i, p in enumerate(programs_sa) if keep_mask[i]]
        if program_prior_log_probs is not None:
            program_prior_log_probs = [
                lp for i, lp in enumerate(program_prior_log_probs) if keep_mask[i]
            ]
        if round_idx is None:
            logging.info("Filtered constant features. New X shape: %s", X.shape)
        else:
            logging.info(
                "Filtered constant features after feedback round %d. New X shape: %s",
                round_idx,
                X.shape,
            )
    return X, programs_sa, program_prior_log_probs, col_nnz
