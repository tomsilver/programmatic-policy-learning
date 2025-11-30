"""Tests for DSL partial evaluator."""

from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from programmatic_policy_learning.dsl.llm_primitives.dsl_evaluator import (
    evaluate_primitive,
)
from programmatic_policy_learning.dsl.primitives_sets.grid_v1 import (
    get_core_boolean_primitives,
    out_of_bounds,
)
from programmatic_policy_learning.envs.registry import EnvRegistry


def pattern_check(
    value_pattern: Any,
    neighbor_offset: tuple[int, int],
    cell: tuple[int, int] | None,
    obs: np.ndarray,
) -> bool:
    """A safe, simple primitive:

    - It is NOT constant (depends on neighbor value)
    - It is NOT equivalent to core primitives
    - It checks bounds properly
    """
    if cell is None:
        return False

    nr = cell[0] + neighbor_offset[0]
    nc = cell[1] + neighbor_offset[1]

    if out_of_bounds(nr, nc, obs.shape):
        return False

    return obs[nr, nc] == value_pattern


def always_true(
    cell: tuple[int, int], obs: np.ndarray  # pylint: disable=unused-argument
) -> bool:
    """A trivial predicate that always returns True.

    Useful for testing degeneracy and sanity checks.
    """
    return True


def mimic_cell_is_value(
    v: Any,
    cell: tuple[int, int],
    obs: np.ndarray,
) -> bool:
    """A simple mimic of cell_is_value(v).

    Returns True when obs[cell] equals `value`.
    Used to test L2 semantic similarity detection.
    """
    if cell is None or out_of_bounds(cell[0], cell[1], obs.shape):
        focus = None
    else:
        focus = obs[cell[0], cell[1]]

    return focus == v


def test_function_passes_both_filters() -> None:
    """
    Tests that the safe primitive:
    - L1: degeneracy_score is NOT 1.0 (not constant)
    - L2: It is NOT overly similar to any existing primitive
    """
    cfg: DictConfig = OmegaConf.create(
        {
            "provider": "ggg",
            "make_kwargs": {
                "base_name": "TwoPileNim",
                "id": "TwoPileNim0-v0",
            },
            "instance_num": 0,
        }
    )
    registry = EnvRegistry()
    env = registry.load(cfg)
    env_factory = lambda instance_num: registry.load(cfg)

    object_types = env.get_object_types()

    proposal_signature = (
        ("value_pattern", "VALUE"),
        ("neighbor_offset", "DIRECTION"),
    )

    existing = get_core_boolean_primitives(removed_primitive="shifted")

    # run evaluation
    result = evaluate_primitive(
        new_primitive_fn=pattern_check,
        existing_primitives=existing,
        object_types=object_types,
        env_factory=env_factory,
        proposal_signature=proposal_signature,
        seed=1,
        max_steps=20,
        num_samples=80,
        degeneracy_threshold=0.1,  # must have degeneracy < 0.1
        equivalence_threshold=0.95,  # must NOT match any primitive > 0.95
    )

    # Should be accepted
    assert (
        result["keep"] is True
    ), f"Function pattern_check() should pass filters but got {result}"

    # L1: Degeneracy check
    assert result["deg_score"] > 0.1, "Function pattern_check() should not be constant."

    # L2: Semantic equivalence check
    assert all(
        score < 0.95 for score in result["sim_scores"]
    ), "Function pattern_check() should not be equivalent to any existing primitive."


def test_degenerate_primitive_fails_filter() -> None:
    """
    L1: A degenerate primitive (constant output) must be rejected.
    """
    cfg = OmegaConf.create(
        {
            "provider": "ggg",
            "make_kwargs": {"base_name": "TwoPileNim", "id": "TwoPileNim0-v0"},
            "instance_num": 0,
        }
    )

    registry = EnvRegistry()
    env = registry.load(cfg)
    env_factory = lambda instance_num: registry.load(cfg)
    object_types = env.get_object_types()

    existing = get_core_boolean_primitives(removed_primitive="shifted")

    proposal_signature = (
        ("x", "VALUE"),  # fake args; L1 degeneracy doesn't care
        ("y", "DIRECTION"),
    )

    result = evaluate_primitive(
        new_primitive_fn=always_true,
        existing_primitives=existing,
        object_types=object_types,
        env_factory=env_factory,
        proposal_signature=proposal_signature,
        seed=1,
        max_steps=20,
        num_samples=80,
        degeneracy_threshold=0.1,
        equivalence_threshold=0.95,
    )

    assert result["keep"] is False, "Degenerate function should be rejected"
    assert result["reason"] == "degenerate"
    assert result["deg_score"] == 0.0


def test_duplicate_primitive_fails_similarity() -> None:
    """
    L2: A primitive that semantically replicates an existing one must be rejected.
    """
    cfg = OmegaConf.create(
        {
            "provider": "ggg",
            "make_kwargs": {"base_name": "TwoPileNim", "id": "TwoPileNim0-v0"},
            "instance_num": 0,
        }
    )

    registry = EnvRegistry()
    env = registry.load(cfg)
    env_factory = lambda instance_num: registry.load(cfg)
    object_types = env.get_object_types()

    existing = get_core_boolean_primitives(removed_primitive="shifted")

    proposal_signature = (("v", "VALUE"),)  # same signature as cell_is_value

    result = evaluate_primitive(
        new_primitive_fn=mimic_cell_is_value,
        existing_primitives=existing,
        object_types=object_types,
        env_factory=env_factory,
        proposal_signature=proposal_signature,
        seed=1,
        max_steps=20,
        num_samples=80,
        degeneracy_threshold=0.1,
        equivalence_threshold=0.95,
    )
    assert result["keep"] is False, "Duplicate semantic primitive should be rejected."

    assert len(result["sim_scores"]) >= 1, "There should be at least one sim score."
    assert "cell_is_value" in str(
        result["reason"]
    ), "Error message should indicate similarity with cell_is_value."
