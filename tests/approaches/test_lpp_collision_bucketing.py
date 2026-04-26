"""Tests for collision bucketing and repair prompt selection."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from gymnasium.spaces import Box, MultiDiscrete
from scipy.sparse import csr_matrix

from programmatic_policy_learning.approaches.lpp_approach import (
    LogicProgrammaticPolicyApproach,
)
from programmatic_policy_learning.approaches.lpp_utils import (
    lpp_collision_feedback_utils,
)
from programmatic_policy_learning.approaches.lpp_utils.utils import (
    build_collision_repair_prompt,
    log_feature_collisions,
)
from programmatic_policy_learning.data.demo_types import Trajectory
from programmatic_policy_learning.dsl.state_action_program import StateActionProgram


def _make_grid(offset: int) -> np.ndarray:
    """Create a small grid with offset token positions."""
    grid = np.full((3, 3), ".", dtype="<U1")
    grid[offset % 3, (offset + 1) % 3] = "A"
    grid[(offset + 1) % 3, (offset + 2) % 3] = "B"
    return grid


def test_log_feature_collisions_global_mixed_label_groups_full_bucket() -> None:
    """Global bucketing should keep every mixed-label row in one bucket."""
    X = csr_matrix(
        np.array(
            [
                [1, 0],
                [1, 0],
                [1, 0],
                [0, 1],
            ],
            dtype=np.uint8,
        )
    )
    y = np.array([1, 0, 1, 0], dtype=np.uint8)

    groups = log_feature_collisions(
        X,
        y,
        None,
        bucket_mode="global_mixed_label",
    )

    assert len(groups) == 1
    assert groups[0]["pos"] == [0, 2]
    assert groups[0]["neg"] == [1]
    assert groups[0]["total_count"] == 3
    assert groups[0]["max_occur"] == 2
    assert groups[0]["label_balance"] == pytest.approx(0.5)


def test_log_feature_collisions_positive_anchor_keeps_old_behavior() -> None:
    """Legacy anchor bucketing should still keep one positive anchor bucket."""
    X = csr_matrix(
        np.array(
            [
                [1, 0],
                [1, 0],
                [1, 0],
                [0, 1],
            ],
            dtype=np.uint8,
        )
    )
    y = np.array([1, 0, 1, 0], dtype=np.uint8)

    groups = log_feature_collisions(
        X,
        y,
        None,
        bucket_mode="positive_anchor",
    )

    assert len(groups) == 1
    assert groups[0]["pos"] == [0]
    assert groups[0]["neg"] == [1]
    assert groups[0]["max_occur"] == 1


def test_global_bucket_prompt_uses_multiple_positive_representatives() -> None:
    """Global bucket prompts should show a few positive reps, not just one."""
    examples = [(_make_grid(i), (i % 3, (i + 1) % 3)) for i in range(6)]

    prompt_anchor = build_collision_repair_prompt(
        pos_indices=[0, 1, 2],
        neg_indices=[3, 4, 5],
        examples=examples,
        collision_feedback_enc="enc_2",
        collision_template_feedback=False,
        bucket_mode="positive_anchor",
    )
    prompt_global = build_collision_repair_prompt(
        pos_indices=[0, 1, 2],
        neg_indices=[3, 4, 5],
        examples=examples,
        collision_feedback_enc="enc_2",
        collision_template_feedback=False,
        bucket_mode="global_mixed_label",
    )

    assert prompt_anchor.count("label=1 action=") == 1
    assert prompt_global.count("label=1 action=") >= 2
    assert prompt_global.count("label=0 action=") >= 2


def test_build_train_matrix_passes_collision_bucket_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initial collision scan should honor the configured bucket mode."""
    approach = LogicProgrammaticPolicyApproach(
        environment_description="DummyGrid-v0",
        observation_space=Box(low=0, high=1, shape=(2, 2), dtype=np.int64),
        action_space=MultiDiscrete([2, 2]),
        seed=0,
        expert=None,  # type: ignore[arg-type]
        env_factory=lambda instance_num=None: None,
        base_class_name="DummyGrid",
        demo_numbers=(0,),
        env_specs={"action_mode": "discrete"},
        prior_version="uniform",
        collision_feedback_bucket_mode="global_mixed_label",
        cross_demo_feature_filter={"enabled": False},
    )
    examples = [
        (np.zeros((2, 2), dtype=int), np.array([0, 0], dtype=np.int64)),
        (np.ones((2, 2), dtype=int), np.array([0, 1], dtype=np.int64)),
    ]
    captured_modes: list[str] = []

    def _fake_log_feature_collisions(
        X: Any,
        y: Any,
        examples: Any,
        bucket_mode: str = "positive_anchor",
    ) -> list[Any]:
        del X, y, examples
        captured_modes.append(bucket_mode)
        return []

    def fake_run_all_programs_on_demonstrations(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return (
            csr_matrix(np.array([[1], [1]], dtype=np.uint8)),
            np.array([1, 0], dtype=np.uint8),
            examples,
            np.ones(2, dtype=float),
            np.array([0, 0], dtype=int),
        )

    monkeypatch.setattr(
        "programmatic_policy_learning.approaches.lpp_approach."
        "run_all_programs_on_demonstrations",
        fake_run_all_programs_on_demonstrations,
    )
    monkeypatch.setattr(
        "programmatic_policy_learning.approaches.lpp_approach."
        "_filter_redundant_features",
        lambda X, programs, priors: (X, programs, priors, np.asarray([2])),
    )
    monkeypatch.setattr(
        approach,
        "_score_and_optionally_filter_features",
        lambda X, y, row_demo_ids, programs, priors: (X, programs, priors),
    )
    monkeypatch.setattr(
        "programmatic_policy_learning.approaches.lpp_approach.log_feature_collisions",
        _fake_log_feature_collisions,
    )

    # pylint: disable=protected-access
    approach._build_and_process_train_matrix(
        train_demo_ids=(0,),
        val_demo_ids=(),
        demo_dict_train={0: Trajectory(steps=cast(Any, examples))},
        programs_sa=[StateActionProgram("True")],
        program_prior_log_probs_opt=[0.0],
        dsl_functions={},
        negative_sampling_cfg=None,
        offline_path_name=None,
        start_index=1,
    )

    assert captured_modes == ["global_mixed_label"]


def test_collision_feedback_loop_recomputes_with_configured_bucket_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repair rounds should recompute collisions with the chosen bucket
    mode."""
    captured_modes: list[str] = []
    X = csr_matrix(np.array([[1], [1]], dtype=np.uint8))

    def _fake_log_feature_collisions(
        X: Any,
        y: Any,
        examples: Any,
        bucket_mode: str = "positive_anchor",
    ) -> list[Any]:
        del X, y, examples
        captured_modes.append(bucket_mode)
        return []

    def _append_new_features(
        X: Any,
        programs_sa: Any,
        program_prior_log_probs: Any,
        dsl_functions: Any,
        new_feature_sources: Any,
        examples: Any,
        *,
        start_index: int,
        collision_loop_idx: int,
        prior_version: str = "v1",
        prior_beta: float = 1.0,
    ) -> tuple[Any, int]:
        del (
            programs_sa,
            program_prior_log_probs,
            dsl_functions,
            new_feature_sources,
            examples,
            collision_loop_idx,
            prior_version,
            prior_beta,
        )
        return X, start_index + 1

    monkeypatch.setattr(
        lpp_collision_feedback_utils,
        "_append_new_features_from_sources",
        _append_new_features,
    )
    monkeypatch.setattr(
        lpp_collision_feedback_utils,
        "_filter_redundant_features",
        lambda X, programs_sa, priors, round_idx=None: (
            X,
            programs_sa,
            priors,
            np.asarray(X.getnnz(axis=0)).ravel(),
        ),
    )
    monkeypatch.setattr(
        lpp_collision_feedback_utils,
        "log_feature_collisions",
        _fake_log_feature_collisions,
    )

    lpp_collision_feedback_utils.run_collision_feedback_loop(
        collision_groups=[{"pos": [0], "neg": [1], "max_occur": 1}],
        examples=[
            (_make_grid(0), np.array([0, 0], dtype=np.int64)),
            (_make_grid(1), np.array([1, 1], dtype=np.int64)),
        ],
        max_rounds=1,
        target_collisions=0,
        start_index=1,
        program_prior_log_probs=None,
        X=X,
        y=np.array([1, 0], dtype=np.uint8),
        programs_sa=[StateActionProgram("True")],
        dsl_functions={},
        generate_features=lambda prompt, start_idx, collision_idx: (
            ["def f1(s, a):\n    return True"],
            {"features": []},
            Path.cwd(),
        ),
        make_prompt=lambda collision_groups, examples: "prompt",
        prior_version="uniform",
        prior_beta=1.0,
        collision_bucket_mode="global_mixed_label",
    )

    assert captured_modes == ["global_mixed_label"]
