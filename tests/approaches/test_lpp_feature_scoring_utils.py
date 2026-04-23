"""Tests for cross-demo LPP feature scoring."""

import numpy as np
from scipy.sparse import csr_matrix

from programmatic_policy_learning.approaches.lpp_utils.lpp_feature_scoring_utils import (
    feature_score_keep_mask,
    score_cross_demo_features,
)
from programmatic_policy_learning.dsl.state_action_program import StateActionProgram


def test_cross_demo_feature_scores_support_contrast_and_consistency() -> None:
    """Feature score should reward support and contrast across demos."""
    X = csr_matrix(
        np.array(
            [
                [True, True, False],
                [False, True, True],
                [True, False, False],
                [False, False, True],
            ],
            dtype=bool,
        )
    )
    y = np.array([1, 0, 1, 0], dtype=np.uint8)
    row_demo_ids = np.array([0, 0, 1, 1], dtype=int)
    programs = [
        StateActionProgram("f_good(s, a)"),
        StateActionProgram("f_flat(s, a)"),
        StateActionProgram("f_bad_action(s, a)"),
    ]

    scores = score_cross_demo_features(
        X,
        y,
        row_demo_ids,
        programs,
        consistency_tau=0.05,
    )

    good, flat, bad_action = scores
    assert good["pos_demo_support_count"] == 2
    assert good["pos_demo_support_frac"] == 1.0
    assert good["neg_demo_support_count"] == 0
    assert good["directional_support_frac"] == 1.0
    assert good["p_pos"] == 1.0
    assert good["p_neg"] == 0.0
    assert good["mean_demo_contrast"] == 1.0
    assert good["contrast_consistency_frac"] == 1.0
    assert good["cross_demo_score"] == 1.0

    assert flat["pos_demo_support_count"] == 1
    assert flat["p_pos"] == 0.5
    assert flat["p_neg"] == 0.5
    assert flat["cross_demo_score"] == 0.0

    assert bad_action["pos_demo_support_count"] == 0
    assert bad_action["neg_demo_support_count"] == 2
    assert bad_action["p_pos"] == 0.0
    assert bad_action["p_neg"] == 1.0
    assert bad_action["mean_demo_contrast"] == -1.0
    assert bad_action["contrast_consistency_frac"] == 1.0
    assert bad_action["cross_demo_score"] == 1.0


def test_feature_score_keep_mask_applies_cross_demo_thresholds() -> None:
    """Keep mask should filter low-support or low-contrast features."""
    scores = [
        {
            "pos_demo_support_count": 2,
            "neg_demo_support_count": 0,
            "mean_demo_abs_contrast": 0.5,
            "contrast_consistency_frac": 1.0,
            "total_fire_rate": 0.5,
        },
        {
            "pos_demo_support_count": 1,
            "neg_demo_support_count": 0,
            "mean_demo_abs_contrast": 0.5,
            "contrast_consistency_frac": 1.0,
            "total_fire_rate": 0.5,
        },
        {
            "pos_demo_support_count": 0,
            "neg_demo_support_count": 2,
            "mean_demo_abs_contrast": 0.5,
            "contrast_consistency_frac": 1.0,
            "total_fire_rate": 0.5,
        },
    ]

    keep_mask = feature_score_keep_mask(
        scores,
        min_pos_demo_support=2,
        min_neg_demo_support=2,
        allow_negative_support=True,
        min_abs_mean_demo_contrast=0.02,
        min_consistency_frac=0.4,
        min_total_fire_rate=0.005,
        max_total_fire_rate=0.95,
    )

    assert keep_mask.tolist() == [True, False, True]


def test_feature_score_keep_mask_can_disable_negative_support() -> None:
    """Negative-only support can be disabled when strict positive features are
    wanted."""
    scores = [
        {
            "pos_demo_support_count": 0,
            "neg_demo_support_count": 2,
            "mean_demo_abs_contrast": 0.5,
            "contrast_consistency_frac": 1.0,
            "total_fire_rate": 0.5,
        },
    ]

    keep_mask = feature_score_keep_mask(
        scores,
        min_pos_demo_support=2,
        min_neg_demo_support=2,
        allow_negative_support=False,
        min_abs_mean_demo_contrast=0.02,
        min_consistency_frac=0.4,
        min_total_fire_rate=0.005,
        max_total_fire_rate=0.95,
    )

    assert keep_mask.tolist() == [False]
