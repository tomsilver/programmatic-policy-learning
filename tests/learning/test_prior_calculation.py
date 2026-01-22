"""Tests for feature prior calculation utilities."""

from __future__ import annotations

import math

from programmatic_policy_learning.learning.prior_calculation import (
    normalize_log_scores_to_probs,
    priors_from_features,
    probs_to_logprobs,
    score_features_log_prior,
)


def test_prior_scores_rank_simple_over_complex() -> None:
    """Simpler features should receive higher log-prior scores."""
    f_simple = "def f1(s, a):\n    r, c = a\n    return s[r][c] == 'empty'\n"
    f_complex = (
        "def f2(s, a):\n"
        "    r, c = a\n"
        "    h = len(s)\n"
        "    w = len(s[0]) if h else 0\n"
        "    if r < 0 or r >= h or c < 0 or c >= w:\n"
        "        return False\n"
        "    if s[r][c] != 'empty':\n"
        "        return False\n"
        "    cnt = 0\n"
        "    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:\n"
        "        rr, cc = r + dr, c + dc\n"
        "        if 0 <= rr < h and 0 <= cc < w and s[rr][cc] == 'wall':\n"
        "            cnt += 1\n"
        "    return cnt == 0\n"
    )
    scores = score_features_log_prior([f_simple, f_complex])
    assert len(scores) == 2
    assert scores[0] >= scores[1]


def test_prior_normalization_and_logs() -> None:
    """Probabilities should sum to 1 and logprobs should match."""
    features = [
        "def f1(s, a):\n    r, c = a\n    return s[r][c] == 'empty'\n",
        "def f2(s, a):\n    r, c = a\n    return s[r][c] == 'wall'\n",
        "def f3(s, a):\n    r, c = a\n    return s[r][c] == 'right_arrow'\n",
    ]
    log_scores = score_features_log_prior(features)
    probs = normalize_log_scores_to_probs(log_scores)
    logprobs = probs_to_logprobs(probs)

    assert len(probs) == len(features)
    assert math.isclose(sum(probs), 1.0, rel_tol=1e-6)
    for p, lp in zip(probs, logprobs):
        assert math.isclose(lp, math.log(max(p, 1e-300)), rel_tol=1e-6)

    out = priors_from_features(features)
    assert list(out.keys()) == ["log_scores", "probs", "logprobs"]
