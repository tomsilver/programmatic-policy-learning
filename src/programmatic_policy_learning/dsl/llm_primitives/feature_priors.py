"""Compute grammar-based log priors from LLM feature payloads."""

from __future__ import annotations

import math
from typing import Any, Sequence

START_PROBS = {"at_cell_with_value": 0.5, "at_action_cell": 0.5}
LOCAL_PROB_SHIFTED = 0.5
COND_PROB_CELL_IS_VALUE = 0.5
COND_PROB_SCANNING = 0.5
DIRECTION_PROB = 1.0 / 8.0


def _value_log_prob(object_types: Sequence[str]) -> float:
    if not object_types:
        raise ValueError("object_types must be non-empty")
    return math.log(1.0 / len(object_types))


def _num_log_prob(value: int) -> float:
    """Return log prob for POSITIVE_NUM/NEGATIVE_NUM recursion."""
    if value == 0:
        return 0.0
    steps = abs(value)
    return math.log(0.99) + (steps - 1) * math.log(0.01)


def _dir_log_prob(direction: Sequence[int]) -> float:
    if len(direction) != 2:
        raise ValueError(f"Invalid direction: {direction}")
    dr, dc = direction
    return math.log(DIRECTION_PROB) + _num_log_prob(dr) + _num_log_prob(dc)


def compute_feature_log_probs(
    payload: dict[str, Any],
    object_types: Sequence[str],
) -> list[float]:
    """Compute log priors for a {features: [...]} payload with ASTs."""
    if "features" not in payload:
        raise ValueError("Expected payload with a 'features' key.")

    value_log_prob = _value_log_prob(object_types)

    def walk(node: Any) -> float:
        if isinstance(node, dict):
            if "value" in node:
                return value_log_prob
            if "dir" in node:
                return _dir_log_prob(node["dir"])
            if "var" in node:
                return 0.0
            if "lambda" in node:
                return walk(node.get("body"))
            if "call" in node:
                call = node["call"]
                args = node.get("args", [])
                log_p = 0.0
                if call in START_PROBS:
                    log_p += math.log(START_PROBS[call])
                    if args and isinstance(args[0], dict) and "value" in args[0]:
                        log_p += value_log_prob
                elif call == "shifted":
                    log_p += math.log(LOCAL_PROB_SHIFTED)
                elif call == "cell_is_value":
                    log_p += math.log(COND_PROB_CELL_IS_VALUE)
                    if args and isinstance(args[0], dict) and "value" in args[0]:
                        log_p += value_log_prob
                elif call == "scanning":
                    log_p += math.log(COND_PROB_SCANNING)
                elif call in {"and", "or", "not"}:
                    log_p += 0.0

                for arg in args:
                    log_p += walk(arg)
                return log_p
        if isinstance(node, list):
            return sum(walk(x) for x in node)
        return 0.0

    priors = []
    for feature in payload["features"]:
        if not isinstance(feature, dict) or "ast" not in feature:
            raise ValueError("Each feature must be a dict with an 'ast' key.")
        priors.append(walk(feature["ast"]))
    return priors
