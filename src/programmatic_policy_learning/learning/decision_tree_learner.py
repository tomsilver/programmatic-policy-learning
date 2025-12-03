"""Decision tree-based programmatic policy learning utilities.

This module provides functions to train decision trees, extract logical
programs from them, and combine these programs into higher-level policy
representations using StateActionProgram class.
"""

import logging
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.tree import DecisionTreeClassifier

from programmatic_policy_learning.dsl.state_action_program import (
    StateActionProgram,
    set_dsl_functions,
)


def learn_plps(
    X: csr_matrix,
    y: list[bool],
    programs: list[StateActionProgram],
    program_prior_log_probs: list[float],
    num_dts: int = 5,
    program_generation_step_size: int = 10,
    dsl_functions: dict | None = None,
) -> tuple[list[StateActionProgram], list[float]]:
    """
    Parameters
    ----------
    X : csr_matrix
    y : [ bool ]
    programs : [ StateActionProgram ]
    program_prior_log_probs : [ float ]
    num_dts : int
    program_generation_step_size : int

    Returns
    -------
    plps : [ StateActionProgram ]
    plp_priors : [ float ]
        Log probabilities.
    """
    plps = []
    plp_priors = []

    num_programs = len(programs)

    for i in range(0, num_programs, program_generation_step_size):
        logging.info(f"Learning plps with {i} programs")
        for clf in learn_single_batch_decision_trees(y, num_dts, X[:, : i + 1]):
            plp, plp_prior_log_prob = extract_plp_from_dt(
                clf, programs, program_prior_log_probs, dsl_functions
            )
            plps.append(plp)
            plp_priors.append(plp_prior_log_prob)

    return plps, plp_priors


def learn_single_batch_decision_trees(
    y: list[bool], num_dts: int, X_i: csr_matrix
) -> list[DecisionTreeClassifier]:
    """
    Parameters
    ----------
    y : [ bool ]
    num_dts : int
    X_i : csr_matrix

    Returns
    -------
    clfs : [ DecisionTreeClassifier ]
    """

    clfs = []

    for seed in range(num_dts):
        clf = DecisionTreeClassifier(random_state=seed)
        clf.fit(X_i, y)
        clfs.append(clf)
    return clfs


def get_path_to_leaf(
    leaf: int, parents: dict[int, tuple[int, str] | None]
) -> list[tuple[int, str]]:
    """Return the path from root to the given leaf node."""
    reverse_path = []
    current = leaf

    while True:
        value = parents[current]
        if value is None:
            break
        parent, parent_choice = value
        reverse_path.append((parent, parent_choice))
        current = parent

    return reverse_path[::-1]


def get_conjunctive_program(
    path: list[tuple[int, str]],
    node_to_features: np.ndarray,
    features: list[StateActionProgram],
    feature_log_probs: list[float],
) -> tuple[StateActionProgram, float]:
    """Build a conjunctive program and its log probability from a decision tree
    path."""

    program = "("
    log_p = 0.0

    for i, (node_id, sign) in enumerate(path):
        feature_idx = node_to_features[node_id]
        precondition = features[feature_idx]
        feature_log_p = feature_log_probs[feature_idx]
        log_p += feature_log_p

        if sign == "right":
            program = program + precondition
        else:
            assert sign == "left"
            program = program + "not (" + precondition + ")"

        if i < len(path) - 1:
            program = program + " and "

    program = program + ")"
    return StateActionProgram(program), log_p


def get_disjunctive_program(
    conjunctive_programs: list[StateActionProgram],
) -> StateActionProgram:
    """Combine conjunctive programs into a disjunctive StateActionProgram."""

    if len(conjunctive_programs) == 0:
        return StateActionProgram("False")  # BUG?

    program = ""

    for i, conjunctive_program in enumerate(conjunctive_programs):
        program = program + "(" + str(conjunctive_program) + ")"  # converted to str
        if i < len(conjunctive_programs) - 1:
            program = program + " or "

    return StateActionProgram(program)


def extract_plp_from_dt(
    estimator: DecisionTreeClassifier,
    features: list[StateActionProgram],
    feature_log_probs: list[float],
    dsl_functions: dict[str, Any] | None,
) -> tuple[StateActionProgram, float]:
    """Extract a program and its log probability from a decision tree."""
    # n_nodes = estimator.tree_.node_count
    if dsl_functions is None:
        raise ValueError("dsl_functions cannot be None when calling set_dsl_functions.")

    set_dsl_functions(dsl_functions)
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    node_to_features = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    value = estimator.tree_.value.squeeze()

    stack = [0]
    parents: dict[int, tuple[int, str] | None] = {0: None}
    true_leaves = []
    while len(stack) > 0:
        node_id = stack.pop()
        if children_left[node_id] != children_right[node_id]:
            assert 0 < threshold[node_id] < 1
            stack.append(children_left[node_id])
            parents[children_left[node_id]] = (node_id, "left")
            stack.append(children_right[node_id])
            parents[children_right[node_id]] = (node_id, "right")

        elif value[node_id][1] > value[node_id][0]:
            true_leaves.append(node_id)

    paths_to_true_leaves = [get_path_to_leaf(leaf, parents) for leaf in true_leaves]

    conjunctive_programs = []
    program_log_prob = 0.0

    for path in paths_to_true_leaves:
        and_program, log_p = get_conjunctive_program(
            path, node_to_features, features, feature_log_probs
        )
        conjunctive_programs.append(and_program)
        program_log_prob += log_p

    disjunctive_program = get_disjunctive_program(conjunctive_programs)

    return disjunctive_program, program_log_prob
