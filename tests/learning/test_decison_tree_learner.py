"""Tests for decision tree learner."""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.tree import DecisionTreeClassifier

from programmatic_policy_learning.dsl.state_action_program import StateActionProgram
from programmatic_policy_learning.learning.decision_tree_learner import (
    extract_plp_from_dt,
    get_conjunctive_program,
    get_disjunctive_program,
    get_path_to_leaf,
    learn_plps,
    learn_single_batch_decision_trees,
)


def test_learn_single_batch_decision_trees() -> None:
    """Test training multiple decision trees."""
    X = csr_matrix(np.array([[0, 1], [1, 0], [1, 1]]))
    y = [True, False, True]
    clfs = learn_single_batch_decision_trees(y, 2, X)
    assert len(clfs) == 2
    assert all(isinstance(clf, DecisionTreeClassifier) for clf in clfs)


def test_get_path_to_leaf() -> None:
    """Test extracting path from root to leaf."""
    parents = {0: None, 1: (0, "left"), 2: (1, "right")}
    path = get_path_to_leaf(2, parents)
    assert path == [(0, "left"), (1, "right")]  # from root to leaf


def test_get_conjunctive_program() -> None:
    """Test building a conjunctive program from a path."""
    path = [(0, "right"), (1, "left")]
    node_to_features = np.array([0, 1])
    features = [StateActionProgram("a"), StateActionProgram("b")]
    feature_log_probs = [0.5, 0.2]
    program, log_p = get_conjunctive_program(
        path, node_to_features, features, feature_log_probs
    )
    assert isinstance(program, StateActionProgram)
    assert isinstance(log_p, float)


def test_get_disjunctive_program() -> None:
    """Test combining conjunctive programs into a disjunctive program."""
    programs = [StateActionProgram("a"), StateActionProgram("b")]
    disjunctive = get_disjunctive_program(programs)
    assert isinstance(disjunctive, StateActionProgram)


def test_extract_plp_from_dt() -> None:
    """Test extracting a program and log probability from a decision tree."""
    X = csr_matrix(np.array([[0, 1], [1, 0], [1, 1]]))
    y = [True, False, True]
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X, y)
    features = [StateActionProgram("a"), StateActionProgram("b")]
    feature_log_probs = [0.5, 0.2]
    plp, log_p = extract_plp_from_dt(clf, features, feature_log_probs)
    assert isinstance(plp, StateActionProgram)
    assert isinstance(log_p, float)


def test_learn_plps() -> None:
    """Test learning programmatic policies using decision trees."""
    X = csr_matrix(np.array([[0, 1], [1, 0], [1, 1]]))
    y = [True, False, True]
    programs = [StateActionProgram("a"), StateActionProgram("b")]
    program_prior_log_probs = [0.5, 0.2]
    plps, plp_priors = learn_plps(
        X,
        y,
        programs,
        program_prior_log_probs,
        num_dts=1,
        program_generation_step_size=1,
    )
    assert isinstance(plps, list)
    assert isinstance(plp_priors, list)
    assert all(isinstance(plp, StateActionProgram) for plp in plps)
    assert all(isinstance(prior, float) for prior in plp_priors)
