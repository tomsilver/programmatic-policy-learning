"""Compute heuristic priors for feature definitions based on AST complexity."""

from __future__ import annotations

import ast
import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence, Union

Feature = Union[str, Callable[..., Any]]


# -----------------------------
# Complexity metrics
# -----------------------------


@dataclass(frozen=True)
class FeatureComplexity:
    """Structural complexity summary for a feature AST."""

    ast_nodes: int
    calls: int
    branches: int  # if / ifexp
    bool_ops: int  # and/or
    comparisons: int  # ==, <, in, etc
    loops: int  # for/while
    comprehensions: int  # list/set/dict/gen comps
    lambdas: int
    returns: int
    max_depth: int
    assigns: int
    imports: int
    try_blocks: int
    mod_ops: int  # counts '%' usage
    floordiv_ops: int  # counts '//' usage
    pow_ops: int  # counts '**' usage


class _ComplexityVisitor(ast.NodeVisitor):
    """AST visitor that counts structural elements and depth."""

    def __init__(self) -> None:
        """Initialize counters for AST complexity."""
        self.ast_nodes = 0
        self.calls = 0
        self.branches = 0
        self.bool_ops = 0
        self.comparisons = 0
        self.loops = 0
        self.comprehensions = 0
        self.lambdas = 0
        self.returns = 0
        self.assigns = 0
        self.imports = 0
        self.try_blocks = 0
        self.mod_ops = 0
        self.floordiv_ops = 0
        self.pow_ops = 0
        self.eq_comparisons = 0
        self.ineq_comparisons = 0
        self.big_constants = 0
        self.big_range_lengths = 0
        self.big_constant_k = 7
        self.range_length_k = 7

        self._depth = 0
        self.max_depth = 0

    def generic_visit(self, node: ast.AST) -> None:
        """Visit a node while tracking depth and node count."""
        self.ast_nodes += 1
        self._depth += 1
        self.max_depth = max(self.max_depth, self._depth)
        super().generic_visit(node)
        self._depth -= 1

    def visit_If(self, node: ast.If) -> None:
        """Count if statements."""
        self.branches += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        """Count inline if expressions."""
        self.branches += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        """Count boolean operations."""
        self.bool_ops += 1
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        """Count comparisons."""
        self.comparisons += 1
        for op in node.ops:
            if isinstance(op, ast.Eq):
                self.eq_comparisons += 1
            elif isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                self.ineq_comparisons += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        """Count for loops."""
        self.loops += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        """Count while loops."""
        self.loops += 1
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        """Count list comprehensions."""
        self.comprehensions += 1
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        """Count set comprehensions."""
        self.comprehensions += 1
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        """Count dict comprehensions."""
        self.comprehensions += 1
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        """Count generator expressions."""
        self.comprehensions += 1
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Count lambdas."""
        self.lambdas += 1
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        """Count return statements."""
        self.returns += 1
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Count assignments."""
        self.assigns += 1
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Count annotated assignments."""
        self.assigns += 1
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Count augmented assignments."""
        self.assigns += 1
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        """Count imports."""
        self.imports += 1
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Count from-imports."""
        self.imports += 1
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        """Count try blocks."""
        self.try_blocks += 1
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Count specific arithmetic operators like %, //, **."""
        if isinstance(node.op, ast.Mod):
            self.mod_ops += 1
        elif isinstance(node.op, ast.FloorDiv):
            self.floordiv_ops += 1
        elif isinstance(node.op, ast.Pow):
            self.pow_ops += 1
        self.generic_visit(node)

    def _const_num(self, node: ast.AST) -> float | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        return None

    def _range_len_if_literal(self, node: ast.Call) -> int | None:
        if not isinstance(node.func, ast.Name) or node.func.id != "range":
            return None
        if len(node.args) == 0 or len(node.args) > 3:
            return None
        vals = [self._const_num(arg) for arg in node.args]
        if any(v is None for v in vals):
            return None
        ivals = [int(v) for v in vals if v is not None]
        if len(ivals) == 1:
            return max(0, ivals[0])
        if len(ivals) == 2:
            start, stop = ivals
            step = 1
        else:
            start, stop, step = ivals
            if step == 0:
                return None
        # Python range length formula.
        if (stop - start) * step <= 0:
            return 0
        return max(0, math.ceil((stop - start) / step))

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, (int, float)):
            if abs(float(node.value)) > float(self.big_constant_k):
                self.big_constants += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Count function calls + literal range lengths."""
        self.calls += 1
        rlen = self._range_len_if_literal(node)
        if rlen is not None and rlen > self.range_length_k:
            self.big_range_lengths += 1
        self.generic_visit(node)


def _extract_source(feature: Feature) -> str:
    """
    Accept either:
      - a string containing 'def fX(s, a): ...'
      - a Python function object (best-effort inspect.getsource)
    """
    if isinstance(feature, str):
        return feature
    try:
        return inspect.getsource(feature)
    except Exception as exc:
        raise ValueError(
            "Could not extract source from callable. "
            "Pass the feature as a source string instead."
        ) from exc


def analyze_feature_complexity(feature: Feature) -> FeatureComplexity:
    """Compute complexity metrics for a single feature."""
    src = _extract_source(feature)
    tree = ast.parse(src)
    v = _ComplexityVisitor()
    v.visit(tree)
    return FeatureComplexity(
        ast_nodes=v.ast_nodes,
        calls=v.calls,
        branches=v.branches,
        bool_ops=v.bool_ops,
        comparisons=v.comparisons,
        loops=v.loops,
        comprehensions=v.comprehensions,
        lambdas=v.lambdas,
        returns=v.returns,
        max_depth=v.max_depth,
        assigns=v.assigns,
        imports=v.imports,
        try_blocks=v.try_blocks,
        mod_ops=v.mod_ops,
        floordiv_ops=v.floordiv_ops,
        pow_ops=v.pow_ops,
    )


@dataclass(frozen=True)
class FeatureComplexityV2:
    """V2 complexity summary with operator-specific comparisons and large-
    constant/range penalties."""

    ast_nodes: int
    calls: int
    branches: int
    bool_ops: int
    comparisons: int
    eq_comparisons: int
    ineq_comparisons: int
    loops: int
    comprehensions: int
    lambdas: int
    returns: int
    max_depth: int
    assigns: int
    imports: int
    try_blocks: int
    mod_ops: int
    floordiv_ops: int
    pow_ops: int
    big_constants: int
    big_range_lengths: int


def analyze_feature_complexity_v2(
    feature: Feature,
    *,
    big_constant_k: int = 7,
    range_length_k: int = 7,
) -> FeatureComplexityV2:
    """Compute V2 complexity metrics for a single feature."""
    src = _extract_source(feature)
    tree = ast.parse(src)
    v = _ComplexityVisitor()
    v.big_constant_k = int(big_constant_k)
    v.range_length_k = int(range_length_k)
    v.visit(tree)
    return FeatureComplexityV2(
        ast_nodes=v.ast_nodes,
        calls=v.calls,
        branches=v.branches,
        bool_ops=v.bool_ops,
        comparisons=v.comparisons,
        eq_comparisons=v.eq_comparisons,
        ineq_comparisons=v.ineq_comparisons,
        loops=v.loops,
        comprehensions=v.comprehensions,
        lambdas=v.lambdas,
        returns=v.returns,
        max_depth=v.max_depth,
        assigns=v.assigns,
        imports=v.imports,
        try_blocks=v.try_blocks,
        mod_ops=v.mod_ops,
        floordiv_ops=v.floordiv_ops,
        pow_ops=v.pow_ops,
        big_constants=v.big_constants,
        big_range_lengths=v.big_range_lengths,
    )


# -----------------------------
# Turn complexity -> prior score
# -----------------------------


@dataclass(frozen=True)
class PriorWeights:
    """Weights mapping complexity metrics to a log-prior penalty."""

    # penalty weights (tune these)
    ast_nodes: float = 0.01
    calls: float = 0.08
    branches: float = 0.20
    bool_ops: float = 0.12
    comparisons: float = 0.06
    loops: float = 0.40
    comprehensions: float = 0.35
    lambdas: float = 0.10
    returns: float = 0.05
    max_depth: float = 0.03
    assigns: float = 0.06
    imports: float = 1.00  # strongly discourage
    try_blocks: float = 0.50  # discourage
    base: float = 0.0  # additive constant
    mod_ops: float = 3.0
    floordiv_ops: float = 1.5
    pow_ops: float = 1.5


def feature_log_prior_from_complexity(
    c: FeatureComplexity, w: PriorWeights = PriorWeights()
) -> float:
    """Log prior ∝ -penalty(complexity) Higher is better (simpler features get
    higher log-prior)."""
    penalty = (
        w.ast_nodes * c.ast_nodes
        + w.calls * c.calls
        + w.branches * c.branches
        + w.bool_ops * c.bool_ops
        + w.comparisons * c.comparisons
        + w.loops * c.loops
        + w.comprehensions * c.comprehensions
        + w.lambdas * c.lambdas
        + w.returns * c.returns
        + w.max_depth * c.max_depth
        + w.assigns * c.assigns
        + w.imports * c.imports
        + w.try_blocks * c.try_blocks
        + w.mod_ops * c.mod_ops
        + w.floordiv_ops * c.floordiv_ops
        + w.pow_ops * c.pow_ops
        + w.base
    )
    return -penalty


@dataclass(frozen=True)
class PriorWeightsV2:
    """V2 penalty weights (eq comparisons intentionally heavier than ineq)."""

    ast_nodes: float = 0.01
    calls: float = 0.08
    branches: float = 0.20
    bool_ops: float = 0.12
    loops: float = 0.40
    comprehensions: float = 0.35
    lambdas: float = 0.10
    returns: float = 0.05
    max_depth: float = 0.03
    assigns: float = 0.06
    imports: float = 1.00
    try_blocks: float = 0.50
    mod_ops: float = 3.0
    floordiv_ops: float = 1.5
    pow_ops: float = 1.5
    eq_comparisons: float = 0.20
    ineq_comparisons: float = 0.08
    big_constants: float = 0.35
    big_range_lengths: float = 0.45
    base: float = 0.0


def feature_log_prior_from_complexity_v2(
    c: FeatureComplexityV2, w: PriorWeightsV2 = PriorWeightsV2()
) -> float:
    """V2 unnormalized log-prior: higher is better, penalize brittle patterns."""
    penalty = (
        w.ast_nodes * c.ast_nodes
        + w.calls * c.calls
        + w.branches * c.branches
        + w.bool_ops * c.bool_ops
        + w.loops * c.loops
        + w.comprehensions * c.comprehensions
        + w.lambdas * c.lambdas
        + w.returns * c.returns
        + w.max_depth * c.max_depth
        + w.assigns * c.assigns
        + w.imports * c.imports
        + w.try_blocks * c.try_blocks
        + w.mod_ops * c.mod_ops
        + w.floordiv_ops * c.floordiv_ops
        + w.pow_ops * c.pow_ops
        + w.eq_comparisons * c.eq_comparisons
        + w.ineq_comparisons * c.ineq_comparisons
        + w.big_constants * c.big_constants
        + w.big_range_lengths * c.big_range_lengths
        + w.base
    )
    return -penalty


# -----------------------------
# list scoring + normalization
# -----------------------------


def score_features_log_prior(
    features: Sequence[Feature], w: PriorWeights = PriorWeights()
) -> list[float]:
    """
    Input:  [feature1, feature2, ...] (each is source string or callable)
    Output: [log_prior1, log_prior2, ...] aligned with input order
    """
    out: list[float] = []
    for feat in features:
        c = analyze_feature_complexity(feat)
        out.append(feature_log_prior_from_complexity(c, w=w))
    return out


def score_features_log_prior_v2(
    features: Sequence[Feature],
    *,
    w: PriorWeightsV2 = PriorWeightsV2(),
    big_constant_k: int = 7,
    range_length_k: int = 7,
) -> list[float]:
    """V2 unnormalized log prior scores."""
    out: list[float] = []
    for feat in features:
        c = analyze_feature_complexity_v2(
            feat,
            big_constant_k=big_constant_k,
            range_length_k=range_length_k,
        )
        out.append(feature_log_prior_from_complexity_v2(c, w=w))
    return out


def normalize_log_scores_to_probs(log_scores: Sequence[float]) -> list[float]:
    """Softmax over log-scores to get a proper prior distribution.

    Returns probabilities aligned with input order.
    """
    if not log_scores:
        return []
    m = max(log_scores)
    exps = [math.exp(v - m) for v in log_scores]
    z = sum(exps)
    return [v / z for v in exps]


def probs_to_logprobs(probs: Sequence[float], eps: float = 1e-300) -> list[float]:
    """Convert probabilities to log-probabilities (aligned with input order).

    eps avoids log(0).
    """
    return [math.log(max(float(p), eps)) for p in probs]


def priors_from_features(
    features: Sequence[Feature],
    w: PriorWeights = PriorWeights(),
) -> dict:
    """
    Convenience wrapper:
      - log_scores: unnormalized log prior scores (higher = simpler)
      - probs: softmax(log_scores)
      - logprobs: log(probs)

    All outputs are lists aligned with the input order.
    """
    log_scores = score_features_log_prior(features, w=w)
    probs = normalize_log_scores_to_probs(log_scores)
    logprobs = probs_to_logprobs(probs)
    return {
        "log_scores": log_scores,
        "probs": probs,
        "logprobs": logprobs,
    }


def combine_log_likelihood_with_log_prior(
    log_likelihoods: Sequence[float],
    log_scores: Sequence[float],
    *,
    beta: float = 1.0,
) -> list[float]:
    """Compute unnormalized combined log weights:
    log_weight = log_likelihood + beta * log_score
    """
    if len(log_likelihoods) != len(log_scores):
        raise ValueError("log_likelihoods and log_scores must have the same length.")
    return [
        float(ll) + float(beta) * float(ls)
        for ll, ls in zip(log_likelihoods, log_scores)
    ]


def priors_from_features_v2(
    features: Sequence[Feature],
    *,
    w: PriorWeightsV2 = PriorWeightsV2(),
    big_constant_k: int = 7,
    range_length_k: int = 7,
    beta: float = 1.0,
) -> dict[str, Any]:
    """V2 convenience wrapper with unnormalized log scores + beta-scaled
    scores."""
    log_scores = score_features_log_prior_v2(
        features,
        w=w,
        big_constant_k=big_constant_k,
        range_length_k=range_length_k,
    )
    probs = normalize_log_scores_to_probs(log_scores)
    logprobs = probs_to_logprobs(probs)
    beta_log_scores = [float(beta) * v for v in log_scores]
    return {
        "version": "v3",
        "beta": float(beta),
        "log_scores": log_scores,
        "beta_log_scores": beta_log_scores,
        "probs": probs,
        "logprobs": logprobs,
    }
