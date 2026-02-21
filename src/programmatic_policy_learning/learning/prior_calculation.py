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

        self._depth = 0
        self.max_depth = 0

    def generic_visit(self, node: ast.AST) -> None:
        """Visit a node while tracking depth and node count."""
        self.ast_nodes += 1
        self._depth += 1
        self.max_depth = max(self.max_depth, self._depth)
        super().generic_visit(node)
        self._depth -= 1

    def visit_Call(self, node: ast.Call) -> None:
        """Count function calls."""
        self.calls += 1
        self.generic_visit(node)

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
