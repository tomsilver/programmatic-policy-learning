from __future__ import annotations

import ast
import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, List, Sequence, Union


# A "feature" can be either:
#  - a string containing "def fX(s, a): ..."
#  - a Python function object (inspect.getsource best-effort)
Feature = Union[str, Callable[..., Any]]


# -----------------------------
# 1) Complexity metrics
# -----------------------------

@dataclass(frozen=True)
class FeatureComplexity:
    ast_nodes: int
    calls: int
    branches: int          # if / ifexp
    bool_ops: int          # and/or
    comparisons: int       # ==, <, in, etc
    loops: int             # for/while
    comprehensions: int    # list/set/dict/gen comps
    lambdas: int
    returns: int
    max_depth: int
    assigns: int
    imports: int
    try_blocks: int


class _ComplexityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
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

        self._depth = 0
        self.max_depth = 0

    def generic_visit(self, node: ast.AST) -> None:
        self.ast_nodes += 1
        self._depth += 1
        self.max_depth = max(self.max_depth, self._depth)
        super().generic_visit(node)
        self._depth -= 1

    def visit_Call(self, node: ast.Call) -> None:
        self.calls += 1
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self.branches += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.branches += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.bool_ops += 1
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        self.comparisons += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.loops += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.loops += 1
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.comprehensions += 1
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.comprehensions += 1
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.comprehensions += 1
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.comprehensions += 1
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.lambdas += 1
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self.returns += 1
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.assigns += 1
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.assigns += 1
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.assigns += 1
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.imports += 1
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.imports += 1
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.try_blocks += 1
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
    )


# -----------------------------
# 2) Turn complexity -> prior score
# -----------------------------

@dataclass(frozen=True)
class PriorWeights:
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
    imports: float = 1.00     # strongly discourage
    try_blocks: float = 0.50  # discourage
    base: float = 0.0         # additive constant


def feature_log_prior_from_complexity(c: FeatureComplexity, w: PriorWeights = PriorWeights()) -> float:
    """
    log prior ∝ -penalty(complexity)
    Higher is better (simpler features get higher log-prior).
    """
    penalty = (
        w.ast_nodes * c.ast_nodes +
        w.calls * c.calls +
        w.branches * c.branches +
        w.bool_ops * c.bool_ops +
        w.comparisons * c.comparisons +
        w.loops * c.loops +
        w.comprehensions * c.comprehensions +
        w.lambdas * c.lambdas +
        w.returns * c.returns +
        w.max_depth * c.max_depth +
        w.assigns * c.assigns +
        w.imports * c.imports +
        w.try_blocks * c.try_blocks +
        w.base
    )
    return -penalty


# -----------------------------
# 3) List scoring + normalization
# -----------------------------

def score_features_log_prior(features: Sequence[Feature], w: PriorWeights = PriorWeights()) -> List[float]:
    """
    Input:  [feature1, feature2, ...] (each is source string or callable)
    Output: [log_prior1, log_prior2, ...] aligned with input order
    """
    out: List[float] = []
    for feat in features:
        c = analyze_feature_complexity(feat)
        out.append(feature_log_prior_from_complexity(c, w=w))
    return out


def normalize_log_scores_to_probs(log_scores: Sequence[float]) -> List[float]:
    """
    Softmax over log-scores to get a proper prior distribution.
    Returns probabilities aligned with input order.
    """
    if not log_scores:
        return []
    m = max(log_scores)
    exps = [math.exp(v - m) for v in log_scores]
    z = sum(exps)
    return [v / z for v in exps]


def probs_to_logprobs(probs: Sequence[float], eps: float = 1e-300) -> List[float]:
    """
    Convert probabilities to log-probabilities (aligned with input order).
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

