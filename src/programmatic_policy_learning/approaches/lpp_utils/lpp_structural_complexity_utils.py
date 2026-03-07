"""Structural complexity helpers for LPP program priors."""

import ast
from typing import Any


def _flatten_boolop(node: ast.AST, op_type: type[ast.boolop]) -> list[ast.AST]:
    if isinstance(node, ast.BoolOp) and isinstance(node.op, op_type):
        out: list[ast.AST] = []
        for v in node.values:
            out.extend(_flatten_boolop(v, op_type))
        return out
    return [node]


def _ast_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return 1
    return 1 + max(_ast_depth(child) for child in children)


def compute_program_structural_complexity(program: Any) -> dict[str, int]:
    """Compute structural complexity metrics from program syntax only."""
    expr = str(program).strip()
    try:
        tree = ast.parse(expr, mode="eval")
        root: ast.AST = tree.body
    except SyntaxError:
        return {
            "num_clauses": 1,
            "total_literals": 1,
            "max_clause_len": 1,
            "depth": 1,
            "ops": 0,
        }

    clauses = _flatten_boolop(root, ast.Or)
    num_clauses = max(1, len(clauses))
    clause_lit_counts: list[int] = []
    for clause in clauses:
        lits = _flatten_boolop(clause, ast.And)
        clause_lit_counts.append(max(1, len(lits)))

    total_literals = int(sum(clause_lit_counts)) if clause_lit_counts else 1
    max_clause_len = int(max(clause_lit_counts)) if clause_lit_counts else 1
    depth = _ast_depth(root)

    ops = 0
    for node in ast.walk(root):
        if isinstance(node, ast.BoolOp):
            ops += max(0, len(node.values) - 1)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            ops += 1

    return {
        "num_clauses": num_clauses,
        "total_literals": total_literals,
        "max_clause_len": max_clause_len,
        "depth": depth,
        "ops": int(ops),
    }


def compute_program_structural_log_prior(
    program: Any,
    *,
    alpha: float,
    w_clauses: float,
    w_literals: float,
    w_max_clause: float,
    w_depth: float,
    w_ops: float,
) -> float:
    """Structural log-prior term: -alpha * C_struct(program)."""
    c = compute_program_structural_complexity(program)
    cost = (
        w_clauses * c["num_clauses"]
        + w_literals * c["total_literals"]
        + w_max_clause * c["max_clause_len"]
        + w_depth * c["depth"]
        + w_ops * c["ops"]
    )
    return -float(alpha) * float(cost)


def _run_structural_prior_sanity_checks() -> None:
    """Unit-test-like sanity checks for structural prior monotonicity."""
    cfg = {
        "alpha": 1.0,
        "w_clauses": 1.0,
        "w_literals": 1.0,
        "w_max_clause": 1.0,
        "w_depth": 1.0,
        "w_ops": 1.0,
    }
    p_short = "f1(s, a)"
    p_long = "(f1(s, a) and f2(s, a)) or (f3(s, a) and f4(s, a))"
    lp_short = compute_program_structural_log_prior(p_short, **cfg)
    lp_long = compute_program_structural_log_prior(p_long, **cfg)
    assert lp_short > lp_long

    p_two = "f1(s, a) or f2(s, a)"
    p_three = "f1(s, a) or f2(s, a) or f3(s, a)"
    lp_two = compute_program_structural_log_prior(p_two, **cfg)
    lp_three = compute_program_structural_log_prior(p_three, **cfg)
    assert lp_two > lp_three
