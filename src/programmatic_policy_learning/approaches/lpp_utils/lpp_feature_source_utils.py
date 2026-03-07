"""Feature source parsing helpers for LPP."""

import ast
from typing import Any


def _parse_py_feature_sources(
    feature_programs: list[str],
    dsl_functions: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Parse feature sources once and return callables + function names."""
    functions: dict[str, Any] = {}
    names: list[str] = []
    for source in feature_programs:
        tree = ast.parse(source)
        func_names = [
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        ]
        if not func_names:
            raise ValueError("Expected at least one function definition in feature.")
        names.extend(func_names)
        module_globals = dict(dsl_functions)
        exec(source, module_globals)  # pylint: disable=exec-used
        for name in func_names:
            fn = module_globals.get(name)
            if not callable(fn):
                raise ValueError(f"Feature function '{name}' is not callable.")
            functions[name] = fn
    return functions, names


def _extract_feature_names(feature_programs: list[str]) -> list[str]:
    """Extract function names from feature source strings."""
    names: list[str] = []
    for source in feature_programs:
        tree = ast.parse(source)
        func_names = [
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        ]
        if not func_names:
            raise ValueError("Expected at least one function definition in feature.")
        names.extend(func_names)
    return names
