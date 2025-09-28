"""Core DSL class and types."""

from dataclasses import dataclass
from typing import Any, Callable, Generic, Mapping, TypeVar

ProgT = TypeVar("ProgT")  # program representation (callable, AST, str, …)
InT = TypeVar("InT")  # the *input bundle* the evaluator expects
OutT = TypeVar("OutT")  # the evaluator’s return type (e.g., bool, Action, dict)


@dataclass(frozen=True)
class DSL(Generic[ProgT, InT, OutT]):
    """Tiny DSL handle."""

    id: str
    primitives: Mapping[str, Callable[..., Any]]
    evaluate_fn: Callable[[ProgT, InT], OutT]

    def evaluate(self, program: ProgT, inputs: InT) -> OutT:
        """Run program on inputs.

        Domain primitives define both types.
        """
        return self.evaluate_fn(program, inputs)
