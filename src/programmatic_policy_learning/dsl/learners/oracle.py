"""Oracle DSL learner: loads a hand-specified DSL by id."""

from typing import Callable, TypeVar

from programmatic_policy_learning.dsl.core import DSL
from programmatic_policy_learning.dsl.learners.base import DSLLearner
from programmatic_policy_learning.dsl.primitives_sets import grid_v1

ProgT = TypeVar("ProgT")
InT = TypeVar("InT")
OutT = TypeVar("OutT")


class OracleDSLLearner(DSLLearner[ProgT, InT, OutT]):
    """Loads a DSL from a lookup table by id."""

    def __init__(self, dsl_id: str) -> None:
        self._dsl_id = dsl_id
        self._dsl = self._load_from_id(dsl_id)  # build once; immutable

    def _load_from_id(self, dsl_id: str) -> DSL[ProgT, InT, OutT]:
        table: dict[str, Callable[[], DSL[ProgT, InT, OutT]]] = {
            "grid_v1": grid_v1.make_dsl,  # type: ignore
        }
        if dsl_id not in table:
            choices = ", ".join(sorted(table))
            raise ValueError(f"Unknown dsl_id '{dsl_id}'. Available: {choices}")
        return table[dsl_id]()

    def generate_dsl(self) -> DSL[ProgT, InT, OutT]:
        return self._dsl
