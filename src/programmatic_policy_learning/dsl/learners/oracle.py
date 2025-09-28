"""Oracle DSL learner: loads a hand-specified DSL by id."""

from typing import Any, Callable, Dict

from programmatic_policy_learning.dsl.core import DSL
from programmatic_policy_learning.dsl.learners.base import DSLLearner
from programmatic_policy_learning.dsl.primitives_sets import grid_v1


class OracleDSLLearner(DSLLearner):
    """Loads a DSL from a lookup table by id."""

    def __init__(self, dsl_id: str) -> None:
        self._dsl_id = dsl_id
        self._dsl = self._load_from_id(dsl_id)  # build once; immutable

    def _load_from_id(self, dsl_id: str) -> DSL[Any, Any, Any]:
        table: Dict[str, Callable[[], DSL[Any, Any, Any]]] = {
            "grid_v1": grid_v1.make_dsl,  # add more DSL here if you had more
        }
        if dsl_id not in table:
            choices = ", ".join(sorted(table))
            raise ValueError(f"Unknown dsl_id '{dsl_id}'. Available: {choices}")
        return table[dsl_id]()

    def generate_dsl(self) -> DSL[Any, Any, Any]:
        return self._dsl
