"""Base interface for DSL learners."""

import abc
from typing import Generic, TypeVar

from programmatic_policy_learning.dsl.core import DSL

ProgT = TypeVar("ProgT")
InT = TypeVar("InT")
OutT = TypeVar("OutT")


class DSLLearner(Generic[ProgT, InT, OutT], abc.ABC):
    """Base class for DSL learning algorithms."""

    @abc.abstractmethod
    def generate_dsl(self) -> DSL[ProgT, InT, OutT]:
        """Return a DSL object for this learner."""
        raise NotImplementedError
