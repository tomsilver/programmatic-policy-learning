"""Base interface for DSL learners."""

import abc
from typing import Any

from programmatic_policy_learning.dsl.core import DSL


class DSLLearner(abc.ABC):
    """Base class for DSL learning algorithms."""

    @abc.abstractmethod
    def generate_dsl(self) -> DSL[Any, Any, Any]:
        """Build or load a DSL object."""
        raise NotImplementedError
