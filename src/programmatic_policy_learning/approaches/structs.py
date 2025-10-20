"""Common data structures."""

import abc
from typing import Any


class ParametricPolicyBase(abc.ABC):
    """Minimal base to store tunable parameters and bounds. Subclasses must
    implement act(obs).

    - set_params(...) clips values into declared bounds.
    - parameter_names(), bounds(), get_params() are convenience helpers.
    """

    def __init__(
        self,
        init_params: dict[str, float],
        param_bounds: dict[str, tuple[float, float]],
    ):
        self._params: dict[str, float] = dict(init_params)
        self._bounds: dict[str, tuple[float, float]] = dict(param_bounds)
        assert set(self._params) == set(
            self._bounds
        ), "Params and bounds must have same param names"

    def parameter_names(self) -> set[str]:
        """Returns parameter names."""
        return set(self._params.keys())

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """Returns bounds."""
        return dict(self._bounds)

    def get_params(self) -> dict[str, float]:
        """Gets parameters."""
        return dict(self._params)

    def set_params(self, new_params: dict[str, float]) -> None:
        """Set parameters: update and clip to bounds."""
        for name, value in new_params.items():
            if name not in self._bounds:
                raise KeyError(f"Unknown parameter: {name}")
            lo, hi = self._bounds[name]
            if value < lo:
                value = lo
            elif value > hi:
                value = hi
            self._params[name] = float(value)

    @abc.abstractmethod
    def act(self, obs: Any) -> Any:
        """Override in subclasses to compute an action from an observation."""
