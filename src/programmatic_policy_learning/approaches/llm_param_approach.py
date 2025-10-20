"""This is the approach program for the starting LLM approach in which we get
an intial baseline prompt to the LLM and then feed it like refining prompts to
help it figure out a proper function."""

from typing import Any, Dict, List, Tuple


class ParametricPolicyBase:
    """Minimal base to store tunable parameters and bounds. Subclasses must
    implement act(obs).

    - set_params(...) clips values into declared bounds.
    - parameter_names(), bounds(), get_params() are convenience helpers.
    """

    def __init__(
        self,
        init_params: Dict[str, float],
        param_bounds: Dict[str, Tuple[float, float]],
    ):
        self._params: Dict[str, float] = dict(init_params)
        self._bounds: Dict[str, Tuple[float, float]] = dict(param_bounds)

    def parameter_names(self) -> List[str]:
        """Returns parameter names."""
        return list(self._params.keys())

    def bounds(self) -> Dict[str, Tuple[float, float]]:
        """Returns bounds."""
        return dict(self._bounds)

    def get_params(self) -> Dict[str, float]:
        """Gets parameters."""
        return dict(self._params)

    def set_params(self, new_params: Dict[str, float]) -> None:
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

    def act(self, obs: Any):
        """Override in subclasses to compute an action from an observation."""
        raise NotImplementedError("Subclass must implement act(obs)")
