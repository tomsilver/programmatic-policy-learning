"""A simple PD-style controller policy for Pendulum-v1.

Provides policy with a tunable gain kp value and a fixed gain kd value.
It is then used in the tests to preform grid serach over kp.
"""

from typing import Any, Optional, TypeVar, cast

import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray

from programmatic_policy_learning.approaches.base_approach import BaseApproach

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")


class PendulumParametric(BaseApproach[_ObsType, _ActType]):
    """A minimal PD controller for the Pendulum environment.

    - Apply feedback torque if near top or apply forceful torque
    to gain momentum.
    """

    def __init__(
        self,
        name: str,
        observation_space,
        action_space,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        seed_int = 0 if seed is None else int(seed)
        super().__init__(name, observation_space, action_space, seed=seed_int)
        self.kp = float(kwargs.get("kp", 12.0))
        self.kd = 2.0

    def set_param(self, key: str, value: float) -> None:
        """Set a tunable parameter (currently only supports 'kp')."""
        if key != "kp":
            raise AttributeError("only checking kp right now")
        self.kp = float(value)

    def _get_action(self) -> _ActType:
        """Compute the next torque action given the last observation."""
        space = cast(Box, self._action_space)

        if self._last_observation is None:
            return cast(_ActType, np.array([0.0], dtype=space.dtype))

        obs = cast(NDArray[np.floating], self._last_observation)
        x, y, angvel = float(obs[0]), float(obs[1]), float(obs[2])
        theta = float(np.arctan2(y, x))

        near_top = abs(theta) < 0.5
        if near_top:
            torque = -self.kp * theta - self.kd * angvel
        else:
            max_torque = float(cast(NDArray[np.floating], space.high)[0])
            torque = max_torque if angvel >= 0.0 else -max_torque

        low = float(cast(NDArray[np.floating], space.low)[0])
        high = float(cast(NDArray[np.floating], space.high)[0])
        torque = float(np.clip(torque, low, high))

        act = np.array([torque], dtype=space.dtype)
        return cast(_ActType, act)
