"""A simple hardcoded approach for pendulum balancing."""

from typing import TypeVar, cast

import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray

from programmatic_policy_learning.approaches.base_approach import BaseApproach

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")


class PendulumStupidAlgorithm(BaseApproach[_ObsType, _ActType]):
    """A hardcoded approach that tries to balance the pendulum at the top."""

    def _get_action(self) -> _ActType:
        if self._last_observation is None:
            space = cast(Box, self._action_space)
            zero = np.zeros(1, dtype=space.dtype)
            return cast(_ActType, zero)

        obs = cast(NDArray[np.floating], self._last_observation)
        space = cast(Box, self._action_space)

        x = float(obs[0])
        y = float(obs[1])
        angvel = float(obs[2])

        theta = float(np.arctan2(y, x))

        print(f"angle={theta:.3f}, angvel={angvel:.3f}")

        is_hanging_down = abs(abs(theta) - np.pi) < 1.0
        is_near_top = abs(theta) < 0.5

        if is_hanging_down:
            if abs(angvel) > 0.1:
                torque = 2.0 * np.sign(angvel)
            else:
                torque = 2.0 * np.sign(theta)
        elif is_near_top:
            kp = 12.0
            kd = 3.0
            torque = -kp * theta - kd * angvel
        else:
            kp = 5.0
            kd = 1.0
            torque = -kp * theta - kd * angvel
            if abs(angvel) > 1.0:
                torque += 1.0 * np.sign(angvel)

        low = float(cast(NDArray[np.floating], space.low)[0])
        high = float(cast(NDArray[np.floating], space.high)[0])
        torque = float(np.clip(torque, low, high))

        print(f"torque={torque:.3f}")

        act = np.array([torque], dtype=space.dtype)
        return cast(_ActType, act)
