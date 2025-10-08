from typing import TypeVar, Optional, Any
import numpy as np
from programmatic_policy_learning.approaches.base_approach import BaseApproach

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")

class PendulumParametric(BaseApproach[_ObsType, _ActType]):
    def __init__(
        self,
        name: str,
        observation_space,
        action_space,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, observation_space, action_space, seed=seed)
        self.kp = float(kwargs.get("kp", 12.0))  
        self.kd = 2.0                    

    def set_param(self, key: str, value: float) -> None:
        if key != "kp":
            raise AttributeError("only checking kp right now")
        self.kp = float(value)

    def _get_action(self) -> _ActType:
        obs = self._last_observation
        if obs is None:
            return np.array([0.0], dtype=self._action_space.dtype)

        x, y, angvel = float(obs[0]), float(obs[1]), float(obs[2])
        theta = float(np.arctan2(y, x))

        near_top = abs(theta) < 0.5
        if near_top:
            torque = -self.kp * theta - self.kd * angvel
        else:
            max_torque = float(self._action_space.high[0])
            torque = max_torque * (1.0 if angvel >= 0.0 else -1.0)

        low, high = float(self._action_space.low[0]), float(self._action_space.high[0])
        torque = float(np.clip(torque, low, high))
        return np.array([torque], dtype=self._action_space.dtype)
