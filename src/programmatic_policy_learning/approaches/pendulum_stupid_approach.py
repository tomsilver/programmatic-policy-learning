"""A simple hardcoded approach for pendulum balancing."""

from typing import TypeVar, cast

import gymnasium as gym
import numpy as np

from programmatic_policy_learning.approaches.base_approach import BaseApproach

_ObsType = TypeVar("_ObsType")  
_ActType = TypeVar("_ActType")


class PendulumStupidAlgorithm(BaseApproach[_ObsType, _ActType]):
    """A hardcoded approach that tries to balance the pendulum at the top."""

    def _get_action(self) -> _ActType:  
        currobs = self._last_observation

        # Safety check
        if currobs is None:
            action_space = cast(gym.spaces.Box, self._action_space)
            return cast(_ActType, np.array([0.0], dtype=action_space.dtype))

        # Parse observation: [cos(θ), sin(θ), angular_velocity]
        obs = np.asarray(currobs, dtype=np.float32)
        x = float(obs[0])  
        y = float(obs[1])  
        angvel = float(obs[2])  

        theta = np.arctan2(y, x)

        is_hanging_down = abs(abs(theta) - np.pi) < 1.0  # Near bottom
        is_near_top = abs(theta) < 0.5  # Near top

        if is_hanging_down:
            # If already moving push it in the direction that it's going in 
            if abs(angvel) > 0.1:
                torque = 2.0 * np.sign(angvel)  
            else:
                torque = 2.0 * np.sign(theta)  

        elif is_near_top:
            # small adjustments if near the target position 
            kp = 12.0
            kd = 3.0
            torque = -kp * theta - kd * angvel

        else:
            kp = 5.0
            kd = 1.0
            torque = -kp * theta - kd * angvel
            # if too slow, add more torque 
            if abs(angvel) > 1.0:
                torque += 1.0 * np.sign(angvel)

        # Clip to action space bounds
        action_space = cast(gym.spaces.Box, self._action_space)
        low, high = float(action_space.low[0]), float(action_space.high[0])
        torque = float(np.clip(torque, low, high))
        print(f"torque={torque:.3f}")
        return cast(_ActType, np.array([torque], dtype=action_space.dtype))
