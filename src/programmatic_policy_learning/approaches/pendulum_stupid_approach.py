"""A simple hardcoded approach for pendulum balancing."""

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from programmatic_policy_learning.approaches.base_approach import BaseApproach

Obs = NDArray[np.float32]
Act = NDArray[np.float32]


class PendulumStupidAlgorithm(BaseApproach[Obs, Act]):
    """A hardcoded approach that tries to balance the pendulum at the top."""

    def _get_action(self) -> Act:

        assert (
            self._last_observation is not None
        ), "Expected a last observation before calling _get_action()."
        assert isinstance(
            self._action_space, gym.spaces.Box
        ), "PendulumStupidAlgorithm requires a Box action space."

        currobs = self._last_observation

        # Parse observation: [cos(θ), sin(θ), angular_velocity]
        obs = np.asarray(currobs, dtype=np.float32)
        x = float(obs[0])
        y = float(obs[1])
        angvel = float(obs[2])

        theta = np.arctan2(y, x)

        is_hanging_down = abs(abs(theta) - np.pi) < 1.0  # Near bottom
        is_near_top = abs(theta) < 0.5  # Near top

        if is_hanging_down:
            # If already moving push it in the direction that it's going in
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
        action_space = self._action_space
        low, high = float(action_space.low[0]), float(action_space.high[0])
        torque = float(np.clip(torque, low, high))
        return np.array([torque], dtype=action_space.dtype)
