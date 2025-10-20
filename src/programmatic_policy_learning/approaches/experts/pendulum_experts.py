"""Expert policies for the pendulum environment (may not be optimal!)."""

from typing import Any, Callable

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from programmatic_policy_learning.approaches.structs import ParametricPolicyBase

Obs = NDArray[np.float32]
Act = NDArray[np.float32]


def create_manual_pendulum_policy(action_space: gym.spaces.Box) -> Callable[[Obs], Act]:
    """Create a manual pendulum policy given an action space."""

    def manual_pendulum_policy(obs: Obs) -> Act:
        """A manually defined policy for the pendulum environment."""
        currobs = obs

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
        low, high = float(action_space.low[0]), float(action_space.high[0])
        torque = float(np.clip(torque, low, high))
        return np.array([torque], dtype=action_space.dtype)

    return manual_pendulum_policy


class PendulumParametricPolicy(ParametricPolicyBase):
    """An "expert" parametric policy for the pendulum environment."""

    def __init__(
        self,
        *args: dict[str, Any],
        min_torque: float = -2.0,
        max_torque: float = 2.0,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(*args, **kwargs)
        self._min_torque = min_torque
        self._max_torque = max_torque

    def act(self, obs: Any) -> Any:
        # Extract the parameters.
        kp = self._params["kp"]
        kd = self._params["kd"]

        # Main logic.
        assert isinstance(obs, np.ndarray)
        x, y, angvel = float(obs[0]), float(obs[1]), float(obs[2])
        theta = float(np.arctan2(y, x))

        near_top = abs(theta) < 0.5
        if near_top:
            torque = -kp * theta - kd * angvel
        else:
            torque = self._max_torque if angvel >= 0.0 else self._min_torque

        # Clip.
        torque = np.clip(torque, self._min_torque, self._max_torque)

        # Return the action as a numpy arary.
        act = np.array([torque], dtype=np.float32)
        return act
