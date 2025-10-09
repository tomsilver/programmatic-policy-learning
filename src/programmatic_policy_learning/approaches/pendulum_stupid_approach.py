"""A simple hardcoded approach for pendulum balancing."""

from typing import Any, Callable

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from programmatic_policy_learning.approaches.expert_approach import ExpertApproach

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


class PendulumStupidAlgorithm(ExpertApproach[Obs, Act]):
    """A hardcoded approach that tries to balance the pendulum at the top."""

    def __init__(
        self,
        environment_description: str,
        observation_space: Any,
        action_space: Any,
        seed: int,
    ) -> None:
        assert isinstance(action_space, gym.spaces.Box)
        expert_fn = create_manual_pendulum_policy(action_space)
        super().__init__(
            environment_description, observation_space, action_space, seed, expert_fn
        )
