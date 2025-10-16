# programmatic_policy_learning/experts/pendulum_stupid_expert.py
import numpy as np
from gymnasium.spaces import Space, Box
from programmatic_policy_learning.approaches.pendulum_stupid_approach import create_manual_pendulum_policy

class PendulumStupidExpert:
    def __init__(self, environment_description: str, observation_space: Space, action_space: Space, seed: int):
        assert isinstance(action_space, Box)
        self._fn = create_manual_pendulum_policy(action_space)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return self._fn(np.asarray(obs))