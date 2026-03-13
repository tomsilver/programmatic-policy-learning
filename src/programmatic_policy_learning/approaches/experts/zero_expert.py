import numpy as np
from gymnasium.spaces import Box


class ZeroExpertPolicy:
    """Expert that always returns the zero action.

    Used with ResidualApproach so that:
        total_action = base_action (0) + residual
    i.e., the residual is effectively the whole policy.
    """

    def __init__(
        self,
        env_description: str,
        observation_space: Box,
        action_space: Box,
        seed: int,
    ):
        self.env_description = env_description
        self.observation_space = observation_space
        self.action_space = action_space
        self.seed = seed

        if not isinstance(action_space, Box):
            raise TypeError("ZeroExpertPolicy requires a Box action space.")

        # Precompute a valid zero action with correct shape/dtype,
        # and clip it to the Box bounds just in case.
        zeros = np.zeros(action_space.shape, dtype=np.float32)
        self._zero_action = np.clip(zeros, action_space.low, action_space.high)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Return the zero action (ignores obs)."""
        return self._zero_action
