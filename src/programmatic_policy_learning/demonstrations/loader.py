"""Demonstration loading and processing utilities."""

from typing import Any, Callable

import gym
import numpy as np

from programmatic_policy_learning.demonstrations.expert_policies import ExpertPolicies


class DemonstrationLoader:
    """Loads and processes expert demonstrations from environments."""

    def get_demo(
        self,
        base_name: str,
        expert_policy: Callable,
        env_num: int,
        max_demo_length: float = np.inf,
    ) -> list[tuple[np.ndarray[Any, Any], tuple[int, int]]]:
        """Get a single demonstration from an expert policy.

        Args:
            base_name: Environment base name (e.g., 'TwoPileNim')
            expert_policy: Function that takes state and returns action
            env_num: Environment instance number
            max_demo_length: Maximum demonstration length

        Returns:
            list of (state, action) pairs
        """
        demonstrations = []

        env = gym.make(f"{base_name}{env_num}-v0")
        layout, _ = env.reset()  # Unpack (observation, info) - changed

        t = 0
        while True:
            action = expert_policy(layout)
            demonstrations.append((layout, action))
            layout, reward, terminated, truncated, _ = env.step(action)  # changed!
            done = terminated or truncated
            t += 1
            if done or (t >= max_demo_length):
                if not reward > 0:
                    print("WARNING: demo did not succeed!")
                break

        return demonstrations

    def get_demonstrations(
        self,
        env_name: str,
        demo_numbers: tuple[int, ...] = (1, 2, 3, 4),
        max_demo_length: float = np.inf,
    ) -> list[tuple[np.ndarray, tuple[int, int]]]:
        """Get multiple demonstrations for an environment.

        Args:
            env_name: Environment name
            demo_numbers: tuple of demonstration instance numbers
            max_demo_length: Maximum length per demonstration

        Returns:
            list of (state, action) pairs from all demonstrations
        """
        expert_policy = self.get_expert_policy(env_name)
        demonstrations = []

        for i in demo_numbers:
            demonstrations += self.get_demo(env_name, expert_policy, i, max_demo_length)

        # Convert to object arrays for consistency with original code
        return [
            (np.array(layout, dtype=object), action)
            for (layout, action) in demonstrations
        ]

    def get_expert_policy(self, env_name: str) -> Callable:
        """Get the expert policy function for an environment.

        Args:
            env_name: Name of the environment

        Returns:
            Expert policy function
        """

        return ExpertPolicies.get_policy(env_name)
