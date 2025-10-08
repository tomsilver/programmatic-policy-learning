"""Policy class for logic programmatic policies (LPP)."""

from typing import Any, List, Sequence

import numpy as np


class LPPPolicy:
    """Policy class for logic programmatic policies (LPP)."""

    def __init__(
        self,
        plps: Sequence[Any],
        probs: Sequence[float],
        seed: int = 0,
        map_choices: bool = True,
    ):
        """Initialize the LPPPolicy.

        Parameters
        ----------
        plps : Sequence[Any]
            List of programmatic logical policies.
        probs : Sequence[float]
            Probabilities associated with each PLP.
        seed : int
            Random seed for stochastic choices.
        map_choices : bool
            If True, select action with highest probability; otherwise, sample.
        """
        assert abs(np.sum(probs) - 1.0) < 1e-5

        self.plps = plps
        self.probs = probs
        self.map_choices = map_choices
        self.rng = np.random.RandomState(seed)

        self._action_prob_cache: dict[Any, np.ndarray] = {}

    def __call__(self, obs: np.ndarray) -> tuple[int, int]:
        """Select an action given an observation.

        Parameters
        ----------
        obs : np.ndarray
            The observation (grid).

        Returns
        -------
        action : tuple[int, int]
            Selected action coordinates.
        """
        action_probs = self.get_action_probs(obs).flatten()
        if self.map_choices:
            idx = int(np.argmax(action_probs).squeeze())
        else:
            idx = int(self.rng.choice(len(action_probs), p=action_probs))

        row, col = np.unravel_index(  # pylint: disable=unbalanced-tuple-unpacking
            idx, obs.shape
        )
        # TODOO: handle for other than 2D grids, pylint error

        return int(row), int(col)

    def hash_obs(self, obs: np.ndarray) -> Any:
        """Hash an observation for caching.

        Parameters
        ----------
        obs : np.ndarray
            The observation.

        Returns
        -------
        hash : Any
            Hashable representation of the observation.
        """
        return tuple(tuple(l) for l in obs)

    def get_action_probs(self, obs: np.ndarray) -> np.ndarray:
        """Compute action probabilities for a given observation.

        Parameters
        ----------
        obs : np.ndarray
            The observation.

        Returns
        -------
        action_probs : np.ndarray
            Array of action probabilities.
        """
        hashed_obs = self.hash_obs(obs)
        if hashed_obs in self._action_prob_cache:
            return self._action_prob_cache[hashed_obs]

        action_probs = np.zeros(obs.shape, dtype=np.float32)

        for plp, prob in zip(self.plps, self.probs):
            for r, c in self.get_plp_suggestions(plp, obs):
                action_probs[r, c] += prob

        denom = np.sum(action_probs)
        if denom == 0.0:
            action_probs += 1.0 / (action_probs.shape[0] * action_probs.shape[1])
        else:
            action_probs = action_probs / denom
        self._action_prob_cache[hashed_obs] = action_probs
        return action_probs

    def get_plp_suggestions(self, plp: Any, obs: np.ndarray) -> List[tuple[int, int]]:
        """Get suggested actions from a PLP for a given observation.

        Parameters
        ----------
        plp : Any
            A programmatic logical policy.
        obs : np.ndarray
            The observation.

        Returns
        -------
        suggestions : List[tuple[int, int]]
            List of suggested action coordinates.
        """
        suggestions = []

        for r in range(obs.shape[0]):
            for c in range(obs.shape[1]):
                if plp(obs, (r, c)):
                    suggestions.append((r, c))

        return suggestions
