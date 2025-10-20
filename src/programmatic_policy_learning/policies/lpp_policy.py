"""Policy class for logic programmatic policies (LPP)."""

from typing import Any, Generic, List, Sequence, TypeVar, cast

import numpy as np

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")


class LPPPolicy(Generic[_ObsType, _ActType]):
    """Policy for selecting actions using a set of logical programmatic
    policies (PLPs) and their probabilities."""

    def __init__(
        self,
        plps: Sequence[Any],
        probs: Sequence[float],
        seed: int = 0,
        map_choices: bool = True,
    ) -> None:
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

    def __call__(self, obs: _ObsType) -> _ActType:
        """Select an action given an observation.

        Parameters
        ----------
        obs : _ObsType
            The observation.

        Returns
        -------
        action : _ActType
            Selected action.
        """
        action_probs = self.get_action_probs(obs).flatten()
        if self.map_choices:
            idx = int(np.argmax(action_probs).squeeze())
        else:
            idx = int(self.rng.choice(len(action_probs), p=action_probs))
        # For grid-based environments, this returns (row, col)
        # For general environments, override this logic as needed
        result = np.unravel_index(idx, obs.shape)  # type: ignore[attr-defined]
        row, col = result  # pylint: disable=unbalanced-tuple-unpacking
        return cast(_ActType, (int(row), int(col)))  # type: ignore

    def hash_obs(self, obs: _ObsType) -> Any:
        """Hash an observation for caching.

        Parameters
        ----------
        obs : _ObsType
            The observation.

        Returns
        -------
        hash : Any
            Hashable representation of the observation.
        """
        if not hasattr(obs, "__iter__"):
            raise NotImplementedError(
                "hash_obs assumes obs is iterable. "
                "Override this method for non-grid environments."
            )
        return tuple(tuple(l) for l in obs)  # type: ignore[attr-defined]

    def get_action_probs(self, obs: _ObsType) -> np.ndarray:
        """Compute action probabilities for a given observation.

        Parameters
        ----------
        obs : _ObsType
            The observation.

        Returns
        -------
        action_probs : np.ndarray
            Array of action probabilities.
        """
        hashed_obs = self.hash_obs(obs)

        if hashed_obs in self._action_prob_cache:
            return self._action_prob_cache[hashed_obs]

        action_probs = np.zeros(
            obs.shape, dtype=np.float32  # type: ignore[attr-defined]
        )

        for plp, prob in zip(self.plps, self.probs):
            for action in self.get_plp_suggestions(plp, obs):
                # For grid-based environments, action is a tuple of indices
                # For general environments, override this logic as needed
                if isinstance(action, tuple) and len(action) == action_probs.ndim:
                    action_probs[action] += prob
                else:
                    raise NotImplementedError(
                        "get_action_probs assumes action is a tuple of indices\
                        for grid environments. \
                        Override this method for non-grid environments."
                    )

        denom = np.sum(action_probs)
        if denom == 0.0:
            action_probs += 1.0 / action_probs.size
        else:
            action_probs = action_probs / denom
        self._action_prob_cache[hashed_obs] = action_probs
        return action_probs

    def get_plp_suggestions(self, plp: Any, obs: _ObsType) -> List[_ActType]:
        """Get suggested actions from a PLP for a given observation.

        Parameters
        ----------
        plp : Any
            A programmatic logical policy.
        obs : _ObsType
            The observation.

        Returns
        -------
        suggestions : List[_ActType]
            List of suggested actions.
        """
        suggestions: List[_ActType] = []

        if not hasattr(obs, "shape"):
            raise NotImplementedError(
                "get_plp_suggestions assumes obs has a .shape attribute. "
                "Override this method for non-grid environments."
            )

        # For grid-based environments, actions are (row, col)
        # For general environments, override this logic as needed
        for r in range(obs.shape[0]):  # type: ignore[attr-defined]
            for c in range(obs.shape[1]):  # type: ignore[attr-defined]
                action = (r, c)
                if plp(obs, action):
                    suggestions.append(action)  # type: ignore[arg-type]

        return cast(List[_ActType], suggestions)  # cast to match the return type
