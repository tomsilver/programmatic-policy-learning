"""Policy class for logic programmatic policies (LPP)."""

from typing import Any, Generic, Sequence, TypeVar, cast

import numpy as np
from gymnasium.spaces import Box, Space

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
        normalize_plp_actions: bool = False,
        action_mode: str = "discrete",
        action_space: Space[Any] | None = None,
        continuous_num_candidates: int = 64,
    ) -> None:
        """Initialize the LPPPolicy.

        Parameters
        ----------
        plps : Sequence[Any]
            list of programmatic logical policies.
        probs : Sequence[float]
            Probabilities associated with each PLP.
        seed : int
            Random seed for stochastic choices.
        map_choices : bool
            If True, select action with highest probability; otherwise, sample.
        normalize_plp_actions : bool
            If True, normalize the action probabilities for each PLP.
        """
        assert abs(np.sum(probs) - 1.0) < 1e-5

        self.plps = plps
        self.probs = probs
        self.map_choices = map_choices
        self.normalize_plp_actions = normalize_plp_actions
        self.action_mode = action_mode
        self.action_space = action_space
        self.continuous_num_candidates = max(1, int(continuous_num_candidates))
        self.rng = np.random.RandomState(seed)
        self._action_prob_cache: dict[Any, np.ndarray] = {}
        self.map_program = ""
        self.map_posterior = 0.0

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
        if self.action_mode == "continuous":
            return cast(_ActType, self._select_continuous_action(obs))
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
        if self.action_mode == "continuous":
            if isinstance(obs, np.ndarray):
                return ("np", str(obs.dtype), tuple(obs.shape), obs.tobytes())
            return ("repr", repr(obs))
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
        if self.action_mode == "continuous":
            raise NotImplementedError(
                "get_action_probs is grid-specific. "
                "Use get_continuous_action_score/get_action_prob for continuous mode."
            )
        hashed_obs = self.hash_obs(obs)

        if hashed_obs in self._action_prob_cache:
            return self._action_prob_cache[hashed_obs]

        action_probs = np.zeros(
            obs.shape, dtype=np.float32  # type: ignore[attr-defined]
        )

        for plp, prob in zip(self.plps, self.probs):
            # for action in self.get_plp_suggestions(plp, obs):
            suggestions = self.get_plp_suggestions(plp, obs)
            if self.normalize_plp_actions:
                if not suggestions:
                    continue
                per_action_prob = prob / len(suggestions)
            else:
                per_action_prob = prob
            for action in suggestions:
                # For grid-based environments, action is a tuple of indices
                # For general environments, override this logic as needed
                if isinstance(action, tuple) and len(action) == action_probs.ndim:
                    action_probs[action] += per_action_prob
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

    def get_plp_suggestions(self, plp: Any, obs: _ObsType) -> list[_ActType]:
        """Get suggested actions from a PLP for a given observation.

        Parameters
        ----------
        plp : Any
            A programmatic logical policy.
        obs : _ObsType
            The observation.

        Returns
        -------
        suggestions : list[_ActType]
            list of suggested actions.
        """
        suggestions: list[_ActType] = []

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

        return cast(list[_ActType], suggestions)  # cast to match the return type

    def get_action_prob(self, obs: _ObsType, action: _ActType) -> float:
        """Return action probability proxy used by risk computation."""
        if self.action_mode != "continuous":
            action_probs = self.get_action_probs(obs)
            return float(action_probs[cast(Any, action)])
        return self._continuous_action_score(obs, action)

    def _continuous_action_score(self, obs: _ObsType, action: _ActType) -> float:
        """Score a continuous action by PLP posterior mass that accepts it."""
        score = 0.0
        for plp, prob in zip(self.plps, self.probs):
            try:
                if plp(obs, action):
                    score += float(prob)
            except Exception:  # pylint: disable=broad-exception-caught
                continue
        return max(1e-12, min(1.0, score))

    def _select_continuous_action(self, obs: _ObsType) -> Any:
        """Pick a continuous action by scoring sampled candidates."""
        if self.action_space is None:
            raise ValueError("action_space is required for continuous LPPPolicy.")

        candidates: list[Any] = []
        if isinstance(self.action_space, Box):
            # Include center action plus random samples from the box.
            center = ((self.action_space.low + self.action_space.high) / 2.0).astype(
                self.action_space.dtype
            )
            candidates.append(center)
        for _ in range(self.continuous_num_candidates):
            candidates.append(self.action_space.sample())

        if not candidates:
            raise RuntimeError("No action candidates generated for continuous policy.")

        scores = np.array(
            [self._continuous_action_score(obs, cast(_ActType, a)) for a in candidates],
            dtype=np.float64,
        )
        if self.map_choices:
            best_idx = int(np.argmax(scores))
            return candidates[best_idx]
        denom = float(scores.sum())
        if denom <= 0.0:
            probs = np.full(len(candidates), 1.0 / len(candidates), dtype=np.float64)
        else:
            probs = scores / denom
        idx = int(self.rng.choice(len(candidates), p=probs))
        return candidates[idx]

    def set_map_program(self, program: str, posterior: float) -> None:
        """Set the MAP program and its posterior value after it's found."""
        self.map_program = program
        self.map_posterior = posterior
