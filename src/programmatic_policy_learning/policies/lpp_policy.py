"""Policy class for logic programmatic policies (LPP)."""

import re
from typing import Any, Generic, Sequence, TypeVar, cast

import numpy as np
from gymnasium.spaces import Space

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
        candidate_actions: Sequence[Any] | None = None,
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
        assert len(plps) == len(probs), "plps and probs must have the same length."
        assert abs(np.sum(probs) - 1.0) < 1e-5

        self.plps = plps
        self.probs = probs
        self.map_choices = map_choices
        self.normalize_plp_actions = normalize_plp_actions
        self.action_mode = action_mode
        self.action_space = action_space
        self.continuous_num_candidates = max(1, int(continuous_num_candidates))
        self.candidate_actions = self._normalize_action_catalog(
            candidate_actions, action_space
        )
        self.rng = np.random.RandomState(seed)
        self._action_prob_cache: dict[Any, np.ndarray] = {}
        self.map_program = ""
        self.map_posterior = 0.0
        self.map_plp: Any | None = None
        self.explanation_functions: dict[str, Any] = {}

    @staticmethod
    def _action_value(action: Any) -> Any:
        """Return a plain value for enum-like actions when possible."""
        value = getattr(action, "value", action)
        try:
            return int(value)
        except (TypeError, ValueError):
            return value

    @classmethod
    def _normalize_action_catalog(
        cls,
        candidate_actions: Sequence[Any] | None,
        action_space: Space[Any] | None,
    ) -> list[Any]:
        """Build a finite action catalog from explicit candidates or spaces."""
        if candidate_actions is not None and len(candidate_actions) > 0:
            return [cls._action_value(action) for action in candidate_actions]
        actions = getattr(action_space, "actions", None)
        if actions is None:
            return []
        return [cls._action_value(action) for action in actions]

    def _uses_finite_discrete_actions(self) -> bool:
        """Whether this non-continuous policy should score explicit actions."""
        return self.action_mode != "continuous" and bool(self.candidate_actions)

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
        if self._uses_finite_discrete_actions():
            return cast(_ActType, self._select_finite_discrete_action(obs))
        action_probs = self.get_action_probs(obs).flatten()
        # print(f"Action probabilities: {action_probs}")
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
        if self.action_mode == "continuous" or self._uses_finite_discrete_actions():
            if isinstance(obs, np.ndarray):
                return ("np", str(obs.dtype), tuple(obs.shape), obs.tobytes())
            return self._make_hashable(obs)
        if not hasattr(obs, "__iter__"):
            raise NotImplementedError(
                "hash_obs assumes obs is iterable. "
                "Override this method for non-grid environments."
            )
        return tuple(tuple(l) for l in obs)  # type: ignore[attr-defined]

    @classmethod
    def _make_hashable(cls, value: Any) -> Any:
        """Convert common nested observation structures into cache keys."""
        if isinstance(value, np.ndarray):
            return ("np", str(value.dtype), tuple(value.shape), value.tobytes())
        if isinstance(value, dict):
            return (
                "dict",
                tuple(
                    sorted(
                        (
                            repr(key),
                            cls._make_hashable(item),
                        )
                        for key, item in value.items()
                    )
                ),
            )
        if isinstance(value, (list, tuple)):
            return tuple(cls._make_hashable(item) for item in value)
        if isinstance(value, set):
            return (
                "set",
                tuple(sorted((cls._make_hashable(item) for item in value), key=repr)),
            )
        try:
            hash(value)
        except TypeError:
            return ("repr", repr(value))
        return value

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
        if self._uses_finite_discrete_actions():
            return self._get_finite_discrete_action_probs(obs)
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

        if self._uses_finite_discrete_actions():
            for action in self.candidate_actions:
                try:
                    if plp(obs, action):
                        suggestions.append(cast(_ActType, action))
                except Exception:  # pylint: disable=broad-exception-caught
                    continue
            return suggestions

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
            if self._uses_finite_discrete_actions():
                action_value = self._action_value(action)
                for idx, candidate in enumerate(self.candidate_actions):
                    if self._actions_equal(candidate, action_value):
                        return float(action_probs[idx])
                return 0.0
            return float(action_probs[cast(Any, action)])
        return self._continuous_action_score(obs, action)

    @staticmethod
    def _actions_equal(left: Any, right: Any) -> bool:
        """Compare actions that may be arrays, enums, or plain scalars."""
        try:
            left_arr = np.asarray(left)
            right_arr = np.asarray(right)
            if left_arr.shape == right_arr.shape and np.array_equal(
                left_arr, right_arr
            ):
                return True
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        return left == right

    def _get_finite_discrete_action_probs(self, obs: _ObsType) -> np.ndarray:
        """Compute probabilities over an explicit finite action catalog."""
        hashed_obs = self.hash_obs(obs)
        if hashed_obs in self._action_prob_cache:
            return self._action_prob_cache[hashed_obs]

        candidates = list(self.candidate_actions)
        action_probs = np.zeros(len(candidates), dtype=np.float32)
        for plp, prob in zip(self.plps, self.probs):
            suggestions = self.get_plp_suggestions(plp, obs)
            if self.normalize_plp_actions:
                if not suggestions:
                    continue
                per_action_prob = prob / len(suggestions)
            else:
                per_action_prob = prob
            for suggestion in suggestions:
                suggestion_value = self._action_value(suggestion)
                for idx, candidate in enumerate(candidates):
                    if self._actions_equal(candidate, suggestion_value):
                        action_probs[idx] += per_action_prob
                        break

        denom = np.sum(action_probs)
        if denom == 0.0:
            action_probs += 1.0 / len(candidates)
        else:
            action_probs = action_probs / denom
        self._action_prob_cache[hashed_obs] = action_probs
        return action_probs

    def _select_finite_discrete_action(self, obs: _ObsType) -> Any:
        """Pick an action from an explicit finite action catalog."""
        if not self.candidate_actions:
            raise ValueError("candidate_actions is required for finite action policy.")

        action_probs = self.get_action_probs(obs).flatten()
        if self.map_choices:
            idx = int(np.argmax(action_probs).squeeze())
        else:
            idx = int(self.rng.choice(len(action_probs), p=action_probs))
        return self.candidate_actions[idx]

    def set_explanation_context(
        self,
        *,
        map_plp: Any,
        dsl_functions: dict[str, Any],
    ) -> None:
        """Store the learned MAP PLP and feature functions for inspection."""
        self.map_plp = map_plp
        self.explanation_functions = {
            name: fn
            for name, fn in dsl_functions.items()
            if re.fullmatch(r"f\d+", name) and callable(fn)
        }

    def explain_finite_discrete_decision(self, obs: _ObsType) -> dict[str, Any]:
        """Explain one decision over a finite discrete action catalog."""
        if not self._uses_finite_discrete_actions():
            raise ValueError("Decision explanation requires finite discrete actions.")

        candidates = list(self.candidate_actions)
        probs = self.get_action_probs(obs).flatten()
        chosen = self(obs)
        feature_ids = sorted(
            set(re.findall(r"\bf\d+\b", self.map_program)),
            key=lambda name: int(name[1:]),
        )
        action_rows: list[dict[str, Any]] = []
        for idx, action in enumerate(candidates):
            active_features: list[str] = []
            for feature_id in feature_ids:
                feature_fn = self.explanation_functions.get(feature_id)
                if feature_fn is None:
                    continue
                try:
                    if bool(feature_fn(obs, action)):
                        active_features.append(feature_id)
                except Exception:  # pylint: disable=broad-exception-caught
                    continue
            map_accepts = False
            if self.map_plp is not None:
                try:
                    map_accepts = bool(self.map_plp(obs, action))
                except Exception:  # pylint: disable=broad-exception-caught
                    map_accepts = False
            action_rows.append(
                {
                    "action": action,
                    "probability": float(probs[idx]),
                    "map_accepts": map_accepts,
                    "active_features": active_features,
                }
            )
        return {
            "chosen_action": chosen,
            "map_program": self.map_program,
            "map_posterior": float(self.map_posterior),
            "actions": action_rows,
        }

    def _find_continuous_candidate_index(self, action: _ActType) -> int | None:
        """Return the index of a candidate action if it exists in the
        catalog."""
        action_arr = np.asarray(action)
        for idx, candidate in enumerate(self.candidate_actions):
            candidate_arr = np.asarray(candidate)
            if candidate_arr.shape != action_arr.shape:
                continue
            if np.array_equal(candidate_arr, action_arr):
                return idx
        return None

    def _get_continuous_candidate_scores(self, obs: _ObsType) -> np.ndarray:
        """Score the fixed continuous candidate catalog for one observation."""
        if not self.candidate_actions:
            raise ValueError("candidate_actions is required for continuous LPPPolicy.")
        hashed_obs = self.hash_obs(obs)
        if hashed_obs in self._action_prob_cache:
            return self._action_prob_cache[hashed_obs]

        candidates = list(self.candidate_actions)
        scores = np.zeros(len(candidates), dtype=np.float64)
        for plp, prob in zip(self.plps, self.probs):
            accepted_indices: list[int] = []
            for idx, candidate in enumerate(candidates):
                try:
                    if plp(obs, candidate):
                        accepted_indices.append(idx)
                except Exception:  # pylint: disable=broad-exception-caught
                    continue
            if not accepted_indices:
                continue
            per_action_prob = float(prob)
            if self.normalize_plp_actions:
                per_action_prob /= len(accepted_indices)
            for idx in accepted_indices:
                scores[idx] += per_action_prob
        self._action_prob_cache[hashed_obs] = scores
        return scores

    def _continuous_action_score(self, obs: _ObsType, action: _ActType) -> float:
        """Score a continuous action by PLP posterior mass that accepts it."""
        candidate_idx = self._find_continuous_candidate_index(action)
        if candidate_idx is not None:
            scores = self._get_continuous_candidate_scores(obs)
            return max(1e-12, min(1.0, float(scores[candidate_idx])))

        score = 0.0
        for plp, prob in zip(self.plps, self.probs):
            try:
                if not plp(obs, action):
                    continue
                per_action_prob = float(prob)
                if self.normalize_plp_actions and self.candidate_actions:
                    allowed_count = 0
                    for candidate in self.candidate_actions:
                        try:
                            if plp(obs, candidate):
                                allowed_count += 1
                        except Exception:  # pylint: disable=broad-exception-caught
                            continue
                    if allowed_count > 0:
                        per_action_prob /= allowed_count
                score += per_action_prob
            except Exception:  # pylint: disable=broad-exception-caught
                continue
        return max(1e-12, min(1.0, score))

    def _select_continuous_action(self, obs: _ObsType) -> Any:
        """Pick a continuous action by scoring a fixed candidate catalog."""
        if not self.candidate_actions:
            raise ValueError("candidate_actions is required for continuous LPPPolicy.")

        candidates = list(self.candidate_actions)
        if not candidates:
            raise RuntimeError("No action candidates generated for continuous policy.")

        scores = self._get_continuous_candidate_scores(obs)
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
