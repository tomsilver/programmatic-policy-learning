"""Logical Programmatic Policies implementation."""

import numpy as np


class StateActionProgram:
    """A callable object with input (state, action) and Boolean output.

    Made a class to have nice strs and pickling and to avoid redundant
    evals.
    """

    def __init__(self, program) -> None:
        """Initialize with a program string."""
        self.program = program
        self.wrapped = None

    def __call__(self, *args, **kwargs):
        """Evaluate the program on given state-action pair."""
        if self.wrapped is None:
            self.wrapped = eval("lambda s, a: " + self.program)
        return self.wrapped(*args, **kwargs)

    def __repr__(self):
        """String representation of the program."""
        return self.program

    def __str__(self):
        """String representation of the program."""
        return self.program

    def __getstate__(self):
        """Get state for pickling."""
        return self.program

    def __setstate__(self, program):
        """Set state for unpickling."""
        self.program = program
        self.wrapped = None

    def __add__(self, s):
        """Concatenate programs or strings."""
        if isinstance(s, str):
            return StateActionProgram(self.program + s)
        if isinstance(s, StateActionProgram):
            return StateActionProgram(self.program + s.program)
        raise TypeError(f"Cannot add {type(s)} to StateActionProgram")

    def __radd__(self, s):
        """Right-side concatenation of programs or strings."""
        if isinstance(s, str):
            return StateActionProgram(s + self.program)
        if isinstance(s, StateActionProgram):
            return StateActionProgram(s.program + self.program)
        raise TypeError(f"Cannot add StateActionProgram to {type(s)}")


class PLPPolicy:
    """Probabilistic Logic Program Policy.

    A policy that combines multiple logical programs (PLPs) with associated
    probabilities to make action decisions. Each program suggests actions
    for given states, and the final action is chosen based on aggregated
    probabilities.

    Attributes:
        plps (list): List of StateActionProgram objects
        probs (list): Probabilities associated with each program
        map_choices (bool): If True, use argmax; if False, sample from distribution
        rng (np.random.RandomState): Random number generator
    """

    def __init__(self, plps, probs, seed=0, map_choices=True):
        """Initialize the PLP policy."""
        assert abs(np.sum(probs) - 1.0) < 1e-5

        self.plps = plps
        self.probs = probs
        self.map_choices = map_choices
        self.rng = np.random.RandomState(seed)

        self._action_prob_cache = {}

    def __call__(self, obs):
        """Select an action for the given observation."""

        action_probs = self.get_action_probs(obs).flatten()
        if self.map_choices:
            idx = np.argmax(action_probs).squeeze()
        else:
            idx = self.rng.choice(len(action_probs), p=action_probs)
        return np.unravel_index(idx, obs.shape)

    def hash_obs(self, obs):
        """Create a hashable representation of the observation."""

        return tuple(tuple(l) for l in obs)

    def get_action_probs(self, obs):
        """Compute action probabilities for the given observation."""

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

    def get_plp_suggestions(self, plp, obs):
        """Get action suggestions from a single program."""

        suggestions = []

        for r in range(obs.shape[0]):
            for c in range(obs.shape[1]):
                if plp(obs, (r, c)):
                    suggestions.append((r, c))

        return suggestions
