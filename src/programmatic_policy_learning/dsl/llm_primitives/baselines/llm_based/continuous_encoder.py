"""Helpers for encoding continuous (vector) observations into text."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ContinuousStateEncoderConfig:
    """Configuration for rendering vector observations as named fields.

    Attributes:
        obs_field_names (list[str]): Human-readable names for each dimension
            in the observation vector (e.g. ``["robot_x", "robot_y", ...]``).
        action_field_names (list[str]): Human-readable names for each dimension
            in the action vector (e.g. ``["dx", "dy", ...]``).
        precision (int): Number of decimal places when formatting floats.
        salient_indices (list[int] | None): If set, only these observation
            indices are shown in the per-step text.  ``None`` means show all.
    """

    obs_field_names: list[str]
    action_field_names: list[str]
    precision: int = 3
    salient_indices: list[int] | None = field(default=None)


class ContinuousStateEncoder:
    """Encode flat-vector observations and Box actions into readable text.

    Attributes:
        cfg (ContinuousStateEncoderConfig): Encoder configuration specifying
            field names, precision, and optional salient-index filtering.
    """

    def __init__(self, cfg: ContinuousStateEncoderConfig) -> None:
        """Initialize the encoder.

        Parameters
        ----------
        cfg : ContinuousStateEncoderConfig
            Encoder configuration specifying field names, precision,
            and optional salient-index filtering.
        """
        self.cfg = cfg

    def encode_obs(
        self,
        obs: np.ndarray,
        step_index: int,
    ) -> str:
        """Render the observation vector as named fields.

        When ``salient_indices`` is set on the config, only those indices
        are included in the per-step text.  The full vector is still
        available to :meth:`compute_deltas`.

        Parameters
        ----------
        obs : np.ndarray
            1-D float observation vector from the environment.
        step_index : int
            Zero-based timestep number shown in the header.

        Returns
        -------
        str
            Multi-line string starting with ``*** Step <n> ***``,
            followed by the named observation fields.

        Raises
        ------
        IndexError
            If any observation index (from *obs* length or
            ``salient_indices``) exceeds the length of
            ``cfg.obs_field_names``.

        Examples
        --------
        >>> cfg = ContinuousStateEncoderConfig(
        ...     obs_field_names=["x", "y"], action_field_names=["dx"],
        ... )
        >>> enc = ContinuousStateEncoder(cfg)
        >>> print(enc.encode_obs(np.array([1.5, 2.0]), step_index=0))
        *** Step 0 ***
        Observation (s_0):
          obs[0] (x)=1.500, obs[1] (y)=2.000
        """
        indices = (
            self.cfg.salient_indices
            if self.cfg.salient_indices is not None
            else range(len(obs))
        )
        entries: list[str] = []
        for i in indices:
            if i >= len(self.cfg.obs_field_names):
                raise IndexError(
                    f"Observation index {i} out of range for "
                    f"obs_field_names (length {len(self.cfg.obs_field_names)})"
                )
            val = obs[i]
            name = f"obs[{i}] ({self.cfg.obs_field_names[i]})"
            entries.append(f"{name}={float(val):.{self.cfg.precision}f}")
        obs_text = ", ".join(entries)
        return f"*** Step {step_index} ***\nObservation (s_{step_index}):\n  {obs_text}"

    def encode_action(self, action: np.ndarray | None) -> str:
        """Render a continuous action vector as a bracketed string.

        Parameters
        ----------
        action : np.ndarray | None
            1-D float action vector, or ``None`` for a terminal state.

        Returns
        -------
        str
            ``"Action: [dx=..., dy=...]"`` when *action* is an array,
            or ``"Action: None (terminal state)."`` when *action* is
            ``None``.

        Raises
        ------
        IndexError
            If the *action* vector has more dimensions than
            ``cfg.action_field_names``.

        Examples
        --------
        >>> cfg = ContinuousStateEncoderConfig(
        ...     obs_field_names=["x"], action_field_names=["dx", "dy"],
        ... )
        >>> ContinuousStateEncoder(cfg).encode_action(np.array([0.1, -0.2]))
        'Action: [dx=0.100, dy=-0.200]'
        >>> ContinuousStateEncoder(cfg).encode_action(None)
        'Action: None (terminal state).'
        """
        if action is None:
            return "Action: None (terminal state)."
        entries: list[str] = []
        for i, val in enumerate(action):
            if i >= len(self.cfg.action_field_names):
                raise IndexError(
                    f"Action index {i} out of range for "
                    f"action_field_names (length {len(self.cfg.action_field_names)})"
                )
            name = self.cfg.action_field_names[i]
            entries.append(f"{name}={float(val):.{self.cfg.precision}f}")
        return f"Action: [{', '.join(entries)}]"

    def encode_step(
        self,
        obs: np.ndarray,
        action: np.ndarray | None,
        step_index: int,
    ) -> str:
        """Combine observation and action text for one timestep.

        Parameters
        ----------
        obs : np.ndarray
            1-D float observation vector.
        action : np.ndarray | None
            1-D float action vector, or ``None`` for a terminal step.
        step_index : int
            Zero-based timestep number.

        Returns
        -------
        str
            The concatenation of :meth:`encode_obs` and
            :meth:`encode_action`, separated by a blank line.

        Raises
        ------
        IndexError
            Propagated from :meth:`encode_obs` or :meth:`encode_action`
            if the vector dimensions exceed the configured field names.
        """
        obs_text = self.encode_obs(obs, step_index)
        action_text = self.encode_action(action)
        return f"{obs_text}\n\n{action_text}"

    def compute_deltas(
        self,
        obs_t: np.ndarray,
        obs_t1: np.ndarray,
        threshold: float = 1e-4,
    ) -> list[str]:
        """Summarise the per-field changes between two consecutive
        observations.

        Only fields whose absolute delta exceeds *threshold* are included.

        Parameters
        ----------
        obs_t : np.ndarray
            1-D observation vector at timestep *t*.
        obs_t1 : np.ndarray
            1-D observation vector at timestep *t + 1*.
        threshold : float, optional
            Minimum absolute delta to report (default is 1e-4).

        Returns
        -------
        list[str]
            One string per changed field, e.g.
            ``"obs[0] (x): 1.000 -> 1.500 (delta=+0.500)"``.
            Empty list when no field changes exceed the threshold.

        Raises
        ------
        IndexError
            If the observation vectors have more dimensions than
            ``cfg.obs_field_names``.

        Examples
        --------
        >>> cfg = ContinuousStateEncoderConfig(
        ...     obs_field_names=["x", "y"], action_field_names=["dx"],
        ... )
        >>> enc = ContinuousStateEncoder(cfg)
        >>> deltas = enc.compute_deltas(np.array([1.0, 2.0]), np.array([1.5, 2.3]))
        >>> len(deltas)
        2
        >>> deltas[0]
        'obs[0] (x): 1.000 -> 1.500 (delta=+0.500)'
        >>> deltas[1]
        'obs[1] (y): 2.000 -> 2.300 (delta=+0.300)'
        """
        changes: list[str] = []
        for i in range(len(obs_t)):
            delta = float(obs_t1[i]) - float(obs_t[i])
            if abs(delta) < threshold:
                continue
            if i >= len(self.cfg.obs_field_names):
                raise IndexError(
                    f"Observation index {i} out of range for "
                    f"obs_field_names (length {len(self.cfg.obs_field_names)})"
                )
            name = f"obs[{i}] ({self.cfg.obs_field_names[i]})"
            changes.append(
                f"{name}: {float(obs_t[i]):.{self.cfg.precision}f} -> "
                f"{float(obs_t1[i]):.{self.cfg.precision}f} "
                f"(delta={delta:+.{self.cfg.precision}f})"
            )
        return changes
