"""Expert approach that replays saved demonstration actions by seed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from programmatic_policy_learning.approaches.base_approach import BaseApproach
from programmatic_policy_learning.data.demo_io import (
    DemoRecord,
    load_demo_records_from_dir,
)


class ManualDemoReplayExpertApproach(BaseApproach[Any, Any]):
    """Stateful expert that replays a saved demo trajectory for the current
    seed."""

    def __init__(
        self,
        environment_description: str,
        observation_space: Any,
        action_space: Any,
        seed: int,
        *,
        demos_root: str = "manual_demos",
        env_id: str | None = None,
        demo_glob: str = "*.pkl",
        observation_tolerance: float = 1e-5,
    ) -> None:
        super().__init__(environment_description, observation_space, action_space, seed)
        self._env_id = env_id
        self._observation_tolerance = float(observation_tolerance)
        self._env: Any | None = None
        self._current_record: DemoRecord | None = None
        self._step_index = 0
        self._demo_records_by_seed: dict[int, DemoRecord] = {}

        records = load_demo_records_from_dir(Path(demos_root), glob=demo_glob)
        for record in records:
            if env_id is not None and record.env_id != env_id:
                continue
            if record.seed in self._demo_records_by_seed:
                raise ValueError(
                    "Duplicate manual demo found for seed "
                    f"{record.seed} under {demos_root}."
                )
            self._demo_records_by_seed[record.seed] = record
        if not self._demo_records_by_seed:
            raise ValueError(
                f"No manual demo records found in {demos_root!r}"
                + (f" for env_id={env_id!r}" if env_id is not None else "")
                + "."
            )

    def set_env(self, env: Any) -> None:
        """Remember the live env wrapper so reset() can inspect its seed."""
        self._env = env

    def _resolve_seed(self, info: dict[str, Any]) -> int:
        if isinstance(info.get("seed"), int):
            return int(info["seed"])
        if self._env is not None and hasattr(self._env, "last_reset_seed"):
            seed = self._env.last_reset_seed
            if isinstance(seed, int):
                return seed
        raise ValueError(
            "ManualDemoReplayExpertApproach could not determine the current env seed."
        )

    def reset(self, obs: Any, info: dict[str, Any]) -> None:
        """Select the saved demonstration corresponding to the current seed."""
        seed = self._resolve_seed(info)
        try:
            record = self._demo_records_by_seed[seed]
        except KeyError as exc:
            available = sorted(self._demo_records_by_seed)
            raise KeyError(
                f"No saved manual demo available for seed {seed}. "
                f"Available seeds: {available}"
            ) from exc
        if record.env_id != self._env_id and self._env_id is not None:
            raise ValueError(
                "Loaded demo env_id="
                f"{record.env_id!r} does not match expected {self._env_id!r}."
            )
        if not record.trajectory.steps:
            raise ValueError(f"Saved manual demo for seed {seed} contains no steps.")

        first_obs = np.asarray(record.trajectory.steps[0][0], dtype=np.float32)
        current_obs = np.asarray(obs, dtype=np.float32)
        if first_obs.shape != current_obs.shape or not np.allclose(
            first_obs,
            current_obs,
            atol=self._observation_tolerance,
        ):
            raise ValueError(
                "Reset observation for seed "
                f"{seed} does not match the saved manual demo."
            )
        self._current_record = record
        self._step_index = 0

    def _get_action(self) -> Any:
        if self._current_record is None:
            raise RuntimeError("Manual demo replay expert must be reset before acting.")
        if self._step_index >= len(self._current_record.trajectory.steps):
            raise RuntimeError(
                "Manual demo replay expert ran out of saved actions before termination."
            )
        _obs, action = self._current_record.trajectory.steps[self._step_index]
        return np.asarray(action, dtype=np.float32).copy()

    def update(self, obs: Any, reward: float, done: bool, info: dict[str, Any]) -> None:
        del obs, reward, done, info
        self._step_index += 1
