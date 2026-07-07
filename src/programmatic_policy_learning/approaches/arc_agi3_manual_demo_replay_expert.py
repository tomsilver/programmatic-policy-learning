"""Replay saved ARC-AGI-3 manual demonstrations as an LPP expert."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from programmatic_policy_learning.approaches.base_approach import BaseApproach
from programmatic_policy_learning.data.demo_io import (
    DemoRecord,
    load_demo_records_from_dir,
)
from programmatic_policy_learning.data.demo_types import Trajectory
from programmatic_policy_learning.envs.arc_agi3 import preprocess_arc_observation


class ArcAgi3ManualDemoReplayExpertApproach(BaseApproach[Any, Any]):
    """Expert that replays ARC demos by demo index.

    For ARC, demo_numbers are indices into the sorted saved demo files,
    not necessarily SDK seeds. This allows multiple demonstrations from
    the same official game start state.
    """

    def __init__(
        self,
        environment_description: str,
        observation_space: Any,
        action_space: Any,
        seed: int,
        *,
        demos_root: str = "manual_demos",
        env_id: str = "arc_agi3/ls20",
        demo_glob: str = "*.pkl",
    ) -> None:
        super().__init__(environment_description, observation_space, action_space, seed)
        self._records = [
            record
            for record in load_demo_records_from_dir(Path(demos_root), glob=demo_glob)
            if record.env_id == env_id
        ]
        if not self._records:
            raise ValueError(
                f"No ARC-AGI-3 manual demo records found in {demos_root!r} "
                f"for env_id={env_id!r}."
            )
        self._current_record: DemoRecord | None = None
        self._step_index = 0
        self._processed_trajectories: dict[int, Trajectory[Any, Any]] = {}

    def set_env(self, env: Any) -> None:
        """Keep interface parity with collect_demo; env state is in reset
        info."""
        del env

    def reset(self, obs: Any, info: dict[str, Any]) -> None:
        """Select the saved demo by the requested reset seed/demo index."""
        del obs
        demo_index = info.get("requested_seed", info.get("seed", 0))
        if not isinstance(demo_index, int):
            demo_index = 0
        if demo_index < 0 or demo_index >= len(self._records):
            raise IndexError(
                f"ARC demo index {demo_index} is unavailable. "
                f"Available indices: 0..{len(self._records) - 1}."
            )
        record = self._records[demo_index]
        if not record.trajectory.steps:
            raise ValueError(f"ARC demo index {demo_index} has no steps.")
        self._current_record = record
        self._step_index = 0

    def get_trajectory(self, demo_index: int) -> Trajectory[Any, Any]:
        """Return a saved trajectory by demo index."""
        if demo_index not in self._processed_trajectories:
            record = self._record_for_index(demo_index)
            game_id = str(
                record.metadata.get("game_id", self._game_id_from_record(record))
            )
            self._processed_trajectories[demo_index] = Trajectory(
                steps=[
                    (
                        preprocess_arc_observation(obs, game_id=game_id),
                        int(getattr(action, "value", action)),
                    )
                    for obs, action in record.trajectory.steps
                ]
            )
        return self._processed_trajectories[demo_index]

    def get_transition_trajectory(self, demo_index: int) -> list[tuple[Any, Any, Any]]:
        """Return saved transitions as (obs, action, next_obs) triples."""
        record = self._record_for_index(demo_index)
        steps = self.get_trajectory(demo_index).steps
        transitions: list[tuple[Any, Any, Any]] = []
        final_obs = record.metadata.get("final_observation")
        if final_obs is not None:
            final_obs = preprocess_arc_observation(
                final_obs,
                game_id=str(
                    record.metadata.get("game_id", self._game_id_from_record(record))
                ),
            )
        for idx, (obs, action) in enumerate(steps):
            if idx + 1 < len(steps):
                next_obs = steps[idx + 1][0]
            else:
                next_obs = final_obs
            transitions.append((obs, action, next_obs))
        return transitions

    @staticmethod
    def _game_id_from_record(record: DemoRecord) -> str:
        return record.env_id.rsplit("/", maxsplit=1)[-1]

    def _record_for_index(self, demo_index: int) -> DemoRecord:
        if demo_index < 0 or demo_index >= len(self._records):
            raise IndexError(
                f"ARC demo index {demo_index} is unavailable. "
                f"Available indices: 0..{len(self._records) - 1}."
            )
        record = self._records[demo_index]
        if not record.trajectory.steps:
            raise ValueError(f"ARC demo index {demo_index} has no steps.")
        return record

    def _get_action(self) -> Any:
        if self._current_record is None:
            raise RuntimeError("ARC demo replay expert must be reset before acting.")
        if self._step_index >= len(self._current_record.trajectory.steps):
            raise RuntimeError(
                "ARC demo replay expert ran out of saved actions before termination."
            )
        _obs, action = self._current_record.trajectory.steps[self._step_index]
        return int(action)

    def update(self, obs: Any, reward: float, done: bool, info: dict[str, Any]) -> None:
        del obs, reward, done, info
        self._step_index += 1
