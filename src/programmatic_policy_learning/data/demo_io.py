"""Persistence helpers for offline demonstration data."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from programmatic_policy_learning.data.demo_types import Trajectory


@dataclass(frozen=True)
class DemoRecord:
    """A saved demonstration plus metadata needed for offline reuse."""

    env_id: str
    seed: int
    trajectory: Trajectory[Any, Any]
    rewards: list[float] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def save_demo_record(path: str | Path, record: DemoRecord) -> Path:
    """Serialize a demo record to a pickle file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(record, f, protocol=4)
    return out_path


def load_demo_record(path: str | Path) -> DemoRecord:
    """Load a previously saved demo record from disk."""
    in_path = Path(path)
    with in_path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, DemoRecord):
        raise TypeError(f"Expected DemoRecord in {in_path}, got {type(payload)!r}")
    return payload


def load_demo_records_from_dir(
    root: str | Path,
    *,
    glob: str = "*.pkl",
) -> list[DemoRecord]:
    """Load all saved demo records from a directory tree."""
    root_path = Path(root)
    return [load_demo_record(path) for path in sorted(root_path.rglob(glob))]
