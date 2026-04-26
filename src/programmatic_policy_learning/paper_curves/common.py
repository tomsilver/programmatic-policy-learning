"""Shared helpers for the paper-curves pipeline."""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf


def utc_timestamp() -> str:
    """Return a stable UTC timestamp string."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    """Create *path* and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config into a plain Python dict."""
    data = OmegaConf.load(path)
    resolved = OmegaConf.to_container(data, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f"Expected mapping config at {path}, got {type(resolved)}.")
    return cast(dict[str, Any], resolved)


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with a serializer that handles Paths."""
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    """Read JSON from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    """JSON fallback for small utility types."""
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def setup_logging(log_path: Path) -> None:
    """Log to stdout and a per-run log file."""
    ensure_dir(log_path.parent)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )


def slugify(value: str) -> str:
    """Convert a label into a filesystem-friendly slug."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value.strip()).strip("-").lower()
    return cleaned or "item"


def demo_ids_for_count(demo_id_pool: list[int], demo_count: int) -> list[int]:
    """Return the first *demo_count* ids from *demo_id_pool*."""
    if demo_count <= 0:
        raise ValueError(f"demo_count must be positive, got {demo_count}.")
    if demo_count > len(demo_id_pool):
        raise ValueError(
            "demo_count exceeds demo_id_pool size: "
            f"{demo_count} > {len(demo_id_pool)}."
        )
    return [int(each) for each in demo_id_pool[:demo_count]]


def find_result_files(results_dir: Path) -> list[Path]:
    """Return all per-run normalized result JSON files under *results_dir*."""
    if not results_dir.exists():
        return []
    return sorted(results_dir.glob("runs/*/result.json"))


def jsonable(value: Any) -> Any:
    """Convert nested values into JSON-friendly structures."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value
