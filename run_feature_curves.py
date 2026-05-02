"""Convenience entrypoint for the feature-curves pipeline."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def main() -> int:
    """Add the local src directory to sys.path and delegate to the package CLI."""
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    driver_module = importlib.import_module(
        "programmatic_policy_learning.paper_curves.feature_driver"
    )
    return driver_module.main()


if __name__ == "__main__":
    raise SystemExit(main())
