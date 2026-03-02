"""Merge feature JSON files into one.

Each input file must have the form:
{
  "features": [ {"id": ..., "name": ..., "source": ...}, ... ]
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_features(path: Path) -> list[dict[str, Any]]:
    """Load features from a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    features = data.get("features")
    if not isinstance(features, list):
        raise ValueError(f"{path} does not contain a 'features' list")
    return features


def main() -> None:
    """Merge all feature JSON files in CWD into merged_features.json."""
    input_dir = Path.cwd()
    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No .json files found in {input_dir}")

    merged: list[dict[str, Any]] = []
    for path in files:
        for feat in load_features(path):
            merged.append(feat)

    output_path = input_dir / "merged_features.json"
    output_path.write_text(json.dumps({"features": merged}, indent=4), encoding="utf-8")


if __name__ == "__main__":
    main()
