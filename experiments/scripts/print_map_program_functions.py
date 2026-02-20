"""Extract and print feature source code for each function referenced by a MAP
program.

Temporary helper for manual testing; remove when no longer needed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

FUNC_RE = re.compile(r"\b(f\d+)\b")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_funcs(program_text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in FUNC_RE.findall(program_text):
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _index_features(payload: dict[str, Any]) -> dict[str, str]:
    feats = payload.get("features", [])
    if not isinstance(feats, list):
        raise ValueError("Expected 'features' list in JSON payload.")
    by_name: dict[str, str] = {}
    for feat in feats:
        if not isinstance(feat, dict):
            continue
        name = feat.get("name") or feat.get("id")
        source = feat.get("source")
        if isinstance(name, str) and isinstance(source, str):
            by_name[name] = source.replace("\\n", "\n")
    return by_name


def main() -> None:
    """Docstring for main."""
    if len(sys.argv) != 3:
        print(
            "Usage: python scripts/print_map_program_functions.py "
            "<features_json> <map_program_txt>",
            file=sys.stderr,
        )
        raise SystemExit(2)

    features_path = Path(sys.argv[1])
    program_path = Path(sys.argv[2])

    payload = _load_json(features_path)
    if not isinstance(payload, dict):
        raise ValueError("Expected top-level JSON dict in features file.")

    program_text = program_path.read_text(encoding="utf-8").strip()
    if not program_text:
        raise ValueError("Empty map program file.")

    ordered_names = _extract_funcs(program_text)
    by_name = _index_features(payload)

    missing: list[str] = []
    for name in ordered_names:
        src = by_name.get(name)
        if src is None:
            missing.append(name)
            continue
        print(src)
        print()

    if missing:
        print(f"Missing features: {missing}", file=sys.stderr)


if __name__ == "__main__":
    main()
