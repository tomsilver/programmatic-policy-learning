"""Query an OpenAI LLM with a fixed prompt file and save the response."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel
from prpl_llm_utils.reprompting import query_with_reprompts
from prpl_llm_utils.structs import Query

from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based import (
    grid_hint_config,
)


def _read_prompt(path: Path) -> str:
    """Read the prompt template from disk and ensure it is non-empty."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Prompt file is empty: {path}")
    return text


BASE_DIR = Path(__file__).parent
PROMPT_PATH = BASE_DIR / "prompt.txt"
MODEL_NAME = "gpt-4.1"
CACHE_PATH = Path("llm_cache.db")
MAX_ATTEMPTS = 3
ENV_NAME = "TwoPileNim"
ENCODING_METHOD = "enc_4"
NUM_FEATURES = 3
HINTS_ROOT = BASE_DIR.parent.parent / "hint-generation" / "llm-based" / "hints"


def _load_env_object_types(env_name: str) -> tuple[str, ...]:
    """Return object types from the symbol map, plus the None sentinel."""
    symbol_map = grid_hint_config.SYMBOL_MAPS.get(env_name)
    if symbol_map is None:
        raise KeyError(f"No symbol map configured for {env_name}")
    return tuple(list(symbol_map.keys()) + ["None"])


def _load_hint_text(env_name: str, encoding_method: str) -> str:
    """Load the latest hint file for an env/encoding and return its text."""
    hint_dir = HINTS_ROOT / env_name / encoding_method
    if not hint_dir.exists():
        raise FileNotFoundError(f"Missing hint directory: {hint_dir}")
    hint_files = sorted(hint_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not hint_files:
        raise FileNotFoundError(f"No hint files found in {hint_dir}")
    latest_file = hint_files[-1]
    raw_text = latest_file.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return raw_text

    if isinstance(data, list):
        return "\n".join(str(x) for x in data)
    if isinstance(data, dict):
        if "hints" in data:
            return "\n".join(str(x) for x in data["hints"])
        if "aggregated_hints" in data:
            return "\n".join(str(x) for x in data["aggregated_hints"])
    return raw_text


def _fill_prompt(template: str, object_types: tuple[str, ...]) -> str:
    """Replace prompt placeholders with object types, hints, and counts."""
    rendered = (
        template.replace("${OBJECT_TYPES}", json.dumps(list(object_types)))
        .replace("${HINT_TEXT}", _load_hint_text(ENV_NAME, ENCODING_METHOD))
        .replace("${NUM_FEATURES}", str(NUM_FEATURES))
    )
    if (
        "${OBJECT_TYPES}" in rendered
        or "${HINT_TEXT}" in rendered
        or "${NUM_FEATURES}" in rendered
    ):
        raise ValueError("Prompt template still has unresolved variables.")
    return rendered


def _query_llm(
    prompt: str, model_name: str, cache_path: Path, max_attempts: int
) -> str:
    """Query the LLM and return the raw response text."""
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    llm = OpenAIModel(model_name, cache)
    query = Query(prompt)
    response = query_with_reprompts(
        llm,
        query,
        reprompt_checks=[],
        max_attempts=max_attempts,
    )
    # Response objects in this codebase expose `.text`, but fall back defensively.
    return response.text if hasattr(response, "text") else str(response)


def main() -> int:
    """Run the fixed prompt pipeline and write the response to disk."""
    logging.basicConfig(level="INFO")
    try:
        prompt_template = _read_prompt(PROMPT_PATH)
        object_types = _load_env_object_types(ENV_NAME)
        prompt = _fill_prompt(prompt_template, object_types)
        response_text = _query_llm(
            prompt=prompt,
            model_name=MODEL_NAME,
            cache_path=CACHE_PATH,
            max_attempts=MAX_ATTEMPTS,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logging.error("Failed to query LLM: %s", exc)
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{ENV_NAME}_{ENCODING_METHOD}_n{NUM_FEATURES}_{timestamp}.txt"
    output_path = output_dir / output_name
    output_path.write_text(response_text, encoding="utf-8")
    print(f"Wrote response to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
