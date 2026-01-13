"""Integration test for feature generation via the prompt template."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pytest
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel
from prpl_llm_utils.reprompting import query_with_reprompts
from prpl_llm_utils.structs import Query

from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based import (
    grid_hint_config,
)

llm_runs = pytest.mark.skipif("not config.getoption('runllms')")

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = (
    REPO_ROOT
    / "src"
    / "programmatic_policy_learning"
    / "dsl"
    / "llm_primitives"
    / "prompts"
    / "feature_generation"
)
PROMPT_PATH = BASE_DIR / "prompt.txt"
MODEL_NAME = "gpt-4.1"
CACHE_PATH = REPO_ROOT / "llm_cache.db"
MAX_ATTEMPTS = 3
ENV_NAME = "TwoPileNim"
ENCODING_METHOD = "enc_4"
NUM_FEATURES = 3
HINTS_ROOT = (
    REPO_ROOT
    / "src"
    / "programmatic_policy_learning"
    / "dsl"
    / "llm_primitives"
    / "hint-generation"
    / "llm-based"
    / "hints"
)


def _read_prompt(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Prompt file is empty: {path}")
    return text


def _load_env_object_types(env_name: str) -> tuple[str, ...]:
    symbol_map = grid_hint_config.SYMBOL_MAPS.get(env_name)
    if symbol_map is None:
        raise KeyError(f"No symbol map configured for {env_name}")
    return tuple(list(symbol_map.keys()) + ["None"])


def _load_hint_text(env_name: str, encoding_method: str) -> str:
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
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    llm = OpenAIModel(model_name, cache)
    query = Query(prompt)
    response = query_with_reprompts(
        llm,
        query,
        reprompt_checks=[],
        max_attempts=max_attempts,
    )
    return response.text if hasattr(response, "text") else str(response)


def _run_query() -> tuple[Path, str]:
    prompt_template = _read_prompt(PROMPT_PATH)
    object_types = _load_env_object_types(ENV_NAME)
    prompt = _fill_prompt(prompt_template, object_types)
    response_text = _query_llm(
        prompt=prompt,
        model_name=MODEL_NAME,
        cache_path=CACHE_PATH,
        max_attempts=MAX_ATTEMPTS,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{ENV_NAME}_{ENCODING_METHOD}_n{NUM_FEATURES}_{timestamp}.txt"
    output_path = output_dir / output_name
    output_path.write_text(response_text, encoding="utf-8")
    return output_path, response_text


@llm_runs
def test_query_prompt_llm_online() -> None:
    """Run the query end-to-end against the live LLM."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    output_path, response_text = _run_query()
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").strip()

    print(response_text)
    if response_text.startswith("```"):
        response_text = response_text.strip("`").lstrip("json").strip()

    payload = json.loads(response_text)
    assert isinstance(payload, dict)
    features = payload.get("features")
    assert isinstance(features, list)
    assert features, "Expected at least one feature"
    required_keys = {"id", "ast", "program", "short_rationale"}
    for feature in features:
        assert isinstance(feature, dict)
        assert required_keys.issubset(feature.keys())
