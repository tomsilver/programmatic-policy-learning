"""LLM-powered feature generation utilities."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from prpl_llm_utils.models import PretrainedLargeModel
from prpl_llm_utils.reprompting import RepromptCheck, query_with_reprompts
from prpl_llm_utils.structs import Query


class LLMFeatureGenerator:
    """Generate candidate feature payloads by querying an LLM."""

    def __init__(
        self,
        llm_client: PretrainedLargeModel | None,
        output_dir: str = "outputs/feature_generation",
    ) -> None:
        self.llm_client = llm_client
        self.base_dir = Path(__file__).parent
        self.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_path = self.base_dir / output_dir / self.run_id
        if llm_client is not None:
            self.output_path.mkdir(parents=True, exist_ok=True)

    def _read_prompt(self, prompt_path: str | Path) -> str:
        prompt_file = Path(prompt_path)
        text = prompt_file.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"Prompt file is empty: {prompt_file}")
        return text

    def _fill_prompt(
        self,
        template: str,
        object_types: Sequence[str],
        hint_text: str,
        num_features: int,
    ) -> str:
        rendered = (
            template.replace("${OBJECT_TYPES}", json.dumps(list(object_types)))
            .replace("${HINT_TEXT}", hint_text)
            .replace("${NUM_FEATURES}", str(num_features))
        )
        if (
            "${OBJECT_TYPES}" in rendered
            or "${HINT_TEXT}" in rendered
            or "${NUM_FEATURES}" in rendered
        ):
            raise ValueError("Prompt template still has unresolved variables.")
        return rendered

    def _parse_response_text(self, response_text: str) -> dict[str, Any]:
        text = response_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            snippet = text[:200].replace("\n", " ")
            raise ValueError(f"Invalid JSON response: {snippet}") from exc

    def query_llm(
        self,
        prompt: str,
        max_attempts: int = 3,
        reprompt_checks: list[RepromptCheck] | None = None,
    ) -> dict[str, Any]:
        """Query LLM with a prompt."""

        if self.llm_client is None:
            raise ValueError("LLM client is not initialized.")

        query = Query(prompt)
        response = query_with_reprompts(
            self.llm_client,
            query,
            reprompt_checks or [],
            max_attempts=max_attempts,
        )
        logging.debug("Response from LLM:")
        logging.debug(response)
        response_text = response.text if hasattr(response, "text") else str(response)
        return self._parse_response_text(response_text)

    def write_json(self, filename: str, data: dict[str, Any]) -> None:
        """Write JSON data to a file in the output directory."""
        json_path = self.output_path / filename
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4)

    def generate(
        self,
        prompt_path: str | Path,
        object_types: Sequence[str],
        hint_text: str,
        num_features: int,
        max_attempts: int = 3,
    ) -> tuple[list[str], dict[str, Any]]:
        """Generate features and return (feature_programs, payload)."""
        prompt_template = self._read_prompt(prompt_path)
        prompt = self._fill_prompt(
            prompt_template,
            object_types=object_types,
            hint_text=hint_text,
            num_features=num_features,
        )
        payload = self.query_llm(prompt, max_attempts=max_attempts)
        feature_payload = payload.get("features")
        if not isinstance(feature_payload, list):
            raise ValueError("Expected payload with a 'features' list.")
        feature_programs: list[str] = []
        for feature in feature_payload:
            if not isinstance(feature, dict) or "program" not in feature:
                raise ValueError("Each feature must be a dict with a 'program' key.")
            program = feature["program"]
            if not isinstance(program, str):
                raise ValueError("Feature 'program' must be a string.")
            feature_programs.append(program)
        if self.llm_client is not None:
            self.write_json("feature_payload.json", payload)
        return feature_programs, payload
