"""LLM-powered Python feature generation utilities (skeleton)."""

from __future__ import annotations

import json
import logging

# import math
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from prpl_llm_utils.models import PretrainedLargeModel
from prpl_llm_utils.reprompting import RepromptCheck, query_with_reprompts
from prpl_llm_utils.structs import Query


class PyFeatureGenerator:
    """Generate candidate Python feature programs by querying an LLM."""

    def __init__(
        self,
        llm_client: PretrainedLargeModel | None,
        output_dir: str = "outputs/py_feature_generation",
    ) -> None:
        self.llm_client = llm_client
        self.base_dir = Path(__file__).parent
        self.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_path = self.base_dir / output_dir / self.run_id
        if llm_client is not None:
            self.output_path.mkdir(parents=True, exist_ok=True)

    def read_prompt(self, prompt_path: str | Path) -> str:
        """Load a prompt template from disk."""
        prompt_file = Path(prompt_path)
        text = prompt_file.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"Prompt file is empty: {prompt_file}")
        return text

    def fill_prompt(
        self,
        template: str,
        object_types: Sequence[str],
        hint_text: str,
        num_features: int,
        state_t_example: str | None = None,
        action_example: str | None = None,
        state_t1_example: str | None = None,
    ) -> str:
        """Replace prompt placeholders with provided values."""
        rendered = (
            template.replace("${OBJECT_TYPES}", json.dumps(list(object_types)))
            .replace("${HINT_TEXT}", hint_text)
            .replace("${NUM_FEATURES}", str(num_features))
        )
        if state_t_example is not None:
            rendered = rendered.replace("${STATE_T_EXAMPLE}", state_t_example)
        if action_example is not None:
            rendered = rendered.replace("${ACTION_EXAMPLE}", action_example)
        if state_t1_example is not None:
            rendered = rendered.replace("${STATE_T1_EXAMPLE}", state_t1_example)
        if (
            "${OBJECT_TYPES}" in rendered
            or "${HINT_TEXT}" in rendered
            or "${NUM_FEATURES}" in rendered
            or "${STATE_T_EXAMPLE}" in rendered
            or "${ACTION_EXAMPLE}" in rendered
            or "${STATE_T1_EXAMPLE}" in rendered
        ):
            raise ValueError("Prompt template still has unresolved variables.")
        return rendered

    def _extract_descriptions(self, payload: dict[str, Any]) -> list[str]:
        """Extract feature descriptions from a payload."""
        features = payload.get("features")
        if not isinstance(features, list):
            return []
        descriptions: list[str] = []
        for feature in features:
            if not isinstance(feature, dict):
                continue
            desc = feature.get("description", "")
            if not isinstance(desc, str):
                continue
            cleaned = " ".join(desc.strip().split())
            if cleaned:
                descriptions.append(cleaned)
        return descriptions

    def fill_batch_prompt(
        self,
        template: str,
        object_types: Sequence[str],
        hint_text: str,
        existing_descriptions: Sequence[str],
        batch_size: int,
        start_index: int,
    ) -> str:
        """Fill batch prompt with base placeholders plus existing feature
        info."""
        rendered = template
        rendered = rendered.replace("${OBJECT_TYPES}", json.dumps(list(object_types)))
        rendered = rendered.replace("${HINT_TEXT}", hint_text)
        rendered = rendered.replace("${BATCH_SIZE}", str(batch_size))
        rendered = rendered.replace("${START_INDEX}", str(start_index))

        if "${EXISTING_FEATURE_DOCSTRINGS}" in rendered:
            if existing_descriptions:
                desc_block = "\n".join(f"- {d}" for d in existing_descriptions)
            else:
                desc_block = "None"
            rendered = rendered.replace("${EXISTING_FEATURE_DOCSTRINGS}", desc_block)

        unresolved = [
            "${OBJECT_TYPES}",
            "${HINT_TEXT}",
            "${EXISTING_FEATURE_DOCSTRINGS}",
            "${BATCH_SIZE}",
            "${START_INDEX}",
        ]
        if any(token in rendered for token in unresolved):
            raise ValueError("Batch prompt template still has unresolved variables.")
        return rendered

    def query_llm(
        self,
        prompt: str,
        max_attempts: int = 3,
        reprompt_checks: list[RepromptCheck] | None = None,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Query the LLM and return parsed JSON payload."""
        if self.llm_client is None:
            raise ValueError("LLM client is not initialized.")
        query = Query(prompt, hyperparameters={"temperature": 0.0, "seed": seed})
        response = query_with_reprompts(
            self.llm_client,
            query,
            reprompt_checks or [],
            max_attempts=max_attempts,
        )
        logging.debug("Response from LLM:")
        logging.debug(response)

        response_text = response.text if hasattr(response, "text") else str(response)
        return json.loads(response_text)

    def write_json(self, filename: str, data: dict[str, Any]) -> None:
        """Write JSON data to a file in the output directory."""
        json_path = self.output_path / filename
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4)

    def parse_feature_programs(self, payload: dict[str, Any]) -> list[str]:
        """Extract feature program strings from the JSON payload."""
        features = payload.get("features")
        if not isinstance(features, list):
            raise ValueError("Expected payload with a 'features' list.")
        programs: list[str] = []
        for feature in features:
            if not isinstance(feature, dict) or "source" not in feature:
                raise ValueError("Each feature must be a dict with a 'source' key.")
            source = feature["source"]
            if not isinstance(source, str):
                raise ValueError("Feature 'source' must be a string.")
            programs.append(source.replace("\\n", "\n"))  # LATER REMOVE
        return programs

    def generate(  # pylint: disable=unused-argument
        self,
        prompt_path: str | Path,
        batch_prompt_path: str | Path | None,
        object_types: Sequence[str],
        hint_text: str,
        num_features: int,
        num_batches: int | None,
        state_t_example: str | None = None,
        action_example: str | None = None,
        state_t1_example: str | None = None,
        max_attempts: int = 3,
        _seed: int = 0,
        reprompt_checks: list[RepromptCheck] | None = None,
        offline_json_path: str | None = None,
    ) -> tuple[list[str], dict[str, Any]]:
        """Run the prompt pipeline and return (feature_programs, payload)."""

        # if num_batches is None or num_batches <= 0:
        #     num_batches = 1
        # num_batches = min(num_batches, num_features)
        # batch_size = math.ceil(num_features / num_batches)

        # all_programs: list[str] = []
        # all_descriptions: list[str] = []
        # batch_payloads: list[dict[str, Any]] = []

        # for batch_idx in range(num_batches):
        #     if batch_idx == 0:
        #         current_prompt_path = prompt_path
        #     else:
        #         if batch_prompt_path is None:
        #             raise ValueError(
        #                 "batch_prompt_path is required when batch_size < num_features."
        #             )
        #         current_prompt_path = batch_prompt_path

        #     remaining = num_features - batch_idx * batch_size
        #     current_batch_size = min(batch_size, remaining)

        #     prompt_template = self.read_prompt(current_prompt_path)
        #     if batch_idx == 0:
        #         prompt = self.fill_prompt(
        #             prompt_template,
        #             object_types=object_types,
        #             hint_text=hint_text,
        #             num_features=current_batch_size,
        #             state_t_example=state_t_example,
        #             action_example=action_example,
        #             state_t1_example=state_t1_example,
        #         )

        #     else:
        #         prompt = self.fill_batch_prompt(
        #             prompt_template,
        #             object_types=object_types,
        #             hint_text=hint_text,
        #             existing_descriptions=all_descriptions,
        #             batch_size=current_batch_size,
        #             start_index=len(all_programs) + 1,
        #         )
        #     prompt = f"{prompt}\n\nSEED: {_seed}\n"
        #     print(prompt)
        #     payload = self.query_llm(
        #         prompt,
        #         max_attempts=max_attempts,
        #         reprompt_checks=reprompt_checks,
        #         seed=_seed,
        #     )
        #     feature_programs = self.parse_feature_programs(payload)
        #     all_descriptions.extend(self._extract_descriptions(payload))
        #     all_programs.extend(feature_programs)
        #     batch_payloads.append(payload)

        #     if self.llm_client is not None:
        #         self.write_json(
        #             f"py_feature_payload_batch_{batch_idx + 1}.json", payload
        #         )

        # combined_payload: dict[str, Any] = {
        #     "features": [
        #         {"id": f"f{i + 1}", "name": f"f{i + 1}", "source": src}
        #         for i, src in enumerate(all_programs)
        #     ],
        #     "descriptions": all_descriptions,
        #     "batches": batch_payloads,
        #     "batch_size": batch_size,
        #     "num_batches": num_batches,
        #     "total_features": len(all_programs),
        # }
        # if self.llm_client is not None:
        #     self.write_json("py_feature_payload.json", combined_payload)
        # return all_programs, combined_payload
        if offline_json_path is None:
            raise ValueError("offline_json_path is required when running offline.")
        payload_text = Path(offline_json_path).read_text(encoding="utf-8")
        payload = json.loads(payload_text)
        feature_programs = self.parse_feature_programs(payload)

        # print(feature_programs)
        return feature_programs, payload
