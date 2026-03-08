"""LLM-powered Python feature generation utilities (skeleton)."""

from __future__ import annotations

import itertools
import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from prpl_llm_utils.models import PretrainedLargeModel
from prpl_llm_utils.reprompting import RepromptCheck, query_with_reprompts
from prpl_llm_utils.structs import Query

from programmatic_policy_learning.dsl.llm_primitives.hint_generation.llm_based import (
    hint_extractor,
)


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
        hint_text: str,
        num_features: int,
        encoding_method: str | None = None,
        *,
        env_name: str | None = None,
        demonstration_data: str | None = None,
    ) -> str:
        """Replace prompt placeholders with provided values."""
        rendered = template

        if "${HINT_TEXT}" in rendered:
            rendered = rendered.replace("${HINT_TEXT}", hint_text)
        if "${NUM_FEATURES}" in rendered:
            rendered = rendered.replace("${NUM_FEATURES}", str(num_features))
        if "${DEMONSTRATION_DATA}" in rendered:
            if demonstration_data is None:
                raise ValueError("demonstration_data is required for this prompt.")
            rendered = rendered.replace("${DEMONSTRATION_DATA}", demonstration_data)
        if "${DEMONSTRATION_BACKGROUND}" in rendered:
            if encoding_method is None:
                raise ValueError(
                    "encoding_method is required for demonstration background."
                )
            enc_label = str(encoding_method)
            background_dir = (
                Path(__file__).parent
                / "prompts"
                / "py_feature_gen"
                / "demo_backgrounds"
            )
            background_path = background_dir / f"{enc_label}.txt"
            background_text = background_path.read_text(encoding="utf-8").strip()
            rendered = rendered.replace("${DEMONSTRATION_BACKGROUND}", background_text)

        needs_token_map = any(
            token in rendered
            for token in (
                "${TOKEN_MAP_SECTION}",
                "${RAW_TOKEN_VALIDATION}",
                "${TOKEN_CONSTANTS_VALIDATION}",
            )
        )
        is_enc1 = encoding_method == "enc_1"

        if needs_token_map:
            if env_name is None:
                raise ValueError("env_name is required to fill token map placeholders.")

            token_map = hint_extractor.build_token_map(env_name)

            raw_chars = sorted(token_map.keys())
            canonical_list = [token_map[ch] for ch in raw_chars]
            prefixes = sorted({val.split(".")[0] for val in token_map.values()})
            prefix = prefixes[0] if len(prefixes) == 1 else None

            if is_enc1 and "${TOKEN_MAP_SECTION}" in rendered:
                lines = [
                    "---",
                    "",
                    "## TOKEN MAP (FOR UNDERSTANDING ONLY — NOT FOR CODE)",
                    "",
                    "TOKEN_MAP = {",
                ]
                for ch in raw_chars:
                    lines.append(f'  "{ch}": {token_map[ch]},')
                if len(raw_chars) > 0:
                    lines[-1] = lines[-1].rstrip(",")
                lines.extend(
                    [
                        "}",
                        "",
                        "You MUST NOT use:",
                        f"- {', '.join(canonical_list)}",
                        f"- raw characters {', '.join(repr(ch) for ch in raw_chars)}",
                        "",
                        "inside feature code.",
                    ]
                )
                token_map_section = "\n".join(lines)
            else:
                token_map_section = ""

            if "${RAW_TOKEN_VALIDATION}" in rendered:
                if is_enc1:
                    raw_token_validation = (
                        "- No raw tokens "
                        + ", ".join(repr(ch) for ch in raw_chars)
                        + " appear in comparisons."
                    )
                else:
                    raw_token_validation = "- No raw tokens appear in comparisons."
                rendered = rendered.replace(
                    "${RAW_TOKEN_VALIDATION}", raw_token_validation
                )

            if "${TOKEN_CONSTANTS_VALIDATION}" in rendered:
                if prefix is None:
                    token_constants_validation = "- No token constants appear."
                else:
                    token_constants_validation = f"- No {prefix}.* constants appear."
                rendered = rendered.replace(
                    "${TOKEN_CONSTANTS_VALIDATION}", token_constants_validation
                )

            if "${TOKEN_MAP_SECTION}" in rendered:
                rendered = rendered.replace("${TOKEN_MAP_SECTION}", token_map_section)

        unresolved = [
            "${TOKEN_MAP_SECTION}",
            "${DEMONSTRATION_DATA}",
            "${DEMONSTRATION_BACKGROUND}",
            "${HINT_TEXT}",
            "${RAW_TOKEN_VALIDATION}",
            "${TOKEN_CONSTANTS_VALIDATION}",
            "${NUM_FEATURES}",
        ]
        if any(token in rendered for token in unresolved):
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

    _PLACEHOLDER_RE = re.compile(r"\$\{(TOKEN(?:_[A-Z0-9]+)?|TOKEN\d+|DRs|K)\}")
    _DEF_RE = re.compile(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
    _QUOTED_PLACEHOLDER_RE = re.compile(
        r"(['\"])\\?\\$\\{(TOKEN(?:_[A-Z0-9]+)?|TOKEN\\d+|DRs|K)\\}\\1"
    )

    def _normalize_placeholders(self, src: str) -> str:
        # Normalize $TOKEN/$K/$DRs -> ${TOKEN}/${K}/${DRs}
        src = re.sub(r"\$(TOKEN(?:_[A-Z0-9]+)?|TOKEN\d+|DRs|K)\b", r"${\1}", src)
        # Remove quotes around placeholders (allow whitespace inside quotes)
        src = re.sub(
            r"(['\"])\s*\$\{(TOKEN(?:_[A-Z0-9]+)?|TOKEN\d+|DRs|K)\}\s*\1",
            r"${\2}",
            src,
        )
        # Remove escaped quotes around placeholders (e.g., \'\${TOKEN}\')
        src = re.sub(
            r"\\(['\"])\s*\$\{(TOKEN(?:_[A-Z0-9]+)?|TOKEN\d+|DRs|K)\}\s*\\\1",
            r"${\2}",
            src,
        )
        return src

    def _assert_no_quoted_placeholders(self, src: str) -> None:
        m = self._QUOTED_PLACEHOLDER_RE.search(src)
        if m:
            raise ValueError(f"Quoted placeholder not allowed: {m.group(0)}")

    def _extract_placeholders(self, src: str) -> list[str]:
        return sorted(set(m.group(1) for m in self._PLACEHOLDER_RE.finditer(src)))

    def _render_placeholder(self, name: str, value: Any) -> str:
        if name.startswith("TOKEN"):
            return str(value).strip()
        if name == "K":
            return str(int(value))
        if name == "DRs":
            items = ", ".join(f"({int(dr)},{int(dc)})" for dr, dc in value)
            return "[" + items + "]"
        raise ValueError(f"Unknown placeholder: {name}")

    def _substitute(self, src: str, assignment: dict[str, Any]) -> str:
        def repl(m: re.Match) -> str:
            """Replace a placeholder match with its rendered value."""
            key = m.group(1)
            return self._render_placeholder(key, assignment[key])

        return self._PLACEHOLDER_RE.sub(repl, src)

    def _rename_def(self, source: str, new_fn_name: str) -> str:
        m = self._DEF_RE.search(source)
        if not m:
            raise ValueError("Could not find a function definition line.")
        start, end = m.span(1)
        return source[:start] + new_fn_name + source[end:]

    def expand_template_payload(
        self,
        payload: dict[str, Any],
        env_name: str | None,
        *,
        start_index: int = 1,
    ) -> dict[str, Any]:
        """Expand a template payload by substituting placeholders and renaming
        IDs."""
        if env_name is None:
            raise ValueError("env_name is required to expand template payloads.")

        token_map = hint_extractor.build_token_map(env_name)
        tokens = sorted(set(token_map.values()))
        dirs8 = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]
        dirs4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        horizontal = [(0, 1), (0, -1)]
        vertical = [(1, 0), (-1, 0)]
        dr_lists = [dirs8, dirs4, horizontal, vertical]
        ks = [1, 2, 3, 5, 10, 20]

        if start_index < 1:
            raise ValueError("start_index must be >= 1.")
        expanded: list[dict[str, str]] = []
        next_id = start_index

        for feat in payload.get("features", []):
            src = self._normalize_placeholders(feat["source"])
            self._assert_no_quoted_placeholders(src)
            phs = self._extract_placeholders(src)

            choices: list[tuple[str, Sequence[Any]]] = []
            for ph in phs:
                if ph.startswith("TOKEN"):
                    choices.append((ph, tokens))
                elif ph == "DRs":
                    choices.append((ph, dr_lists))
                elif ph == "K":
                    choices.append((ph, ks))
                else:
                    raise ValueError(f"Unexpected placeholder {ph}")

            def emit(inst_src: str) -> None:
                nonlocal next_id
                fid = f"f{next_id}"
                inst_src = self._rename_def(inst_src, fid)
                expanded.append({"id": fid, "name": fid, "source": inst_src})
                next_id += 1

            if not choices:
                emit(src)
                continue

            keys = [k for k, _ in choices]
            vals = [v for _, v in choices]
            for combo in itertools.product(*vals):
                assignment = dict(zip(keys, combo))
                inst_src = self._substitute(src, assignment)
                emit(inst_src)

        return {"features": expanded}

    def fill_batch_prompt(
        self,
        template: str,
        hint_text: str,
        existing_descriptions: Sequence[str],
        batch_size: int,
        start_index: int,
    ) -> str:
        """Fill batch prompt with base placeholders plus existing feature
        info."""
        rendered = template
        # rendered = rendered.replace("${OBJECT_TYPES}", json.dumps(list(object_types)))
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
            source = self._normalize_placeholders(source)
            self._assert_no_quoted_placeholders(source)
            programs.append(source.replace("\\n", "\n"))  # LATER REMOVE
        return programs

    def _postprocess_payload_by_mode(
        self,
        payload: dict[str, Any],
        *,
        action_mode: str,
        env_name: str | None,
        start_index: int = 1,
    ) -> dict[str, Any]:
        """Apply payload post-processing based on action modality."""
        if action_mode == "discrete":
            return self.expand_template_payload(
                payload, env_name, start_index=start_index
            )
        if action_mode == "continuous":
            return payload
        raise ValueError(f"Unsupported action_mode: {action_mode!r}")

    def generate(  # pylint: disable=unused-argument
        self,
        prompt_path: str | Path,
        batch_prompt_path: str | Path | None,
        hint_text: str,
        num_features: int,
        num_batches: int | None,
        env_name: str | None = None,
        demonstration_data: str | None = None,
        encoding_method: str | None = None,
        max_attempts: int = 3,
        _seed: int = 0,
        reprompt_checks: list[RepromptCheck] | None = None,
        loading: dict[str, Any] | None = None,
        action_mode: str = "discrete",
    ) -> tuple[list[str], dict[str, Any]]:
        """Run the prompt pipeline and return (feature_programs, payload)."""
        load_offline = bool(loading and loading.get("offline", 0))
        offline_json_path = loading.get("offline_json_path") if loading else None

        if not load_offline:
            if num_batches is None or num_batches <= 0:
                num_batches = 1
            num_batches = min(num_batches, num_features)
            batch_size = math.ceil(num_features / num_batches)

            all_programs: list[str] = []
            all_descriptions: list[str] = []
            batch_payloads: list[dict[str, Any]] = []

            for batch_idx in range(num_batches):
                if batch_idx == 0:
                    current_prompt_path = prompt_path
                else:
                    if batch_prompt_path is None:
                        raise ValueError(
                            "batch_prompt_path is required when batch_size < "
                            "num_features."
                        )
                    current_prompt_path = batch_prompt_path

                remaining = num_features - batch_idx * batch_size
                current_batch_size = min(batch_size, remaining)

                prompt_template = self.read_prompt(current_prompt_path)
                if batch_idx == 0:
                    prompt = self.fill_prompt(
                        prompt_template,
                        hint_text=hint_text,
                        num_features=current_batch_size,
                        encoding_method=encoding_method,
                        env_name=env_name,
                        demonstration_data=demonstration_data,
                    )
                    # print(prompt)
                    # input()

                else:
                    prompt = self.fill_batch_prompt(
                        prompt_template,
                        hint_text=hint_text,
                        existing_descriptions=all_descriptions,
                        batch_size=current_batch_size,
                        start_index=len(all_programs) + 1,
                    )
                prompt = f"{prompt}\n\nSEED: {_seed}\n"
                # logging.info(prompt)
                template_payload = self.query_llm(
                    prompt,
                    max_attempts=max_attempts,
                    reprompt_checks=reprompt_checks,
                    seed=_seed,
                )
                prompt_label = Path(current_prompt_path).stem.replace("/", "-")
                env_label = (env_name or "unknown").replace("/", "-")
                self.write_json(
                    f"template_payload_{prompt_label}_{env_label}.json",
                    template_payload,
                )
                expanded_payload = self._postprocess_payload_by_mode(
                    template_payload,
                    action_mode=action_mode,
                    env_name=env_name,
                    start_index=1,
                )
                feature_programs = self.parse_feature_programs(expanded_payload)
                # all_descriptions.extend(self._extract_descriptions(template_payload))
                all_programs.extend(feature_programs)
                batch_payloads.append(expanded_payload)

                # if self.llm_client is not None:
                #     self.write_json(
                #         f"py_feature_payload_batch_{batch_idx + 1}.json",
                #         expanded_payload,
                #     )

            combined_payload: dict[str, Any] = {
                "features": [
                    {"id": f"f{i + 1}", "name": f"f{i + 1}", "source": src}
                    for i, src in enumerate(all_programs)
                ],
                # "descriptions": all_descriptions,
                "batches": batch_payloads,
                # "batch_size": batch_size,
                # "num_batches": num_batches,
                "total_features": len(all_programs),
            }
            if self.llm_client is not None:
                self.write_json("py_feature_payload.json", combined_payload)
            return all_programs, combined_payload

        # offline mode
        if offline_json_path is None:
            raise ValueError("offline_json_path is required when running offline.")
        payload_text = Path(offline_json_path).read_text(encoding="utf-8")
        payload = json.loads(payload_text)
        payload = self._postprocess_payload_by_mode(
            payload,
            action_mode=action_mode,
            env_name=env_name,
            start_index=1,
        )
        feature_programs = self.parse_feature_programs(payload)
        self.write_json("py_feature_payload.json", payload)
        # logging.info(feature_programs)
        return feature_programs, payload
