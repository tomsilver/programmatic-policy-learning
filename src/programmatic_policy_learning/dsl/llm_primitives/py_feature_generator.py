"""LLM-powered Python feature generation utilities (skeleton)."""

from __future__ import annotations

import itertools
import json
import logging
import math
import re
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from prpl_llm_utils.models import PretrainedLargeModel
from prpl_llm_utils.reprompting import RepromptCheck, query_with_reprompts
from prpl_llm_utils.structs import Query

from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based import (
    continuous_hint_config,
)
from programmatic_policy_learning.dsl.llm_primitives.env_specs import (
    get_env_llm_spec,
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

    def _resolve_demo_background_path(
        self,
        encoding_method: str,
        *,
        action_mode: str,
        env_name: str | None = None,
    ) -> Path:
        prompt_dir = Path(__file__).parent / "prompts" / "py_feature_gen"
        domain_dir = "kinder" if action_mode == "continuous" else "ggg"
        if env_name is not None:
            canonical_name = continuous_hint_config.canonicalize_env_name(
                env_name.split("-p", maxsplit=1)[0]
            )
            env_slug = canonical_name.lower()
            candidate = (
                prompt_dir
                / "demo_backgrounds"
                / domain_dir
                / f"{encoding_method}_{env_slug}.txt"
            )
            if candidate.exists():
                return candidate

        candidate = prompt_dir / "demo_backgrounds" / domain_dir / f"{encoding_method}.txt"
        if candidate.exists():
            return candidate

        fallback = prompt_dir / "demo_backgrounds" / f"{encoding_method}.txt"
        if fallback.exists():
            return fallback

        raise FileNotFoundError(
            f"No demo background found for encoding={encoding_method!r}, "
            f"action_mode={action_mode!r}."
        )

    def _motion2d_num_passages(self, env_name: str) -> int:
        match = re.search(r"-p(\d+)", env_name)
        return int(match.group(1)) if match else 0

    def _motion2d_passage_prompt_values(self, env_name: str | None) -> tuple[str, str]:
        """Return prompt-friendly Motion2D passage variant strings."""
        if env_name is None:
            return ("p0", "no narrow passages or wall obstacles")

        base_env_name = env_name.split("-p", maxsplit=1)[0]
        canonical_name = continuous_hint_config.canonicalize_env_name(base_env_name)
        if canonical_name != "Motion2D":
            return ("unknown", "an environment-specific passage layout")

        num_passages = self._motion2d_num_passages(env_name)
        variant = f"p{num_passages}"
        if num_passages == 0:
            description = "no narrow passages or wall obstacles"
        elif num_passages == 1:
            description = (
                "there is one narrow passage formed by two wall obstacles "
                "between the robot and the target"
            )
        else:
            description = (
                f"there are {num_passages} narrow passages formed by wall "
                "obstacles between the robot and the target"
            )
        return (variant, description)

    def _build_continuous_observation_field_guide(self, env_name: str | None) -> str:
        if env_name is None:
            return "- Observation fields are environment-specific continuous values."

        base_env_name = env_name.split("-p", maxsplit=1)[0]
        canonical_name = continuous_hint_config.canonicalize_env_name(base_env_name)

        num_passages = self._motion2d_num_passages(env_name)
        try:
            obs_fields = continuous_hint_config.obs_field_names_for_kinder(
                canonical_name,
                num_passages,
            )
        except ValueError:
            return (
                "- Observation fields are object-centric continuous attributes.\n"
                "- Use the serialized object names and attributes shown in the "
                "demonstrations as the source of truth."
            )
        action_fields = continuous_hint_config.ACTION_FIELD_NAMES[canonical_name]

        obs_lines = [
            f"- obs[{idx}] = {field_name}" for idx, field_name in enumerate(obs_fields)
        ]
        action_lines = [
            f"- a[{idx}] = {field_name}" for idx, field_name in enumerate(action_fields)
        ]

        return "\n".join(
            [
                (
                    f"- Environment variant: {canonical_name}-p{num_passages}"
                    if canonical_name == "Motion2D"
                    else f"- Environment variant: {canonical_name}"
                ),
                "- When raw arrays are used, index them with the following schema:",
                *obs_lines,
                "- Action dimensions:",
                *action_lines,
                "- These raw fields are also summarized into stable "
                "object-centric names like robot, target, obstacle0, obstacle1, etc.",
            ]
        )

    def fill_prompt(
        self,
        template: str,
        hint_text: str,
        num_features: int,
        encoding_method: str | None = None,
        *,
        env_name: str | None = None,
        demonstration_data: str | None = None,
        action_mode: str = "discrete",
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
            background_path = self._resolve_demo_background_path(
                enc_label,
                action_mode=action_mode,
                env_name=env_name,
            )
            background_text = background_path.read_text(encoding="utf-8").strip()
            if "${OBSERVATION_FIELD_GUIDE}" in background_text:
                field_guide = self._build_continuous_observation_field_guide(env_name)
                background_text = background_text.replace(
                    "${OBSERVATION_FIELD_GUIDE}",
                    field_guide,
                )
            rendered = rendered.replace("${DEMONSTRATION_BACKGROUND}", background_text)

        if "${PASSAGE_VARIANT}" in rendered or "${PASSAGE_DESCRIPTION}" in rendered:
            passage_variant, passage_description = self._motion2d_passage_prompt_values(
                env_name
            )
            rendered = rendered.replace("${PASSAGE_VARIANT}", passage_variant)
            rendered = rendered.replace("${PASSAGE_DESCRIPTION}", passage_description)

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

            token_map = get_env_llm_spec(env_name).token_map()

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
            "${PASSAGE_VARIANT}",
            "${PASSAGE_DESCRIPTION}",
        ]
        if any(token in rendered for token in unresolved):
            raise ValueError("Prompt template still has unresolved variables.")
        return rendered

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

    def renumber_payload_features(
        self,
        payload: dict[str, Any],
        *,
        start_index: int = 1,
    ) -> dict[str, Any]:
        """Renumber feature ids/names and function defs to start at
        start_index."""
        if start_index < 1:
            raise ValueError("start_index must be >= 1.")

        renumbered: list[dict[str, Any]] = []
        next_id = start_index

        for feat in payload.get("features", []):
            if not isinstance(feat, dict):
                raise ValueError("Each feature must be a dict.")
            source = feat.get("source")
            if not isinstance(source, str):
                raise ValueError("Feature 'source' must be a string.")

            fid = f"f{next_id}"
            updated = dict(feat)
            updated["id"] = fid
            updated["name"] = fid
            updated["source"] = self._rename_def(
                self._normalize_placeholders(source), fid
            )
            renumbered.append(updated)
            next_id += 1

        return {"features": renumbered}

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

        token_map = get_env_llm_spec(env_name).token_map()
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

    def query_llm(
        self,
        prompt: str,
        max_attempts: int = 3,
        reprompt_checks: list[RepromptCheck] | None = None,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Query the LLM and return parsed JSON payload."""
        response_text = self.query_llm_text(
            prompt,
            max_attempts=max_attempts,
            reprompt_checks=reprompt_checks,
            seed=seed,
        )
        return json.loads(response_text)

    def query_llm_text(
        self,
        prompt: str,
        max_attempts: int = 3,
        reprompt_checks: list[RepromptCheck] | None = None,
        seed: int = 0,
    ) -> str:
        """Query the LLM and return raw response text."""
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
        return response_text

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

    def _extract_python_script(self, response_text: str) -> str:
        """Extract raw Python source from an LLM response."""
        stripped = response_text.strip()
        fence_match = re.search(r"```(?:python)?\s*(.*?)```", stripped, re.DOTALL)
        if fence_match:
            stripped = fence_match.group(1).strip()
        if not stripped:
            raise ValueError("LLM returned an empty Python generator script.")
        return stripped

    def _coerce_feature_payload(self, payload: Any) -> dict[str, Any]:
        """Validate and normalize a generated feature-library payload."""
        if isinstance(payload, list):
            payload = {"features": payload}
        if not isinstance(payload, dict):
            raise ValueError("Feature generator script must return a dict payload.")
        features = payload.get("features")
        if not isinstance(features, list):
            raise ValueError(
                "Feature generator payload must contain a 'features' list."
            )
        normalized_features: list[dict[str, Any]] = []
        for idx, feature in enumerate(features, start=1):
            if not isinstance(feature, dict):
                raise ValueError(f"Feature {idx} is not a dict.")
            source = feature.get("source")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"Feature {idx} must contain a non-empty 'source'.")
            fid = feature.get("id", f"f{idx}")
            name = feature.get("name", fid)
            normalized_features.append(
                {
                    **feature,
                    "id": str(fid),
                    "name": str(name),
                    "source": source,
                }
            )
        return {"features": normalized_features}

    def _execute_generator_script(self, script_text: str) -> dict[str, Any]:
        """Execute an LLM-produced script and return its feature payload."""
        module_globals: dict[str, Any] = {
            "__builtins__": __builtins__,
            "__name__": "__py_feature_generator_script__",
            "json": json,
            "math": math,
            "re": re,
            "types": types,
        }
        exec(script_text, module_globals)  # pylint: disable=exec-used
        build_feature_library = module_globals.get("build_feature_library")
        if not callable(build_feature_library):
            raise ValueError(
                "Generator script must define a callable build_feature_library()."
            )
        payload = build_feature_library()
        return self._coerce_feature_payload(payload)

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
        hint_text: str,
        num_features: int,
        env_name: str | None = None,
        demonstration_data: str | None = None,
        encoding_method: str | None = None,
        max_attempts: int = 3,
        _seed: int = 0,
        reprompt_checks: list[RepromptCheck] | None = None,
        loading: dict[str, Any] | None = None,
        action_mode: str = "discrete",
        generation_mode: str = "feature_payload",
    ) -> tuple[list[str], dict[str, Any]]:
        """Run the prompt pipeline and return (feature_programs, payload)."""
        load_offline = bool(loading and loading.get("offline", 0))
        offline_json_path = loading.get("offline_json_path") if loading else None

        if not load_offline:
            prompt_template = self.read_prompt(prompt_path)
            prompt = self.fill_prompt(
                prompt_template,
                hint_text=hint_text,
                num_features=num_features,
                encoding_method=encoding_method,
                env_name=env_name,
                demonstration_data=demonstration_data,
                action_mode=action_mode,
            )
            prompt = f"{prompt}\n\nSEED: {_seed}\n"
            logging.info(prompt)
            input()

            prompt_label = Path(prompt_path).stem.replace("/", "-")
            env_label = (env_name or "unknown").replace("/", "-")
            if generation_mode == "generator_script":
                script_text = self.query_llm_text(
                    prompt,
                    max_attempts=max_attempts,
                    reprompt_checks=reprompt_checks,
                    seed=_seed,
                )
                script_text = self._extract_python_script(script_text)
                if self.llm_client is not None:
                    script_path = (
                        self.output_path
                        / f"feature_generator_script_{prompt_label}_{env_label}.py"
                    )
                    script_path.write_text(script_text, encoding="utf-8")
                template_payload = self._execute_generator_script(script_text)
                expanded_payload = self._postprocess_payload_by_mode(
                    template_payload,
                    action_mode=action_mode,
                    env_name=env_name,
                    start_index=1,
                )
                logging.info(script_text)
                logging.info(expanded_payload)
            else:
                template_payload = self.query_llm(
                    prompt,
                    max_attempts=max_attempts,
                    reprompt_checks=reprompt_checks,
                    seed=_seed,
                )
                expanded_payload = self._postprocess_payload_by_mode(
                    template_payload,
                    action_mode=action_mode,
                    env_name=env_name,
                    start_index=1,
                )
            self.write_json(
                f"template_payload_{prompt_label}_{env_label}.json",
                template_payload,
            )
            feature_programs = self.parse_feature_programs(expanded_payload)
            if self.llm_client is not None:
                self.write_json("py_feature_payload.json", expanded_payload)
            return feature_programs, expanded_payload

        # offline mode
        if offline_json_path is None:
            raise ValueError("offline_json_path is required when running offline.")
        offline_path = Path(offline_json_path)
        if not offline_path.is_absolute() and not offline_path.exists():
            repo_root = Path(__file__).resolve().parents[4]
            candidate = repo_root / offline_path
            if candidate.exists():
                offline_path = candidate
        payload_text = offline_path.read_text(encoding="utf-8")
        payload = json.loads(payload_text)
        payload = self._postprocess_payload_by_mode(
            payload,
            action_mode=action_mode,
            env_name=env_name,
            start_index=1,
        )
        feature_programs = self.parse_feature_programs(payload)
        return feature_programs, payload
