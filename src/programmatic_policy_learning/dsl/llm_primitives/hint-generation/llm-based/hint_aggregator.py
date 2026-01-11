import json
import logging
from pathlib import Path
from typing import Any

from prpl_llm_utils.structs import Query
from prpl_llm_utils.reprompting import RepromptCheck, query_with_reprompts
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache

from prpl_llm_utils.models import PretrainedLargeModel
from prpl_llm_utils.models import OpenAIModel  # or your model


ENV_NAMES = [
    "Chase",
    "CheckmateTactic",
    "ReachForTheStar",
    "StopTheFall",
    "TwoPileNim",
]


class HintAggregator:
    def __init__(
        self,
        llm_client: PretrainedLargeModel,
        root_dir: str,
        encoding_folder: str,
        output_name: str = "aggregated_hints.json",
    ) -> None:
        self.llm_client = llm_client
        self.root_dir = Path(root_dir)
        self.encoding_folder = encoding_folder
        self.output_name = output_name

    # ------------------------------------------------------------------
    # Filesystem helpers
    # ------------------------------------------------------------------

    def _load_latest_hint_file(self, env_name: str) -> list[str]:
        """Open the latest hint file for a given env + encoding.
        The file may be JSON or plain text with one hint per line.
        """
        enc_dir = self.root_dir / env_name / self.encoding_folder
        if not enc_dir.exists():
            raise FileNotFoundError(f"Missing directory: {enc_dir}")

        hint_files = sorted(enc_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not hint_files:
            raise FileNotFoundError(f"No hint files found in {enc_dir}")

        latest_file = hint_files[-1]
        logging.info(f"[{env_name}] Using hint file: {latest_file.name}")

        with open(latest_file, "r") as f:
            raw_text = f.read().strip()

        # First try JSON
        try:
            data = json.loads(raw_text)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                if "hints" in data:
                    return data["hints"]
                if "aggregated_hints" in data:
                    return data["aggregated_hints"]
        except json.JSONDecodeError:
            pass

        # Fallback: treat as plain text (one hint per line)
        lines = [
            line.strip()
            for line in raw_text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return lines
    # ------------------------------------------------------------------
    # LLM aggregation
    # ------------------------------------------------------------------

    def _build_prompt(self, all_hints: dict[str, list[str]]) -> str:

        prompt_template = """
You are given multiple lists of hints extracted independently from several environments.
Each list contains abstract decision-time predicates proposed from expert demonstrations.

Your task is to AGGREGATE these hints into a SINGLE, CLEAN set.

GOAL:
- Produce a compact, efficient collection of hints that:
  - removes duplicates
  - removes near-duplicates
  - collapses semantically equivalent hints
  - preserves diversity across conceptual types
  - maximizes reuse across environments

IMPORTANT: This is a *conceptual aggregation* task, not a rewriting or expansion task.

---

AGGREGATION RULES:

1. DUPLICATE REMOVAL
- Remove exact duplicates (identical wording or trivial rephrasing).

2. SEMANTIC COLLAPSING
- If multiple hints express the same underlying idea,
  keep only ONE representative.
- Prefer the most general and reusable phrasing.

3. CONCEPTUAL DIVERSITY
- Avoid keeping many hints that differ only by small variations
  (e.g., left vs right, above vs below, row vs column variants).
- Keep representatives that cover different conceptual dimensions.

4. ABSTRACTION LEVEL
- Prefer atomic, decision-time predicates.
- Remove hints that:
  - encode multi-step logic
  - embed counting, optimization, or global structure
  - depend on future states or history

5. NON-LEAKAGE
- Do NOT introduce new hints.
- Do NOT use knowledge of specific environments or game mechanics.
- Do NOT reference object names or domain semantics.
- Only reason over the provided hints.

---

OUTPUT FORMAT:

Return ONLY valid JSON with the following fields:

{
  "aggregated_hints": [
    "hint_1",
    "hint_2",
    ...
  ],
  "removed_hints": [
    "hint_a",
    "hint_b",
    ...
  ]
}

- "aggregated_hints" must contain the final kept hints.
- "removed_hints" must list all hints that were removed or merged.
- Do NOT include explanations or extra text.

---

INPUT HINT SETS:
<PASTE ALL HINT LISTS HERE>

"""
        payload = json.dumps(all_hints, indent=2)
        return prompt_template.replace("<PASTE ALL HINT LISTS HERE>", payload)

    def _query_llm(self, prompt: str) -> dict[str, Any]:
        query = Query(prompt)
        response = query_with_reprompts(
            self.llm_client,
            query,
            reprompt_checks=[],
            max_attempts=5,
        )
        return json.loads(response.text)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def aggregate(self) -> dict[str, Any]:
        """Aggregate hints across all environments and save result."""
        all_hints: dict[str, list[str]] = {}

        for env in ENV_NAMES:
            all_hints[env] = self._load_latest_hint_file(env)

        prompt = self._build_prompt(all_hints)
        aggregated = self._query_llm(prompt)

        # Save next to encoding folder (root/aggregated/)
        output_dir = self.root_dir / "aggregated"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / self.output_name
        with open(output_path, "w") as f:
            json.dump(aggregated, f, indent=2)

        logging.info(f"Aggregated hints saved to: {output_path}")
        return aggregated

if __name__ == "__main__":
    cache_path = Path("aggregate.db")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    llm_client = OpenAIModel("gpt-4.1", cache)
    root_dir=Path(__file__).parent
    aggregator = HintAggregator(
        llm_client=llm_client,
        root_dir=root_dir / "hints",
        encoding_folder="enc_4",
    )

    aggregated_hints = aggregator.aggregate()