"""CaP baseline: one-shot end-to-end policy synthesis without DSL learning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.code import (
    FunctionOutputRepromptCheck,
    SyntaxRepromptCheck,
    synthesize_python_function_with_llm,
)
from prpl_llm_utils.models import OpenAIModel, PretrainedLargeModel
from prpl_llm_utils.reprompting import RepromptCheck
from prpl_llm_utils.structs import Query


@dataclass
class CaPBaselineConfig:
    """Paths and inference knobs for the CaP baseline."""

    prompt_path: str = "prompts/baselines/CaP_nim.txt"
    output_dir: str = "outputs/baselines"
    max_attempts: int = 5
    function_name: str = "policy"


class CaPBaseline:
    """Thin wrapper that loads a prompt, hits the LLM, and writes the
    policy."""

    def __init__(
        self,
        llm_client: PretrainedLargeModel,
        example_observation: Any | None,
        action_space: Any | None,
        cfg: CaPBaselineConfig,
    ) -> None:
        """Store config and create an OpenAI client."""

        self.llm_client = llm_client
        self.example_observation = example_observation
        self.action_space = action_space
        self.cfg = cfg

    def load_prompt(self) -> str:
        """Read the prompt file from disk."""
        path = Path(self.cfg.prompt_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path.resolve()}")
        return path.read_text(encoding="utf-8")

    def generate_policy(self) -> str:
        """Produce, save, and echo the policy string returned by the LLM."""
        prompt = self.load_prompt()
        function_name = self.cfg.function_name
        logging.info("LLM model: %s", self.llm_client.get_id())

        query = Query(prompt)

        reprompt_checks: list[RepromptCheck] = [SyntaxRepromptCheck()]
        if self.example_observation is not None and self.action_space is not None:
            reprompt_checks.append(
                FunctionOutputRepromptCheck(
                    function_name,
                    [(self.example_observation,)],
                    [self.action_space.contains],
                )
            )

        policy_code = synthesize_python_function_with_llm(
            function_name,
            self.llm_client,
            query,
            reprompt_checks=reprompt_checks,
        )

        policy_code_str = str(policy_code).strip()

        # SAFTEY CHECK
        if "```" in policy_code_str:
            policy_code_str = (
                policy_code_str.split("```python", 1)[1].rsplit("```", 1)[0].strip()
            )
        logging.info("Synthesized new policy:")
        logging.info(policy_code_str)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(self.cfg.output_dir) / f"cap_policy_{timestamp}.py"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(policy_code_str + "\n", encoding="utf-8")

        logging.info("\n=== LLM policy output (also saved to file) ===\n")
        logging.info(policy_code_str)
        logging.info(f"\n[Saved to: {out_path.resolve()}]\n")

        return policy_code_str


def _main() -> None:
    cache_path = Path("outputs/baselines/baseline_cache.db")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    main_llm_client = OpenAIModel("gpt-4.1", cache)

    main_cfg = CaPBaselineConfig(
        prompt_path="../prompts/baselines/CaP_nim.txt",
        output_dir="outputs/baselines",
        max_attempts=5,
        function_name="policy",
    )

    baseline = CaPBaseline(
        llm_client=main_llm_client,
        example_observation=None,
        action_space=None,
        cfg=main_cfg,
    )

    baseline.generate_policy()


if __name__ == "__main__":
    _main()
