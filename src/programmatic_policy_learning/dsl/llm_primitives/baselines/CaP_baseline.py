"""Minimal CaP baseline that prompts an OpenAI model and saves the result."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from openai import OpenAI


@dataclass
class CaPBaselineConfig:
    """Paths and inference knobs for the CaP baseline."""

    prompt_path: str = "prompts/baselines/CaP.txt"
    model: str = "gpt-4.1"
    temperature: float = 0.0
    output_dir: str = "outputs/baselines"


class CaPBaseline:
    """Thin wrapper that loads a prompt, hits the LLM, and writes the
    policy."""

    def __init__(self, cfg: CaPBaselineConfig):
        """Store config and create an OpenAI client."""
        self.cfg = cfg
        self.client = OpenAI()

    def load_prompt(self) -> str:
        """Read the prompt file from disk."""
        p = Path(self.cfg.prompt_path)
        if not p.exists():
            raise FileNotFoundError(f"Prompt file not found: {p.resolve()}")
        return p.read_text(encoding="utf-8")

    def query_llm(self, prompt: str) -> str:
        """Send the prompt to the configured OpenAI model."""
        resp = self.client.responses.create(
            model=self.cfg.model,
            input=prompt,
            temperature=self.cfg.temperature,
        )
        return resp.output_text

    def generate_policy(self) -> str:
        """Produce, save, and echo the policy string returned by the LLM."""
        prompt = self.load_prompt()
        policy_code = self.query_llm(prompt).strip()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(self.cfg.output_dir) / f"cap_policy_{timestamp}.py"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        out_path.write_text(policy_code + "\n", encoding="utf-8")

        print("\n=== LLM policy output (also saved to file) ===\n")
        print(policy_code)
        print(f"\n[Saved to: {out_path.resolve()}]\n")

        return policy_code


if __name__ == "__main__":
    baseline_cfg = CaPBaselineConfig(
        prompt_path=(
            "src/programmatic_policy_learning/dsl/llm_primitives/"
            "prompts/baselines/CaP_nim.txt"
        ),
        model="gpt-4.1",
        temperature=0.0,
    )
    CaPBaseline(baseline_cfg).generate_policy()
