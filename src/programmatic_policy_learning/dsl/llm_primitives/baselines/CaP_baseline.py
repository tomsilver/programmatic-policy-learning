"""CaP baseline: one-shot end-to-end policy synthesis without DSL learning."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from omegaconf import DictConfig, OmegaConf
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.code import (
    FunctionOutputRepromptCheck,
    SyntaxRepromptCheck,
    synthesize_python_function_with_llm,
)
from prpl_llm_utils.models import OpenAIModel, PretrainedLargeModel
from prpl_llm_utils.reprompting import RepromptCheck
from prpl_llm_utils.structs import Query

from programmatic_policy_learning.approaches.experts.grid_experts import get_grid_expert
from programmatic_policy_learning.approaches.utils import run_single_episode
from programmatic_policy_learning.envs.registry import EnvRegistry


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
        example_observation: Any,
        action_space: Any,
        env_name: str,
        cfg: CaPBaselineConfig,
    ) -> None:
        """Store config and create an OpenAI client."""

        self.llm_client = llm_client
        self.example_observation = example_observation
        self.action_space = action_space
        self.env_name = env_name
        self.cfg = cfg
        self._policy: Callable[[Any], Any] | None = None

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

        reprompt_checks: list[RepromptCheck] = [
            SyntaxRepromptCheck(),
            FunctionOutputRepromptCheck(
                function_name,
                [(self.example_observation,)],
                [self.action_space.contains],
            ),
        ]

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

        def compile_policy_function(
            code: str,
            function_name: str,
        ) -> Any:
            """Compile Python code defining `function_name` and return the
            callable."""
            globals_dict: dict[str, Any] = {}
            locals_dict: dict[str, Any] = {}

            exec(code, globals_dict, locals_dict)  # pylint: disable=exec-used

            if function_name in locals_dict:
                fn = locals_dict[function_name]
            elif function_name in globals_dict:
                fn = globals_dict[function_name]
            else:
                raise RuntimeError(
                    f"Function '{function_name}' not found in generated code."
                )

            if not callable(fn):
                raise RuntimeError(f"'{function_name}' is not callable.")

            return fn

        policy_fn = compile_policy_function(
            policy_code_str,
            function_name=function_name,
        )
        self._policy = policy_fn
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

    def test_policy_on_envs(
        self,
        env_factory: Any,
        test_env_nums: Sequence[int] = range(11, 20),
        max_num_steps: int = 50,
    ) -> None:
        """Roll out the synthesized policy (and expert) across env
        instances."""
        accuracies: list[bool] = []
        expert_accuracies: list[bool] = []
        for i in test_env_nums:
            env = env_factory(i)
            assert self._policy is not None, "Policy not available."
            result = (
                run_single_episode(
                    env,
                    self._policy,  # type: ignore
                    max_num_steps=max_num_steps,
                )
                > 0
            )
            accuracies.append(result)
            env.close()

            # expert comparison
            expert_fn = get_grid_expert(self.env_name)
            new_env = env_factory(i)
            expert_result = (
                run_single_episode(
                    new_env,
                    expert_fn,  # type: ignore
                    max_num_steps=max_num_steps,
                )
                > 0
            )
            expert_accuracies.append(expert_result)
            new_env.close()
        logging.info(f"CaP Test Results: {accuracies}")
        logging.info(f"Expert Test Results: {expert_accuracies}")


def _main() -> None:

    cfg: DictConfig = OmegaConf.create(
        {
            "provider": "ggg",
            "make_kwargs": {
                "base_name": "TwoPileNim",
                "id": "TwoPileNim0-v0",
            },
            "instance_num": 0,
        }
    )
    registry = EnvRegistry()
    env = registry.load(cfg)
    obs, _ = env.reset()
    env_name = cfg.make_kwargs.base_name
    env_factory = lambda instance_num=None: registry.load(
        cfg, instance_num=instance_num
    )

    cache_path = Path(f"outputs/baselines/baseline_cache_{env_name}.db")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    main_llm_client = OpenAIModel("gpt-4.1", cache)

    main_cfg = CaPBaselineConfig(
        prompt_path=f"../prompts/baselines/CaP_{env_name}.txt",
        output_dir=f"outputs/baselines/{env_name}/",
        max_attempts=5,
        function_name="policy",
    )

    baseline = CaPBaseline(
        llm_client=main_llm_client,
        example_observation=obs,
        action_space=env.action_space,
        env_name=env_name,
        cfg=main_cfg,
    )

    baseline.generate_policy()
    baseline.test_policy_on_envs(env_factory, range(11, 20), max_num_steps=50)


if __name__ == "__main__":
    _main()
