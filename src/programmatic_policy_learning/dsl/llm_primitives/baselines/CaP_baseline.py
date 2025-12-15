"""CaP baseline: one-shot end-to-end policy synthesis without DSL learning."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
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
        logging.info(f"LLM model: {self.llm_client.get_id()}")

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

        # SAFETY CHECK
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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(self.cfg.output_dir) / f"cap_policy_{timestamp}.py"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(policy_code_str + "\n", encoding="utf-8")

        logging.info("\n=== LLM policy output (also saved to file) ===\n")
        logging.info(policy_code_str)
        logging.info(f"\n[Saved to: {out_path.resolve()}]\n")

        return policy_code_str

    def evaluate_raw(
        self,
        env_factory: Callable[[int | None], Any],
        test_env_nums: Sequence[int],
        max_num_steps: int,
        run_expert: bool,
    ) -> tuple[list[bool], list[bool]]:
        cap_results: list[bool] = []
        expert_results: list[bool] = []

        expert_fn = get_grid_expert(self.env_name) if run_expert else None

        for i in test_env_nums:
            env = env_factory(i)
            cap_success = (
                run_single_episode(env, self._policy, max_num_steps=max_num_steps) > 0
            )
            cap_results.append(cap_success)
            env.close()

            if run_expert:
                env_e = env_factory(i)
                expert_success = (
                    run_single_episode(
                        env_e, expert_fn, max_num_steps=max_num_steps
                    )
                    > 0
                )
                expert_results.append(expert_success)
                env_e.close()

        return cap_results, expert_results


def bootstrap_ci_success(
    successes: Sequence[bool],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    s = np.asarray(successes, dtype=float)
    N = len(s)

    boots = rng.choice(s, size=(n_boot, N), replace=True).mean(axis=1)
    lo = np.quantile(boots, alpha / 2)
    hi = np.quantile(boots, 1 - alpha / 2)
    return s.mean(), (hi - lo) / 2


def plot_expert_vs_cap(
    domains,
    expert_means,
    expert_cis,
    cap_means,
    cap_cis,
    save_path=None,
):
    
    x = np.arange(len(domains))
    width = 0.22  # thinner bars

    fig, ax = plt.subplots(figsize=(9, 4.2))

    # Expert bars
    ax.bar(
        x - width / 2,
        expert_means,
        width,
        yerr=expert_cis,
        label="Expert",
        capsize=4,
        facecolor="none",
        edgecolor="dimgray",
        linewidth=2,
    )

    # CaP bars
    ax.bar(
        x + width / 2,
        cap_means,
        width,
        yerr=cap_cis,
        label="CaP",
        capsize=4,
        facecolor="tab:blue",
        edgecolor="tab:blue",
        alpha=0.7,
    )

    ax.set_ylabel("Success rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=15)
    ax.set_ylim(0, 105)

    ax.set_title(f"Expert vs CaP performance across {len(domains)} environments")
    ax.legend(frameon=False)

    ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        logging.info(f"Saved figure to {save_path}")
    plt.show()

def _main() -> None:
    registry = EnvRegistry()
    domains = ["TwoPileNim", "Chase", "CheckmateTactic0", "ReachForTheStar", "StopTheFall"]
    NUM_LLM_SEEDS = 5
    TEST_ENV_NUMS = list(range(11, 20))
    MAX_NUM_STEPS = 50
    expert_means, expert_cis = [], []
    cap_means, cap_cis = [], []

    for env_name in domains:
        logging.info(f"\n=== Running CaP on {env_name} ===")

        def env_factory(instance_num: int | None = None, *, _env_name: str = env_name):
            return registry.load(
                OmegaConf.create(
                    {
                        "provider": "ggg",
                        "make_kwargs": {
                            "base_name": _env_name,
                            "id": f"{_env_name}0-v0",
                        },
                        "instance_num": instance_num,
                    }
                )
            )

        env0 = env_factory(0)
        obs, _ = env0.reset()
        action_space = env0.action_space
        env0.close()

        cap_all: list[bool] = []
        expert_all: list[bool] | None = None

        for seed in range(NUM_LLM_SEEDS):
            logging.info(f"  CaP seed {seed}")

            cache_path = Path(f"outputs/baselines/cache_{env_name}_seed{seed}.db")
            cache = SQLite3PretrainedLargeModelCache(cache_path)
            llm_client = OpenAIModel("gpt-4.1", cache)

            cap_cfg = CaPBaselineConfig(
                prompt_path=f"../prompts/baselines/CaP_{env_name}.txt",
                output_dir=f"outputs/baselines/{env_name}/seed{seed}",
            )

            baseline = CaPBaseline(
                llm_client, obs, action_space, env_name, cap_cfg
            )
            baseline.generate_policy()

            cap_res, exp_res = baseline.evaluate_raw(
                env_factory,
                TEST_ENV_NUMS,
                MAX_NUM_STEPS,
                run_expert=(seed == 0),
            )

            cap_all.extend(cap_res)
            if seed == 0:
                expert_all = exp_res

        assert expert_all is not None

        m_e, h_e = bootstrap_ci_success(expert_all)
        m_c, h_c = bootstrap_ci_success(cap_all)

        expert_means.append(100 * m_e)
        expert_cis.append(100 * h_e)
        cap_means.append(100 * m_c)
        cap_cis.append(100 * h_c)
    logging.info(expert_means)
    logging.info(expert_cis)
    logging.info(cap_means)
    logging.info(cap_cis)
    plot_expert_vs_cap(
        domains,
        expert_means,
        expert_cis,
        cap_means,
        cap_cis,
        save_path=Path("plots/cap_vs_expert_all_envs.png"),
    )

if __name__ == "__main__":
    _main()
