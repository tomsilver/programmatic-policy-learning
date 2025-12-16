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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


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
        """Roll out CaP and (optionally) expert policies on specified envs."""
        cap_results: list[bool] = []
        expert_results: list[bool] = []

        if self._policy is None:
            raise RuntimeError("Policy has not been generated.")

        expert_fn = get_grid_expert(self.env_name) if run_expert else None

        for i in test_env_nums:
            env = env_factory(i)
            cap_success = (
                run_single_episode(env, self._policy, max_num_steps=max_num_steps) > 0
            )
            cap_results.append(cap_success)
            env.close()

            if run_expert:
                if expert_fn is None:
                    raise RuntimeError("Expert policy unavailable.")
                env_e = env_factory(i)
                expert_success = (
                    run_single_episode(env_e, expert_fn, max_num_steps=max_num_steps)
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
    """Return mean success and half-width of a bootstrap CI."""
    rng = np.random.default_rng(seed)
    s = np.asarray(successes, dtype=float)
    N = len(s)

    boots = rng.choice(s, size=(n_boot, N), replace=True).mean(axis=1)
    lo = np.quantile(boots, alpha / 2)
    hi = np.quantile(boots, 1 - alpha / 2)
    return s.mean(), (hi - lo) / 2


def plot_expert_vs_cap(
    domains: Sequence[str],
    expert_means: Sequence[float],
    expert_cis: Sequence[float],
    cap_means: Sequence[float],
    cap_cis: Sequence[float],
    save_path: str | Path | None = None,
) -> None:
    """Bar plot comparing expert and CaP performance with error bars."""

    x = np.arange(len(domains))
    width = 0.22  # thinner bars

    _, ax = plt.subplots(figsize=(9, 4.2))

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
    domains = [
        "TwoPileNim",
        "Chase",
        "CheckmateTactic",
        "ReachForTheStar",
        "StopTheFall",
    ]
    NUM_LLM_SEEDS = 5
    TEST_ENV_NUMS = list(range(11, 20))
    MAX_NUM_STEPS = 50
    expert_means, expert_cis = [], []
    cap_means, cap_cis = [], []

    for env_name in domains:
        logging.info(f"\n=== Running CaP on {env_name} ===")

        # ------------------------------------------------------------------
        # Environment factory
        # ------------------------------------------------------------------
        def env_factory(
            instance_num: int | None = None, *, _env_name: str = env_name
        ) -> Any:
            """Env Factory."""
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

        # ------------------------------------------------------------------
        # Get example observation + action space
        # ------------------------------------------------------------------
        env0 = env_factory(0)
        obs, _ = env0.reset()
        action_space = env0.action_space
        env0.close()

        # ------------------------------------------------------------------
        # Storage
        # ------------------------------------------------------------------
        cap_all: list[bool] = []
        expert_all: list[bool] | None = None

        # ------------------------------------------------------------------
        # Loop over LLM seeds
        # ------------------------------------------------------------------
        for seed in range(NUM_LLM_SEEDS):
            logging.info(f"  CaP seed {seed}")

            try:
                cache_path = Path(f"outputs/baselines/cache_{env_name}_seed{seed}.db")
                cache_path.parent.mkdir(parents=True, exist_ok=True)

                cache = SQLite3PretrainedLargeModelCache(cache_path)
                llm_client = OpenAIModel("gpt-4.1", cache)

                cap_cfg = CaPBaselineConfig(
                    prompt_path=f"../prompts/baselines/CaP_{env_name}.txt",
                    output_dir=f"outputs/baselines/{env_name}/seed{seed}",
                )

                baseline = CaPBaseline(
                    llm_client=llm_client,
                    example_observation=obs,
                    action_space=action_space,
                    env_name=env_name,
                    cfg=cap_cfg,
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

            except (RuntimeError, ValueError, OSError) as exc:
                logging.warning(
                    "[CaP failure] env=%s, seed=%d: %s", env_name, seed, exc
                )
                continue

        # ------------------------------------------------------------------
        # Aggregate + CI
        # ------------------------------------------------------------------
        if expert_all is None or len(cap_all) == 0:
            logging.info(f"[Skipping domain] {env_name} (no valid results)")
            expert_means.append(float("nan"))
            expert_cis.append(float("nan"))
            cap_means.append(float("nan"))
            cap_cis.append(float("nan"))
            continue

        m_e, h_e = bootstrap_ci_success(expert_all)
        m_c, h_c = bootstrap_ci_success(cap_all)

        expert_means.append(100 * m_e)
        expert_cis.append(100 * h_e)
        cap_means.append(100 * m_c)
        cap_cis.append(100 * h_c)

    logging.info(f"Expert means: {expert_means}")
    logging.info(f"Expert CIs:   {expert_cis}")
    logging.info(f"CaP means:    {cap_means}")
    logging.info(f"CaP CIs:      {cap_cis}")
    valid = [i for i, m in enumerate(cap_means) if not np.isnan(m)]

    domains_plot = [domains[i] for i in valid]
    expert_means_plot = [expert_means[i] for i in valid]
    expert_cis_plot = [expert_cis[i] for i in valid]
    cap_means_plot = [cap_means[i] for i in valid]
    cap_cis_plot = [cap_cis[i] for i in valid]

    plot_expert_vs_cap(
        domains_plot,
        expert_means_plot,
        expert_cis_plot,
        cap_means_plot,
        cap_cis_plot,
        save_path=f"plots/cap_vs_expert_{len(valid)}_{NUM_LLM_SEEDS}.png",
    )


if __name__ == "__main__":
    _main()
