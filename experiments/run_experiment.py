"""Script for running experiments with hydra."""

import logging
import pdb
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from gymnasium.core import Env
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from prpl_utils.utils import sample_seed_from_rng

from programmatic_policy_learning.approaches.base_approach import BaseApproach
from programmatic_policy_learning.approaches.lpp_utils.lpp_plotting_utils import (
    plot_policy_vector_fields,
)
from programmatic_policy_learning.envs.registry import EnvRegistry
from programmatic_policy_learning.visualization.arc_agi3_plp_trace import (
    generate_arc_lpp_decision_trace,
)

_MODE_TO_ID = {
    "discrete": 1,
    "continuous": 2,
    "hybrid": 3,
}


def _maybe_debug_environment(cfg: DictConfig, env: Any) -> None:
    """Optionally stop after environment construction for interactive inspection."""
    if not bool(OmegaConf.select(cfg, "debug.pdb_after_env_load", default=False)):
        return
    print(
        "Entering pdb after environment creation. Inspect: "
        "type(env), env.observation_format, env.action_space, "
        "env.get_action_values()."
    )
    pdb.set_trace()


def _infer_mode_from_provider(
    provider: str | None, *, kind: str, configured: str | None
) -> str:
    """Infer observation/action mode with provider defaults."""
    if configured in _MODE_TO_ID:
        return configured
    if provider in {"ggg"}:
        return "discrete"
    if provider in {"kinder"}:
        return "continuous"
    logging.warning(
        "Unknown %s_mode '%s' for provider '%s'; defaulting to discrete.",
        kind,
        configured,
        provider,
    )
    return "discrete"


def instantiate_approach(
    cfg: DictConfig, env: Any, registry: EnvRegistry
) -> BaseApproach:
    """Instantiate the approach based on the configuration.

    Handles specific parameters required for certain approaches like `lpp`.
    """

    env_factory = lambda instance_num: registry.load(cfg.env, instance_num=instance_num)
    expert_cfg = OmegaConf.select(cfg, "env.expert", default=None)
    if expert_cfg is None:
        expert_cfg = OmegaConf.select(cfg, "expert", default=None)
    # expert_seed = OmegaConf.select(
    #     cfg,
    #     "env.expert_seed",
    #     default=OmegaConf.select(cfg, "expert_seed", default=0),
    # )

    approach_target = str(OmegaConf.select(cfg, "approach._target_", default=""))
    is_lpp_approach = cfg.approach_name == "lpp" or approach_target.endswith(
        ".LogicProgrammaticPolicyApproach"
    )

    if is_lpp_approach:
        if expert_cfg is None:
            raise ValueError(
                "Missing expert config. Set env.expert or top-level expert."
            )

        if not hasattr(env, "get_object_types"):
            object_types = []
        else:
            object_types = env.get_object_types()
        if not hasattr(env, "get_action_types"):
            action_types: list[str] | tuple[str, ...] = []
        else:
            action_types = env.get_action_types()

        provider = OmegaConf.select(cfg, "env.provider", default=None)
        observation_mode = _infer_mode_from_provider(
            provider,
            kind="observation",
            configured=OmegaConf.select(cfg, "env.observation_mode", default=None),
        )
        action_mode = _infer_mode_from_provider(
            provider,
            kind="action",
            configured=OmegaConf.select(cfg, "env.action_mode", default=None),
        )
        env_specs = {
            "object_types": object_types,
            "action_types": action_types,
            "observation_mode": observation_mode,
            "action_mode": action_mode,
            "observation_mode_id": _MODE_TO_ID[observation_mode],
            "action_mode_id": _MODE_TO_ID[action_mode],
        }
        if provider == "arc_agi3":
            env_specs["domain"] = "arc_agi3"
            get_action_values = getattr(env, "get_action_values", None)
            if callable(get_action_values):
                env_specs["action_values"] = tuple(get_action_values())

        expert = hydra.utils.instantiate(
            expert_cfg,
            cfg.env.description,
            env.observation_space,
            env.action_space,
            # Same expert seed for now; switch to expert_seed if needed.
            cfg.seed,
            # expert_seed,
        )

        # Instantiate the approach with additional parameters.
        return hydra.utils.instantiate(
            cfg.approach,
            cfg.env.description,
            env.observation_space,
            env.action_space,
            cfg.seed,
            expert,
            env_factory,
            cfg.env.make_kwargs.base_name,
            env_specs=env_specs,
        )

    if cfg.approach_name == "fcn":
        if expert_cfg is None:
            raise ValueError(
                "Missing expert config. Set env.expert or top-level expert."
            )

        object_types: list[str] | tuple[str, ...]
        if not hasattr(env, "get_object_types"):
            object_types = []
        else:
            object_types = env.get_object_types()

        expert = hydra.utils.instantiate(
            expert_cfg,
            cfg.env.description,
            env.observation_space,
            env.action_space,
            cfg.seed,
        )

        return hydra.utils.instantiate(
            cfg.approach,
            cfg.env.description,
            env.observation_space,
            env.action_space,
            cfg.seed,
            expert,
            env_factory,
            object_types=object_types,
        )

    # Handle residual learning.
    if cfg.approach_name == "residual":
        if expert_cfg is None:
            raise ValueError(
                "Missing expert config. Set env.expert or top-level expert."
            )

        expert = hydra.utils.instantiate(
            expert_cfg,
            cfg.env.description,
            env.observation_space,
            env.action_space,
            # Same expert seed for now; switch to expert_seed if needed.
            cfg.seed,
            # expert_seed,
        )

        return hydra.utils.instantiate(
            cfg.approach,
            cfg.env.description,
            env.observation_space,
            env.action_space,
            cfg.seed,
            expert,
            env_factory,
        )

    # Default instantiation for other approaches.
    return hydra.utils.instantiate(
        cfg.approach,
        cfg.env.description,
        env.observation_space,
        env.action_space,
        cfg.seed,
        env_factory,
    )


def evaluate_single(
    cfg: DictConfig, env_cfg: DictConfig, dsl_cfg: DictConfig, seed: int
) -> tuple[dict, float]:
    """Evaluate a single environment, DSL, and seed combination."""
    score = {}
    np.random.seed(seed)
    registry = EnvRegistry()
    env = registry.load(env_cfg)
    # _maybe_debug_environment(cfg, env)

    # dynamically update cfg with the specific settings for approach
    run_cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "seed": seed,
                "approach": {
                    "program_generation": {
                        "strategy": dsl_cfg.strategy,
                        **(
                            {"removed_primitive": dsl_cfg.removed_primitive}
                            if "removed_primitive" in dsl_cfg
                            else {}
                        ),
                        **(
                            {"dsl_generator_prompt": dsl_cfg.dsl_generator_prompt}
                            if "dsl_generator_prompt" in dsl_cfg
                            else {}
                        ),
                    }
                },
            }
        ),
    )
    if not isinstance(run_cfg, DictConfig):
        raise TypeError("run_cfg must be a DictConfig")
    approach = instantiate_approach(run_cfg, env, registry)
    rng = np.random.default_rng(seed)

    metrics: list[dict[str, float]] = []
    for eval_episode in range(cfg.num_eval_episodes):
        episode_metrics = _run_single_episode_evaluation(
            approach,
            env,
            rng,
            max_eval_steps=cfg.max_eval_steps,
        )
        episode_metrics["eval_episode"] = eval_episode
        metrics.append(episode_metrics)

    # Aggregate and save results.
    df = pd.DataFrame(metrics)
    score["train"] = df["total_rewards"].mean()
    logging.info(df)

    # Test the approach on new envs
    if hasattr(approach, "test_policy_on_envs"):
        test_accuracies = approach.test_policy_on_envs(
            base_class_name=cfg.env.make_kwargs.base_name,
            test_env_nums=range(10, 20),
            max_num_steps=50,
            record_videos=bool(
                OmegaConf.select(cfg, "eval.record_videos", default=False)
            ),
            video_format=str(OmegaConf.select(cfg, "eval.video_format", default="mp4")),
        )
        logging.info(test_accuracies)
    else:
        logging.warning(
            f"Approach {cfg.approach_name} does not support `test_policy_on_envs`."
        )
    num_correct_test = 0
    for each in test_accuracies:
        if each is True:
            num_correct_test += 1
    score["test"] = num_correct_test // len(test_accuracies)

    map_posterior = (
        # pylint: disable=protected-access
        approach._policy.map_posterior  # type: ignore[attr-defined]
    )
    return (
        score,
        map_posterior,
    )


def evaluate_all(cfg: DictConfig) -> None:
    """Evaluate all environments and DSL variants specified in the
    configuration."""
    dsl_name = cfg.dsl_name

    seed = cfg.seed
    dsl_cfg = cfg.eval.dsl_variants[dsl_name]
    env_name = cfg.env.make_kwargs.base_name

    logging.info(f"Running env={env_name}, dsl={dsl_name}, seed={seed}")

    try:
        score, map_posterior = evaluate_single(cfg, cfg.env, dsl_cfg, seed)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error(
            f"Error during evaluation: env={env_name}, dsl={dsl_name}, "
            f"seed={seed}. Exception: {e}",
            exc_info=True,
        )

        # Save error result so merging later won’t break
        out = pd.DataFrame(
            [
                {
                    "env": env_name,
                    "dsl": dsl_name,
                    "seed": seed,
                    "score": f"ERROR: {e}",
                    "map_posterior": None,
                }
            ]
        )

        out_path = (
            f"logs/{cfg.name_of_removed_func}/{env_name}/"
            f"{cfg.approach.program_generation_step_size}_"
            f"{cfg.approach.num_programs}_{len(cfg.approach.demo_numbers)}/"
            f"{dsl_name}_{seed}_result.csv"
        )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        out.to_csv(out_path, index=False)
        logging.info(f"Wrote error marker to {out_path}")
        return  # don't continue evaluating anything else

    # If no errors:
    out = pd.DataFrame(
        [
            {
                "env": env_name,
                "dsl": dsl_name,
                "seed": seed,
                "score": score,
                "map_posterior": map_posterior,
            }
        ]
    )
    out_path = (
        f"logs/{cfg.name_of_removed_func}/{env_name}/"
        f"{cfg.approach.program_generation_step_size}_{cfg.approach.num_programs}_"
        f"{len(cfg.approach.demo_numbers)}/{dsl_name}_{seed}_result.csv"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(out_path, index=False)
    logging.info(f"Saved result to {out_path}")


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:
    logging.info(
        "Approach config (cfg.approach):\n%s",
        OmegaConf.to_yaml(cfg.approach, resolve=True),
    )

    if cfg.eval.mode == 1:
        evaluate_all(cfg)
    else:
        logging.info(
            f"Running seed={cfg.seed}, env={cfg.env_name}, approach={cfg.approach_name}"
        )
        registry = EnvRegistry()

        env = registry.load(cfg.env)
        # _maybe_debug_environment(cfg, env)

        # Instantiate the approach
        approach = instantiate_approach(cfg, env, registry)

        # Evaluate.
        rng = np.random.default_rng(cfg.seed)

        metrics: list[dict[str, float]] = []
        for eval_episode in range(cfg.num_eval_episodes):
            episode_metrics = _run_single_episode_evaluation(
                approach,
                env,
                rng,
                max_eval_steps=cfg.max_eval_steps,
            )
            episode_metrics["eval_episode"] = eval_episode
            metrics.append(episode_metrics)

        # Aggregate and save results.
        df = pd.DataFrame(metrics)
        logging.info(df)
        _maybe_generate_plp_visualization(cfg, approach, registry)

        # Test the approach on new envs
        if bool(OmegaConf.select(cfg, "eval.skip_policy_tests", default=False)):
            logging.info("Skipping policy train/test sweeps.")
        elif hasattr(approach, "test_policy_on_envs"):
            train_accuracies = approach.test_policy_on_envs(
                base_class_name=cfg.env.make_kwargs.base_name,
                test_env_nums=range(0, 10),
                max_num_steps=1000,
                record_videos=bool(
                    OmegaConf.select(cfg, "eval.record_videos", default=False)
                ),
                video_format=str(
                    OmegaConf.select(cfg, "eval.video_format", default="mp4")
                ),
            )
            logging.info(train_accuracies)
            # logging.info(df["total_rewards"].iloc[0])
            logging.info(sum(train_accuracies) / len(train_accuracies))
            if bool(OmegaConf.select(cfg, "eval.vector_field.enabled", default=False)):
                policy = getattr(approach, "_policy", None)
                env_factory = getattr(approach, "env_factory", None)
                env_specs = getattr(approach, "env_specs", None)
                approach_base_class_name = getattr(approach, "base_class_name", "")
                if (
                    policy is not None
                    and env_factory is not None
                    and env_specs is not None
                ):
                    print("VECTOR FIELD FOR TRAIN ENVS:")
                    plot_policy_vector_fields(
                        base_class_name=cfg.env.make_kwargs.base_name,
                        approach_base_class_name=str(approach_base_class_name),
                        policy=policy,
                        env_factory=env_factory,
                        env_specs=env_specs,
                        env_nums=range(0, 10),
                        grid_size=int(
                            OmegaConf.select(
                                cfg, "eval.vector_field.grid_size", default=21
                            )
                        ),
                        split_name="train",
                    )

            test_accuracies = approach.test_policy_on_envs(
                base_class_name=cfg.env.make_kwargs.base_name,
                test_env_nums=range(10, 20),
                max_num_steps=1000,
                record_videos=bool(
                    OmegaConf.select(cfg, "eval.record_videos", default=False)
                ),
                video_format=str(
                    OmegaConf.select(cfg, "eval.video_format", default="mp4")
                ),
            )
            logging.info(test_accuracies)
            # logging.info(df["total_rewards"].iloc[0])
            logging.info(sum(test_accuracies) / len(test_accuracies))
            if bool(OmegaConf.select(cfg, "eval.vector_field.enabled", default=False)):
                policy = getattr(approach, "_policy", None)
                env_factory = getattr(approach, "env_factory", None)
                env_specs = getattr(approach, "env_specs", None)
                approach_base_class_name = getattr(approach, "base_class_name", "")
                if (
                    policy is not None
                    and env_factory is not None
                    and env_specs is not None
                ):
                    plot_policy_vector_fields(
                        base_class_name=cfg.env.make_kwargs.base_name,
                        approach_base_class_name=str(approach_base_class_name),
                        policy=policy,
                        env_factory=env_factory,
                        env_specs=env_specs,
                        env_nums=range(10, 20),
                        grid_size=int(
                            OmegaConf.select(
                                cfg, "eval.vector_field.grid_size", default=21
                            )
                        ),
                        split_name="test",
                    )
        else:
            logging.warning(
                f"Approach {cfg.approach_name} does not support `test_policy_on_envs`."
            )


def _run_single_episode_evaluation(
    approach: BaseApproach,
    env: Env,
    rng: np.random.Generator,
    max_eval_steps: int,
) -> dict[str, float]:
    # For now, just record total rewards and steps.
    total_rewards = 0.0
    total_steps = 0
    seed = sample_seed_from_rng(rng)
    obs, info = env.reset(seed=seed)

    approach.reset(obs, info)

    for _ in range(max_eval_steps):
        action = approach.step()
        obs, rew, done, truncated, info = env.step(action)
        reward = float(rew)
        env.render()

        assert not truncated
        approach.update(obs, reward, done, info)
        total_rewards += reward
        if done:
            print(
                "Episode finished after "
                f"{total_steps + 1} steps with max step {max_eval_steps:.4f}"
            )
            break
        total_steps += 1
    return {"total_rewards": total_rewards, "total_steps": total_steps}


def _maybe_generate_plp_visualization(
    cfg: DictConfig,
    approach: BaseApproach,
    registry: EnvRegistry,
) -> None:
    """Generate an explained post-training rollout when requested."""
    if not bool(OmegaConf.select(cfg, "eval.plp_visualization.enabled", default=False)):
        return
    if OmegaConf.select(cfg, "env.provider", default=None) != "arc_agi3":
        logging.warning("PLP decision visualization currently supports ARC-AGI-3.")
        return

    policy = getattr(approach, "_policy", None)
    if policy is None:
        logging.warning("No learned policy is available for visualization.")
        return

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_name = str(
        OmegaConf.select(
            cfg,
            "eval.plp_visualization.output_name",
            default="plp_decision_trace.html",
        )
    )
    feature_json_path = OmegaConf.select(
        cfg,
        "approach.program_generation.loading.offline_json_path",
        default=None,
    )
    configured_reset_seeds = OmegaConf.select(
        cfg, "eval.plp_visualization.reset_seeds", default=None
    )
    if configured_reset_seeds is None:
        reset_seeds = [
            int(OmegaConf.select(cfg, "eval.plp_visualization.reset_seed", default=0))
        ]
    else:
        reset_seeds = [int(seed) for seed in configured_reset_seeds]

    max_steps = int(
        OmegaConf.select(cfg, "eval.plp_visualization.max_steps", default=30)
    )
    output_path = output_dir / output_name
    for reset_seed in reset_seeds:
        trace_env = registry.load(cfg.env, instance_num=reset_seed)
        if len(reset_seeds) == 1:
            seed_output_path = output_path
        else:
            seed_output_path = output_path.with_name(
                f"{output_path.stem}_seed{reset_seed:04d}{output_path.suffix}"
            )
        report_path = generate_arc_lpp_decision_trace(
            env=trace_env,
            policy=policy,
            output_path=seed_output_path,
            max_steps=max_steps,
            reset_seed=reset_seed,
            feature_json_path=feature_json_path,
        )
        close = getattr(trace_env, "close", None)
        if callable(close):
            close()
        logging.info(
            "Saved PLP decision visualization for seed %d to %s",
            reset_seed,
            report_path,
        )


if __name__ == "__main__":
    try:
        _main()  # pylint: disable=no-value-for-parameter
    except BaseException as e:  # pylint: disable=broad-exception-caught
        logging.exception(
            "Unhandled exception in run_experiment (%s): %s",
            type(e).__name__,
            e,
        )
        raise
