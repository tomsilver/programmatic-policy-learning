"""Script for running experiments with hydra."""

import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from gymnasium.core import Env
from omegaconf import DictConfig, OmegaConf
from prpl_utils.utils import sample_seed_from_rng

from programmatic_policy_learning.approaches.base_approach import BaseApproach

# from programmatic_policy_learning.dsl.llm_primitives.baseline_vlam import run_baseline
from programmatic_policy_learning.envs.registry import EnvRegistry


def instantiate_approach(
    cfg: DictConfig, env: Any, registry: EnvRegistry
) -> BaseApproach:
    """Instantiate the approach based on the configuration.

    Handles specific parameters required for certain approaches like `lpp`.
    """

    env_factory = lambda instance_num: registry.load(cfg.env, instance_num=instance_num)

    if cfg.approach_name == "lpp":

        if not hasattr(env, "get_object_types"):
            raise AttributeError(
                f"Environment {cfg.env_name} does not support `get_object_types`."
            )

        object_types = env.get_object_types()
        env_specs = {"object_types": object_types}

        expert = hydra.utils.instantiate(
            cfg.expert,
            cfg.env.description,
            env.observation_space,
            env.action_space,
            cfg.seed,
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

    # Handle residual learning.
    if cfg.approach_name == "residual":

        expert = hydra.utils.instantiate(
            cfg.expert,
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
            test_env_nums=range(11, 20),
            max_num_steps=50,
            record_videos=False,
            video_format="mp4",
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

    if cfg.eval.mode == 1:
        evaluate_all(cfg)
    else:
        logging.info(
            f"Running seed={cfg.seed}, env={cfg.env_name}, approach={cfg.approach_name}"
        )
        registry = EnvRegistry()

        env = registry.load(cfg.env)

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

        # Test the approach on new envs
        if hasattr(approach, "test_policy_on_envs"):
            test_accuracies = approach.test_policy_on_envs(
                base_class_name=cfg.env.make_kwargs.base_name,
                test_env_nums=range(11, 20),
                max_num_steps=50,
                record_videos=False,
                video_format="mp4",
            )
            logging.info(test_accuracies)
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
    obs, info = env.reset(seed=sample_seed_from_rng(rng))
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
            break
        total_steps += 1
    return {"total_rewards": total_rewards, "total_steps": total_steps}


if __name__ == "__main__":
    try:
        _main()  # pylint: disable=no-value-for-parameter
    except BaseException as e:  # pylint: disable=broad-exception-caught
        logging.error(str(e))
