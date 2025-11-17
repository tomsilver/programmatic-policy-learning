"""Script for running experiments with hydra."""

import logging
from typing import Any

import hydra
import numpy as np
import pandas as pd
from gymnasium.core import Env
from omegaconf import DictConfig
from prpl_utils.utils import sample_seed_from_rng
from prpl_llm_utils.code import synthesize_python_function_with_llm, SyntaxRepromptCheck, FunctionOutputRepromptCheck
from prpl_llm_utils.structs import Query

from programmatic_policy_learning.approaches.base_approach import BaseApproach
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

        use_llm_expert = getattr(cfg, "use_llm_expert", False)

        if use_llm_expert:
            function_name = cfg.llm_expert.function_name  # e.g., "expert_policy"
            prompt_template = cfg.llm_expert.prompt_template

            # Instantiate the LLM (OpenAIModel via Hydra)
            llm = hydra.utils.instantiate(cfg.llm)

            action_space = env.action_space
            example_observation, _ = env.reset(seed=cfg.seed)
            example_observation = np.asarray(example_observation, dtype=np.float32)

            prompt_text = (
                prompt_template
                + "\n\nEnvironment description:\n"
                + str(cfg.env.description)
                + "\nObservation space:\n"
                + str(env.observation_space)
                + "\nAction space:\n"
                + str(env.action_space)
            )

            # Wrap the prompt string in a Query object (IMPORTANT)
            query = Query(prompt_text)

            reprompt_checks = [
                SyntaxRepromptCheck(),
                FunctionOutputRepromptCheck(
                    function_name,
                    [(example_observation,)],
                    [action_space.contains],
                ),
            ]

            # Call the prpl_llm_utils version, just like your friend
            expert_fn = synthesize_python_function_with_llm(
                function_name,
                llm,
                query,
                reprompt_checks=reprompt_checks,
            )

            # expert_fn is the callable policy function
            expert = expert_fn
            
        else:
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


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:

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
