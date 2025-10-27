"""Script for running experiments with hydra."""

import logging

import hydra
import numpy as np
import pandas as pd
from gymnasium.core import Env
from omegaconf import DictConfig
from prpl_utils.utils import sample_seed_from_rng

from programmatic_policy_learning.approaches.base_approach import BaseApproach
from programmatic_policy_learning.envs.registry import EnvRegistry


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:

    # logging.info(
    #     f"Running seed={cfg.seed}, env={cfg.env_name}, approach={cfg.approach_name}"
    # )

    # registry = EnvRegistry()
    # env = registry.load(cfg.env)
    # env_factory = lambda instance_num: registry.load(cfg.env, instance_num=instance_num)

    # object_types = env.get_object_types()
    # env_specs = {"object_types": object_types}

    # expert = hydra.utils.instantiate(
    #     cfg.expert,
    #     cfg.env.description,
    #     env.observation_space,
    #     env.action_space,
    #     cfg.seed,
    # )
    # # Create the approach.
    # approach = hydra.utils.instantiate(
    #     cfg.approach,
    #     cfg.env.description,
    #     env.observation_space,
    #     env.action_space,
    #     cfg.seed,
    #     expert,
    #     env_factory,
    #     cfg.env.make_kwargs.base_name,
    #     env_specs=env_specs,
    # )
    logging.info(
    f"Running seed={cfg.seed}, env={cfg.env_name}, approach={cfg.approach_name}"
    )

    registry = EnvRegistry()
    env = registry.load(cfg.env)
    # Create the approach.
    approach = hydra.utils.instantiate(
        cfg.approach,
        cfg.env.description,
        env.observation_space,
        env.action_space,
        cfg.seed,
        get_actions=env.get_actions,
        get_next_state=env.get_next_state,
        get_cost=env.get_cost,
        check_goal=env.check_goal,
    )

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
    test_accuracies = approach.test_policy_on_envs(
        base_class_name=cfg.env.make_kwargs.base_name,
        test_env_nums=range(11, 20),
        max_num_steps=50,
        record_videos=False,
        video_format="mp4",
    )
    logging.info(test_accuracies)


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
    _main()  # pylint: disable=no-value-for-parameter
