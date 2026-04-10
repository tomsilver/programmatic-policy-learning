"""Generic bilevel-planning expert approach for KinDER environments."""

from __future__ import annotations

from typing import Any

from programmatic_policy_learning.approaches.base_approach import BaseApproach
from programmatic_policy_learning.approaches.experts.kinder_bilevel_experts import (
    KinderBilevelPlanningExpert,
    create_kinder_bilevel_expert,
)


class KinderBilevelExpertApproach(BaseApproach[Any, Any]):
    """KinDER expert approach backed by a bilevel planner."""

    def __init__(
        self,
        environment_description: str,
        observation_space: Any,
        action_space: Any,
        seed: int,
        env_name: str | None = None,
        env_model_name: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        max_abstract_plans: int = 10,
        samples_per_step: int = 3,
        max_skill_horizon: int = 100,
        heuristic_name: str = "hff",
        planning_timeout: float = 60.0,
    ) -> None:
        super().__init__(environment_description, observation_space, action_space, seed)
        self._last_observation: Any | None = None
        self._last_info: dict[str, Any] | None = None
        self._expert: KinderBilevelPlanningExpert = create_kinder_bilevel_expert(
            observation_space,
            action_space,
            seed=seed,
            env_name=env_name,
            env_model_name=env_model_name,
            model_kwargs=model_kwargs,
            max_abstract_plans=max_abstract_plans,
            samples_per_step=samples_per_step,
            max_skill_horizon=max_skill_horizon,
            heuristic_name=heuristic_name,
            planning_timeout=planning_timeout,
        )

    def reset(self, obs: Any, info: dict[str, Any]) -> None:
        self._last_observation = obs
        self._last_info = info
        self._expert.reset(obs, info)

    def _get_action(self) -> Any:
        return self._expert.step()

    def update(self, obs: Any, reward: float, done: bool, info: dict[str, Any]) -> None:
        self._last_observation = obs
        self._last_info = info
        self._expert.update(obs, reward, done, info)
