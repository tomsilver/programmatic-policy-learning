"""Bilevel-planning experts for Motion2D environments."""

from __future__ import annotations

from typing import Any


def _load_bilevel_components() -> tuple[Any, Any]:
    """Import bilevel-planning modules lazily with a helpful error."""
    try:
        from kinder_bilevel_planning.agent import BilevelPlanningAgent
        from kinder_bilevel_planning.env_models import create_bilevel_planning_models
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "Motion2D bilevel expert requires `kinder_bilevel_planning` and its "
            "dependencies to be installed."
        ) from exc
    return BilevelPlanningAgent, create_bilevel_planning_models


class Motion2DBilevelPlanningExpert:
    """Stateful Motion2D expert backed by KinDER bilevel planning."""

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        seed: int = 0,
        *,
        num_passages: int = 0,
        max_abstract_plans: int = 10,
        samples_per_step: int = 3,
        max_skill_horizon: int = 100,
        heuristic_name: str = "hff",
        planning_timeout: float = 60.0,
    ) -> None:
        (
            bilevel_agent_cls,
            create_bilevel_planning_models,
        ) = _load_bilevel_components()
        env_models = create_bilevel_planning_models(
            "motion2d",
            observation_space,
            action_space,
            num_passages=int(num_passages),
        )
        self._agent = bilevel_agent_cls(
            env_models,
            seed,
            max_abstract_plans=int(max_abstract_plans),
            samples_per_step=int(samples_per_step),
            max_skill_horizon=int(max_skill_horizon),
            heuristic_name=str(heuristic_name),
            planning_timeout=float(planning_timeout),
        )

    def reset(self, obs: Any, info: dict[str, Any]) -> None:
        """Plan from the current initial state."""
        self._agent.reset(obs, info)

    def step(self) -> Any:
        """Return the next action in the current plan."""
        return self._agent.step()

    def update(self, obs: Any, reward: float, done: bool, info: dict[str, Any]) -> None:
        """Update the wrapped planning agent with the latest transition."""
        self._agent.update(obs, reward, done, info)


def create_motion2d_bilevel_expert(
    observation_space: Any,
    action_space: Any,
    seed: int = 0,
    *,
    num_passages: int = 0,
    max_abstract_plans: int = 10,
    samples_per_step: int = 3,
    max_skill_horizon: int = 100,
    heuristic_name: str = "hff",
    planning_timeout: float = 60.0,
) -> Motion2DBilevelPlanningExpert:
    """Factory that builds a Motion2D bilevel-planning expert."""
    return Motion2DBilevelPlanningExpert(
        observation_space,
        action_space,
        seed=seed,
        num_passages=num_passages,
        max_abstract_plans=max_abstract_plans,
        samples_per_step=samples_per_step,
        max_skill_horizon=max_skill_horizon,
        heuristic_name=heuristic_name,
        planning_timeout=planning_timeout,
    )
