"""Generic bilevel-planning experts for KinDER environments."""

from __future__ import annotations

from typing import Any

from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based.continuous_hint_config import (
    canonicalize_env_name,
)

_DEFAULT_BILEVEL_MODEL_NAMES: dict[str, str] = {
    "Motion2D": "motion2d",
    "PushPullHook2D": "pushpullhook2d",
}


def _load_bilevel_components() -> tuple[Any, Any]:
    """Import bilevel-planning modules lazily with a helpful error."""
    try:
        # pylint: disable=import-outside-toplevel
        from kinder_bilevel_planning.agent import BilevelPlanningAgent
        from kinder_bilevel_planning.env_models import create_bilevel_planning_models
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "KinDER bilevel expert requires `kinder_bilevel_planning` and its "
            "dependencies to be installed."
        ) from exc
    return BilevelPlanningAgent, create_bilevel_planning_models


def resolve_bilevel_model_name(
    *, env_name: str | None = None, env_model_name: str | None = None
) -> str:
    """Resolve the module name used by `kinder_bilevel_planning` env models."""
    if env_model_name:
        return str(env_model_name)
    if not env_name:
        raise ValueError("Either env_name or env_model_name must be provided.")
    canonical_name = canonicalize_env_name(env_name)
    try:
        return _DEFAULT_BILEVEL_MODEL_NAMES[canonical_name]
    except KeyError as exc:
        raise KeyError(
            f"No bilevel model-name mapping configured for KinDER env "
            f"'{canonical_name}'."
        ) from exc


class KinderBilevelPlanningExpert:
    """Stateful KinDER expert backed by bilevel planning."""

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        seed: int = 0,
        *,
        env_name: str | None = None,
        env_model_name: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        max_abstract_plans: int = 10,
        samples_per_step: int = 3,
        max_skill_horizon: int = 100,
        heuristic_name: str = "hff",
        planning_timeout: float = 60.0,
    ) -> None:
        bilevel_agent_cls, create_bilevel_planning_models = _load_bilevel_components()
        resolved_model_name = resolve_bilevel_model_name(
            env_name=env_name,
            env_model_name=env_model_name,
        )
        env_model_kwargs = dict(model_kwargs or {})
        try:
            env_models = create_bilevel_planning_models(
                resolved_model_name,
                observation_space,
                action_space,
                **env_model_kwargs,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"No bilevel-planning env model is available for KinDER env "
                f"model '{resolved_model_name}'."
            ) from exc
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


def create_kinder_bilevel_expert(
    observation_space: Any,
    action_space: Any,
    seed: int = 0,
    *,
    env_name: str | None = None,
    env_model_name: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    max_abstract_plans: int = 10,
    samples_per_step: int = 3,
    max_skill_horizon: int = 100,
    heuristic_name: str = "hff",
    planning_timeout: float = 60.0,
) -> KinderBilevelPlanningExpert:
    """Factory that builds a generic KinDER bilevel-planning expert."""
    return KinderBilevelPlanningExpert(
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
