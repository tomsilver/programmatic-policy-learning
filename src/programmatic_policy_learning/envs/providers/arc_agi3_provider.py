"""ARC-AGI-3 environment provider."""

from typing import Any

from omegaconf import DictConfig

from programmatic_policy_learning.envs.arc_agi3 import ArcAgi3Env, ArcAgi3GymEnv


def create_arc_agi3_env(
    env_config: DictConfig, instance_num: int | None = None
) -> ArcAgi3Env | ArcAgi3GymEnv:
    """Create an ARC-AGI-3 environment adapter from a Hydra config."""
    kwargs: dict[str, Any] = dict(env_config.get("make_kwargs", {}))
    kwargs.pop("base_name", None)
    randomize_initial_state = bool(kwargs.pop("randomize_initial_state", False))
    randomize_shape = bool(kwargs.pop("randomize_shape", False))
    game_id = kwargs.pop("game_id", kwargs.pop("id", "ls20"))
    wrapper = kwargs.pop("wrapper", "raw")
    if wrapper == "gym":
        env = ArcAgi3GymEnv(game_id=game_id, **kwargs)
        _maybe_apply_initial_state_variant(
            env,
            instance_num=instance_num,
            randomize_initial_state=randomize_initial_state,
            randomize_shape=randomize_shape,
        )
        return env
    if wrapper != "raw":
        raise ValueError(f"Unknown ARC-AGI-3 wrapper: {wrapper!r}")
    env = ArcAgi3Env(game_id=game_id, **kwargs)
    _maybe_apply_initial_state_variant(
        env,
        instance_num=instance_num,
        randomize_initial_state=randomize_initial_state,
        randomize_shape=randomize_shape,
    )
    return env


def _maybe_apply_initial_state_variant(
    env: ArcAgi3Env | ArcAgi3GymEnv,
    *,
    instance_num: int | None,
    randomize_initial_state: bool,
    randomize_shape: bool,
) -> None:
    """Map paper-curve instance numbers to deterministic ARC variants."""
    if instance_num is None or not randomize_initial_state:
        return
    variant = env.sample_initial_state_variant(
        int(instance_num), randomize_shape=randomize_shape
    )
    raw_env = env.raw_env if isinstance(env, ArcAgi3GymEnv) else env
    raw_env.initial_state_variant = variant
