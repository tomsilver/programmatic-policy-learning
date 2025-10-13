"""GGG environment provider."""

from typing import Any

from omegaconf import DictConfig
from prpl_utils.gym_utils import GymToGymnasium


class GGGEnvWithTypes:
    # When adding new environments, ensure this wrapper class provides access
    # to key methods (e.g., get_object_types) needed by the Env object.
    """Wrapper for GGG environments that provides object type extraction."""

    def __init__(self, base_env: GymToGymnasium, base_class_name: str) -> None:
        self.env = base_env
        self.base_class_name = base_class_name

    def get_object_types(self) -> tuple[str, ...]:
        """Return object types for the each env."""
        if self.base_class_name.startswith("TwoPileNimGymEnv"):
            return ("tpn.EMPTY", "tpn.TOKEN", "None")
        if self.base_class_name.startswith("CheckmateTacticGymEnv"):
            return (
                "ct.EMPTY",
                "ct.HIGHLIGHTED_WHITE_QUEEN",
                "ct.BLACK_KING",
                "ct.HIGHLIGHTED_WHITE_KING",
                "ct.WHITE_KING",
                "ct.WHITE_QUEEN",
                "None",
            )
        if self.base_class_name.startswith("StopTheFallGymEnv"):
            return (
                "stf.EMPTY",
                "stf.FALLING",
                "stf.RED",
                "stf.STATIC",
                "stf.ADVANCE",
                "stf.DRAWN",
                "None",
            )
        if self.base_class_name.startswith("ChaseGymEnv"):
            return (
                "ec.EMPTY",
                "ec.TARGET",
                "ec.AGENT",
                "ec.WALL",
                "ec.DRAWN",
                "ec.LEFT_ARROW",
                "ec.RIGHT_ARROW",
                "ec.UP_ARROW",
                "ec.DOWN_ARROW",
                "None",
            )
        if self.base_class_name.startswith("ReachForTheStarGymEnv"):
            return (
                "rfts.EMPTY",
                "rfts.AGENT",
                "rfts.STAR",
                "rfts.DRAWN",
                "rfts.LEFT_ARROW",
                "rfts.RIGHT_ARROW",
                "None",
            )

        raise ValueError(f"Unknown class name: {self.base_class_name}")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


def get_true_env_class_name(env: Any) -> str:
    """Recursively unwrap gym environment to get the true class name."""
    while hasattr(env, "env"):
        env = env.env
    if hasattr(env, "unwrapped"):
        env = env.unwrapped
    return env.__class__.__name__


def create_ggg_env(env_config: DictConfig) -> GGGEnvWithTypes:
    """Create GGG environment with legacy gym compatibility."""
    # Lazy import, to avoid deprecation warnings at module import time
    import generalization_grid_games  # pylint: disable=unused-import,import-outside-toplevel
    import gym as legacy_gym  # pylint: disable=import-outside-toplevel

    instance_num = env_config.get("instance_num", 0)
    env_id = f"TwoPileNim{instance_num}-v0"
    env_config.make_kwargs.id = env_id
    # env_id = env_config.make_kwargs.id
    base = legacy_gym.make(env_id, disable_env_checker=True)
    env_name = base.unwrapped.__class__.__name__
    wrapped = GymToGymnasium(base)
    return GGGEnvWithTypes(wrapped, env_name)
