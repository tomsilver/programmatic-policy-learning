"""Gym-Maze environment provider."""

from omegaconf import DictConfig
from prpl_utils.gym_utils import GymToGymnasium


def create_maze_env(env_config: DictConfig) -> GymToGymnasium:
    """Create Gym-Maze environment with legacy gym compatibility."""
    # Lazy import, to avoid deprecation warnings at module import time
    import gym as legacy_gym  # pylint: disable=unused-import,import-outside-toplevel
    import gym_maze  # type: ignore[import-untyped]  # pylint: disable=unused-import,import-outside-toplevel

    env_id = env_config.make_kwargs.id
    base = legacy_gym.make(env_id, disable_env_checker=True)

    while hasattr(base, "env"):
        base = base.env

    return GymToGymnasium(base)
