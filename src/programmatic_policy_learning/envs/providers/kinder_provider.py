"""KinDER environment provider."""

from typing import Any

import kinder  # type: ignore[import-not-found]
from gymnasium import spaces
from omegaconf import DictConfig


class KinderEnvWithTypes:
    """Wrapper for KinDER environments that provides object type extraction.

    Similar to GGGEnvWithTypes, this wrapper adds get_object_types()
    needed by the LPP pipeline.
    """

    def __init__(self, env: Any, type_names: tuple[str, ...]) -> None:
        self._env = env
        self._type_names = type_names

    def get_object_types(self) -> tuple[str, ...]:
        """Return object type names for this environment."""
        return self._type_names

    @property
    def observation_space(self) -> spaces.Space:
        """Return the observation space."""
        return self._env.observation_space

    @property
    def action_space(self) -> spaces.Space:
        """Return the action space."""
        return self._env.action_space

    def reset(self, **kwargs: Any) -> tuple[Any, dict]:
        """Reset the environment."""
        return self._env.reset(**kwargs)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Step the environment."""
        return self._env.step(action)

    def render(self) -> Any:
        """Render the environment."""
        return self._env.render()

    def close(self) -> None:
        """Close the environment."""
        self._env.close()


def _extract_type_names(env: Any) -> tuple[str, ...]:
    """Extract object type names from a KinDER environment's observation space.

    KinDER environments use ObjectCentricBoxSpace which has
    type_features mapping Type -> list[str].
    """
    obs_space = env.observation_space
    # ObjectCentricBoxSpace has type_features: dict[Type, list[str]]
    type_features = obs_space.type_features
    return tuple(t.name for t in type_features)


def create_kinder_env(
    env_config: DictConfig, instance_num: int | None = None
) -> KinderEnvWithTypes:
    """Create KinDER environment with object type extraction."""
    kinder.register_all_environments()

    kwargs = dict(env_config["make_kwargs"])
    # Remove metadata keys that are not kinder constructor arguments
    kwargs.pop("base_name", None)

    # Use instance_num to create different env instances if needed
    if instance_num is not None:
        # For kinder envs, instance_num maps to seed for reset
        env = kinder.make(**kwargs)
    else:
        env = kinder.make(**kwargs)

    type_names = _extract_type_names(env)
    return KinderEnvWithTypes(env, type_names)
