"""KinDER environment provider."""

from typing import Any, cast

import kinder  # type: ignore[import-not-found]
from gymnasium import spaces
from omegaconf import DictConfig


class KinderEnvWithTypes:
    """Wrapper for KinDER environments that provides object type extraction.

    Similar to GGGEnvWithTypes, this wrapper adds get_object_types()
    needed by the LPP pipeline.
    """

    def __init__(
        self,
        env: Any,
        type_names: tuple[str, ...],
        action_types: tuple[str, ...],
    ) -> None:
        self._env = env
        self._type_names = type_names
        self._action_types = action_types
        self._last_reset_seed: int | None = None

    def get_object_types(self) -> tuple[str, ...]:
        """Return object type names for this environment."""
        return self._type_names

    def get_action_types(self) -> tuple[str, ...]:
        """Return action type names for this environment."""
        return self._action_types

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
        seed = kwargs.get("seed")
        self._last_reset_seed = int(seed) if isinstance(seed, int) else None
        return self._env.reset(**kwargs)

    @property
    def last_reset_seed(self) -> int | None:
        """Return the most recent reset seed seen by this wrapper."""
        return self._last_reset_seed

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


def _extract_action_types(env: Any) -> tuple[str, ...]:
    """Extract per-dimension action types from a KinDER environment.

    Prefer environment-provided metadata when available. Otherwise fall
    back to a generic inference from the action space, with a KinDER-specific
    hint for the final vacuum/toggle dimension in 5-D continuous robot actions.
    """
    for attr_name in ("action_types", "action_type_names"):
        raw = getattr(env, attr_name, None)
        if raw is not None:
            try:
                values = tuple(str(x) for x in raw)
            except TypeError:
                values = (str(raw),)
            if values:
                return values

    action_space = env.action_space
    if isinstance(action_space, spaces.Discrete):
        return ("discrete",)

    if isinstance(action_space, spaces.MultiBinary):
        raw_n: object = action_space.n
        size = (
            int(raw_n) if isinstance(raw_n, int) else len(cast(tuple[int, ...], raw_n))
        )
        return tuple("boolean" for _ in range(size))

    if isinstance(action_space, spaces.MultiDiscrete):
        return tuple("discrete" for _ in action_space.nvec.tolist())

    if isinstance(action_space, spaces.Box):
        shape = tuple(int(dim) for dim in action_space.shape)
        size = shape[0] if len(shape) == 1 else 0
        inferred = ["continuous"] * size
        if size >= 5:
            inferred[4] = "boolean-like toggle"
        return tuple(inferred)

    return tuple()


def create_kinder_env(
    env_config: DictConfig, instance_num: int | None = None
) -> KinderEnvWithTypes:
    """Create KinDER environment with object type extraction."""
    kinder.register_all_environments()

    kwargs = dict(env_config["make_kwargs"])
    # Remove metadata keys that are not kinder constructor arguments
    kwargs.pop("base_name", None)
    # Some configs store null as the string "null"; normalize to Python None.
    if kwargs.get("render_mode") == "null":
        kwargs["render_mode"] = None

    # Use instance_num to create different env instances if needed
    if instance_num is not None:
        # For kinder envs, instance_num maps to seed for reset
        env = kinder.make(**kwargs)
    else:
        env = kinder.make(**kwargs)

    type_names = _extract_type_names(env)
    action_types = _extract_action_types(env)
    return KinderEnvWithTypes(env, type_names, action_types)
