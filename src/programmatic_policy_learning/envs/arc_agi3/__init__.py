"""ARC-AGI-3 environment adapter."""

from programmatic_policy_learning.envs.arc_agi3.preprocessing import (
    extract_latest_grid,
    extract_objects_by_color,
    preprocess_arc_observation,
    register_arc_object_enricher,
)
from programmatic_policy_learning.envs.arc_agi3.wrapper import (
    ArcAgi3Env,
    ArcAgi3GymEnv,
    coerce_action,
    create_arcade,
    list_environments,
)
from programmatic_policy_learning.envs.arc_agi3.variants import (
    Ls20InitialStateVariant,
)

__all__ = [
    "ArcAgi3Env",
    "ArcAgi3GymEnv",
    "Ls20InitialStateVariant",
    "coerce_action",
    "create_arcade",
    "extract_latest_grid",
    "extract_objects_by_color",
    "list_environments",
    "preprocess_arc_observation",
    "register_arc_object_enricher",
]
