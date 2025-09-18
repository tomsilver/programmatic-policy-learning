"""PRBench environment provider."""

import prbench
from gymnasium import spaces

from programmatic_policy_learning.envs.utils.wrappers import patch_box_float32


def create_prbench_env(env_config: object) -> object:
    """Create PRBench environment with float32 Box spaces."""
    # Register environments with prbench
    prbench.register_all_environments()
    kwargs = env_config.make_kwargs
    # Patch Box creation to use float32 by default and avoid warnings
    original_box_init = patch_box_float32()
    try:
        # Create environment with patched Box class
        env = prbench.make(**kwargs)
    finally:
        # Restore original Box.__init__
        spaces.Box.__init__ = original_box_init

    return env
