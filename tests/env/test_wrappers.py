"""Tests for wrappers."""

import numpy as np
from gymnasium import spaces

from programmatic_policy_learning.envs.utils.wrappers import patch_box_float32


def test_patch_box_float32() -> None:
    """Test patch_box_float32 changes Box dtype."""
    original_box_init = patch_box_float32()
    box = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
    assert box.dtype == np.float32
    spaces.Box.__init__ = original_box_init  # type: ignore
