"""Smoke test for the Virtual Tools environment provider."""

from __future__ import annotations

from pathlib import Path

import pyGameWorld
from omegaconf import OmegaConf

from programmatic_policy_learning.envs.registry import EnvRegistry


def _get_virtual_tools_level_dir() -> Path:
    """Return the Virtual Tools Original trials directory."""

    # If pyGameWorld is installed editable from tool-games/environment,
    # pyGameWorld.__file__ is:
    #   .../tool-games/environment/pyGameWorld/__init__.py
    env_root = Path(pyGameWorld.__file__).resolve().parents[1]
    candidate = env_root / "Trials" / "Original"

    if candidate.exists():
        return candidate

    # Fallbacks for common local clone layouts.
    candidates = [
        Path("../tool-games/environment/Trials/Original"),
        Path("tool-games/environment/Trials/Original"),
        Path("environment/Trials/Original"),
    ]

    for path in candidates:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(
        "Could not find Virtual Tools levels. Expected something like "
        "`tool-games/environment/Trials/Original`."
    )


def test_virtual_tools_env_reset_and_step() -> None:
    """The Virtual Tools env should load through EnvRegistry and step once."""

    level_dir = _get_virtual_tools_level_dir()

    cfg = OmegaConf.create(
        {
            "provider": "virtual_tools",
            "make_kwargs": {
                "level_dir": str(level_dir),
                "level_name": "Basic",
                "grid_step": 100,
                "maxtime": 2.0,
            },
        }
    )

    env = EnvRegistry().load(cfg)

    obs, info = env.reset(seed=0)

    assert obs is not None
    assert isinstance(info, dict)
    assert "state" in info

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    assert obs is not None
    assert reward in {0.0, 1.0}
    assert terminated is True
    assert truncated is False

    assert isinstance(info, dict)
    assert "decoded_action" in info
    assert "success" in info
    assert "time_to_success" in info

    decoded_action = info["decoded_action"]
    assert "tool_name" in decoded_action
    assert "position" in decoded_action


def test_virtual_tools_decode_action() -> None:
    """The env should expose finite discrete actions that decode to placements."""

    level_dir = _get_virtual_tools_level_dir()

    cfg = OmegaConf.create(
        {
            "provider": "virtual_tools",
            "make_kwargs": {
                "level_dir": str(level_dir),
                "level_name": "Basic",
                "grid_step": 100,
                "maxtime": 2.0,
            },
        }
    )

    env = EnvRegistry().load(cfg)

    actions = env.get_actions()

    assert len(actions) > 0

    tool_name, position = env.decode_action(actions[0])

    assert isinstance(tool_name, str)
    assert isinstance(position, tuple)
    assert len(position) == 2