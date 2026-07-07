"""Virtual Tools environment provider."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig


class VirtualToolsEnv(gym.Env):
    """One-shot Gymnasium wrapper around Virtual Tools ToolPicker."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        level_path: Path,
        grid_step: int = 20,
        maxtime: float = 20.0,
    ) -> None:
        super().__init__()

        # Import lazily so the rest of LPP can import without requiring pyGameWorld.
        from pyGameWorld import ToolPicker  # type: ignore[import-not-found]

        self.level_path = level_path
        self.grid_step = grid_step
        self.maxtime = maxtime

        with open(level_path, "r", encoding="utf-8") as f:
            self.level_dict = json.load(f)

        self.tp = ToolPicker(self.level_dict)

        self.tool_names = tuple(sorted(self.level_dict["tools"].keys()))
        self.positions = [
            (x, y)
            for x in range(0, 600, grid_step)
            for y in range(0, 600, grid_step)
        ]
        self._actions = [
            (tool_name, pos)
            for tool_name in self.tool_names
            for pos in self.positions
        ]

        self.action_space = spaces.Discrete(len(self._actions))

        # Minimal obs for now: an action-independent structured state.
        # If the LPP pipeline dislikes arbitrary dict obs, replace this with
        # a flattened Box later and keep the full state in info["state"].
        self.observation_space = spaces.Dict(
            {
                "dummy": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                )
            }
        )

        self._done = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self._done = False

        # Rebuild ToolPicker each episode so step() starts from clean physics.
        from pyGameWorld import ToolPicker  # type: ignore[import-not-found]
        self.tp = ToolPicker(self.level_dict)

        obs = {"dummy": np.array([0.0], dtype=np.float32)}
        info = {"state": self.get_state()}
        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action)

        tool_name, position = self.decode_action(action)

        path_dict, success, time_to_success = self.tp.observePlacementPath(
            toolname=tool_name,
            position=position,
            maxtime=self.maxtime,
        )

        reward = 1.0 if success else 0.0
        terminated = True
        truncated = False
        self._done = True

        obs = {"dummy": np.array([0.0], dtype=np.float32)}
        info = {
            "state": self.get_state(),
            "decoded_action": {
                "tool_name": tool_name,
                "position": position,
            },
            "success": bool(success),
            "time_to_success": time_to_success,
            "path_dict": path_dict,
        }
        return obs, reward, terminated, truncated, info

    def get_state(self) -> dict[str, Any]:
        return {
            "level_path": str(self.level_path),
            "world": self.level_dict["world"],
            "tools": self.level_dict["tools"],
            "tool_names": self.tool_names,
            "grid_step": self.grid_step,
        }

    def get_actions(self) -> list[int]:
        return list(range(len(self._actions)))

    def decode_action(self, action: int) -> tuple[str, tuple[int, int]]:
        tool_name, position = self._actions[action]
        return tool_name, position

    def get_object_types(self) -> tuple[str, ...]:
        return (
            "vt.BALL",
            "vt.GOAL",
            "vt.TABLE",
            "vt.STATIC_OBJECT",
            "vt.DYNAMIC_OBJECT",
            "vt.TOOL",
            "None",
        )

    def get_action_types(self) -> tuple[str, ...]:
        return ("vt.PLACE_TOOL",)


def create_virtual_tools_env(env_config: DictConfig) -> VirtualToolsEnv:
    level_dir = Path(env_config.make_kwargs.level_dir)
    level_name = env_config.make_kwargs.level_name
    grid_step = int(env_config.make_kwargs.get("grid_step", 20))
    maxtime = float(env_config.make_kwargs.get("maxtime", 20.0))

    # First pass: assume level_name maps to a JSON file.
    # Codex should inspect the actual Trials folder and adjust this path.
    level_path = level_dir / f"{level_name}.json"

    return VirtualToolsEnv(
        level_path=level_path,
        grid_step=grid_step,
        maxtime=maxtime,
    )