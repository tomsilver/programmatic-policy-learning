"""Thin adapter around the official ARC-AGI-3 SDK."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import arc_agi
import numpy as np
from arc_agi import OperationMode
from arcengine import GameAction
from gymnasium.spaces import Space

from programmatic_policy_learning.envs.arc_agi3.preprocessing import (
    preprocess_arc_observation,
)
from programmatic_policy_learning.envs.arc_agi3.variants import (
    Ls20InitialStateVariant,
    add_random_ls20_shape,
    clone_local_clean_levels,
    prepare_local_initial_state,
    sample_ls20_variant,
)

DEFAULT_ENVIRONMENTS_DIR = "environment_files"
DEFAULT_RECORDINGS_DIR = "recordings"


def load_env_file(path: str | Path = ".env") -> None:
    """Load ARC_API_KEY-style entries from a local .env file if present."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_operation_mode(mode: OperationMode | str) -> OperationMode:
    """Normalize SDK operation mode inputs."""
    if isinstance(mode, OperationMode):
        return mode
    normalized = mode.strip().lower()
    if normalized == OperationMode.NORMAL.value:
        return OperationMode.NORMAL
    if normalized == OperationMode.OFFLINE.value:
        return OperationMode.OFFLINE
    raise ValueError(
        "ARC-AGI-3 adapter supports only OperationMode.NORMAL and "
        "OperationMode.OFFLINE."
    )


def create_arcade(
    *,
    operation_mode: OperationMode | str = OperationMode.NORMAL,
    arc_api_key: str | None = None,
    environments_dir: str = DEFAULT_ENVIRONMENTS_DIR,
    recordings_dir: str = DEFAULT_RECORDINGS_DIR,
    load_dotenv: bool = True,
) -> arc_agi.Arcade:
    """Create an official SDK Arcade instance.

    The SDK already reads ARC_API_KEY from the environment. This helper
    also loads a repository-local .env file first for developer
    convenience.
    """
    if load_dotenv:
        load_env_file()

    kwargs: dict[str, Any] = {
        "operation_mode": parse_operation_mode(operation_mode),
        "environments_dir": environments_dir,
        "recordings_dir": recordings_dir,
    }
    if arc_api_key is not None:
        kwargs["arc_api_key"] = arc_api_key
    return arc_agi.Arcade(**kwargs)


def coerce_action(action: GameAction | int | str) -> GameAction:
    """Convert CLI/user-friendly action inputs into SDK GameAction values."""
    if isinstance(action, GameAction):
        return action

    if isinstance(action, int):
        action_id = action
    else:
        token = action.strip().lower()
        if token.startswith("action"):
            token = token.removeprefix("action")
        action_id = int(token)

    for candidate in GameAction:
        if candidate.value == action_id:
            return candidate
    raise ValueError(f"Unknown ARC-AGI-3 action: {action!r}")


def arc_observation_to_dict(obs: Any) -> Any:
    """Convert SDK observations to pickle-stable plain data."""
    model_dump = getattr(obs, "model_dump", None)
    if callable(model_dump):
        data = model_dump(mode="json")
        frame = getattr(obs, "frame", None)
        if frame is not None and "frame" not in data:
            data["frame"] = _to_jsonable_frame(frame)
        return data
    as_dict = getattr(obs, "dict", None)
    if callable(as_dict):
        data = as_dict()
        frame = getattr(obs, "frame", None)
        if frame is not None and "frame" not in data:
            data["frame"] = _to_jsonable_frame(frame)
        return data
    return obs


def _to_jsonable_frame(frame: Any) -> Any:
    """Convert SDK frame arrays into plain nested lists."""
    if hasattr(frame, "tolist"):
        return frame.tolist()
    if isinstance(frame, dict):
        return {key: _to_jsonable_frame(val) for key, val in frame.items()}
    if isinstance(frame, (list, tuple)):
        return [_to_jsonable_frame(val) for val in frame]
    return frame


class ArcAgi3Env:
    """Small environment/domain adapter for ARC-AGI-3 games."""

    def __init__(
        self,
        game_id: str = "ls20",
        *,
        operation_mode: OperationMode | str = OperationMode.NORMAL,
        arc_api_key: str | None = None,
        environments_dir: str = DEFAULT_ENVIRONMENTS_DIR,
        recordings_dir: str = DEFAULT_RECORDINGS_DIR,
        seed: int = 0,
        render_mode: str | None = None,
        save_recording: bool = False,
        include_frame_data: bool = True,
        load_dotenv: bool = True,
        initial_state_variant: Ls20InitialStateVariant | dict[str, Any] | None = None,
    ) -> None:
        self.game_id = game_id
        self.arcade = create_arcade(
            operation_mode=operation_mode,
            arc_api_key=arc_api_key,
            environments_dir=environments_dir,
            recordings_dir=recordings_dir,
            load_dotenv=load_dotenv,
        )
        self.env = self.arcade.make(
            game_id,
            seed=seed,
            save_recording=save_recording,
            include_frame_data=include_frame_data,
            render_mode=render_mode,
        )
        if self.env is None:
            raise RuntimeError(
                f"Could not create ARC-AGI-3 environment {game_id!r}. "
                "Use NORMAL mode with a registered ARC_API_KEY to fetch it, "
                "or use OFFLINE mode after the game is cached locally."
            )
        self.initial_state_variant = initial_state_variant
        self._baseline_local_levels = clone_local_clean_levels(self.env)
        self.last_initial_state_variant: dict[str, Any] | None = None
        self._last_observation: Any | None = None

    @property
    def action_space(self) -> Sequence[GameAction]:
        """Return the official SDK action space for this game."""
        return self.env.action_space

    @property
    def available_actions(self) -> Sequence[GameAction]:
        """Return currently available actions, falling back to action_space."""
        if self._last_observation is not None:
            available = getattr(self._last_observation, "available_actions", None)
            if available is not None:
                return available
        return self.action_space

    def reset(
        self,
        *,
        initial_state_variant: Ls20InitialStateVariant | dict[str, Any] | None = None,
    ) -> Any:
        """Reset the ARC game and return the raw SDK observation."""
        selected_variant = (
            self.initial_state_variant
            if initial_state_variant is None
            else initial_state_variant
        )
        self.last_initial_state_variant = prepare_local_initial_state(
            game_id=self.game_id,
            sdk_env=self.env,
            baseline_levels=self._baseline_local_levels,
            variant=selected_variant,
        )
        self._last_observation = self.env.reset()
        return self._last_observation

    def sample_initial_state_variant(
        self, seed: int, *, randomize_shape: bool = False
    ) -> Ls20InitialStateVariant | None:
        """Sample a deterministic local initial-state variant for this game."""
        if self.game_id.split("-", maxsplit=1)[0].lower() != "ls20":
            raise NotImplementedError(
                f"Random variants are not registered for {self.game_id!r}."
            )
        variant = sample_ls20_variant(
            baseline_levels=self._baseline_local_levels,
            seed=seed,
        )
        if randomize_shape:
            variant = add_random_ls20_shape(variant, seed=seed)
        return variant

    def step(self, action: GameAction | int | str) -> Any:
        """Take one SDK action and return the raw SDK observation."""
        self._last_observation = self.env.step(coerce_action(action))
        return self._last_observation

    def get_scorecard(self) -> Any:
        """Return the official SDK scorecard for the current Arcade."""
        return self.arcade.get_scorecard()


class ArcAgi3ActionSpace(Space[GameAction]):
    """Minimal Gymnasium-style space over SDK GameAction values."""

    def __init__(self, actions: Sequence[GameAction]) -> None:
        super().__init__(shape=(), dtype=np.int64)
        self.actions = tuple(actions)
        self._rng = np.random.default_rng()

    def seed(self, seed: int | None = None) -> list[int | None]:
        self._rng = np.random.default_rng(seed)
        return [seed]

    def sample(self, mask: Any | None = None) -> GameAction:
        del mask
        if not self.actions:
            raise RuntimeError("ARC-AGI-3 action space is empty.")
        index = int(self._rng.integers(0, len(self.actions)))
        return self.actions[index]

    def contains(self, x: Any) -> bool:
        try:
            action = coerce_action(x)
        except (TypeError, ValueError):
            return False
        return any(candidate.value == action.value for candidate in self.actions)

    def __repr__(self) -> str:
        actions = ", ".join(action.name for action in self.actions)
        return f"ArcAgi3ActionSpace({actions})"


class RawArcAgi3ObservationSpace(Space[Any]):
    """Placeholder space for raw SDK observations."""

    def __init__(self) -> None:
        super().__init__(shape=None, dtype=None)

    def sample(self, mask: Any | None = None) -> Any:
        del mask
        raise NotImplementedError("Raw ARC-AGI-3 observations cannot be sampled.")

    def contains(self, x: Any) -> bool:
        return x is not None


class ArcAgi3GymEnv:
    """Gymnasium-compatible shell for baseline rollouts.

    This keeps the main ArcAgi3Env raw-SDK-facing, while allowing simple
    existing approaches such as RandomActionsApproach to run end to end.
    """

    metadata = {"render_modes": []}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.observation_format = str(kwargs.pop("observation_format", "raw")).lower()
        self.terminate_on_level_complete = bool(
            kwargs.pop("terminate_on_level_complete", False)
        )
        if self.observation_format not in {"raw", "dict", "processed"}:
            raise ValueError(
                "observation_format must be one of {'raw', 'dict', 'processed'}."
            )
        self.raw_env = ArcAgi3Env(*args, **kwargs)
        self.observation_space = RawArcAgi3ObservationSpace()
        self.action_space = ArcAgi3ActionSpace(self.raw_env.action_space)
        self._last_levels_completed = 0
        self.last_reset_seed: int | None = None
        self.last_requested_seed: int | None = None

    @property
    def available_actions(self) -> Sequence[GameAction]:
        """Return currently available SDK actions."""
        return self.raw_env.available_actions

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self.action_space.seed(seed)
        self.last_requested_seed = seed
        self.last_reset_seed = seed
        variant = None if options is None else options.get("initial_state_variant")
        obs = self.raw_env.reset(initial_state_variant=variant)
        self._last_levels_completed = int(getattr(obs, "levels_completed", 0) or 0)
        return self._format_observation(obs), self._make_info(obs)

    def step(
        self, action: GameAction | int | str
    ) -> tuple[Any, float, bool, bool, dict]:
        obs = self.raw_env.step(action)
        levels_completed = int(getattr(obs, "levels_completed", 0) or 0)
        reward = float(levels_completed - self._last_levels_completed)
        level_completed = levels_completed > self._last_levels_completed
        self._last_levels_completed = levels_completed
        sdk_terminated = self._is_terminal(obs)
        terminated = sdk_terminated or (
            self.terminate_on_level_complete and level_completed
        )
        truncated = False
        info = self._make_info(obs)
        info["level_completed"] = level_completed
        info["sdk_terminated"] = sdk_terminated
        info["episode_boundary"] = (
            "game_terminal"
            if sdk_terminated
            else "level_completed" if terminated else None
        )
        return (
            self._format_observation(obs),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self) -> None:
        return None

    def close(self) -> None:
        close_scorecard = getattr(self.raw_env.arcade, "close_scorecard", None)
        if callable(close_scorecard):
            close_scorecard()

    def get_scorecard(self) -> Any:
        """Return the official SDK scorecard."""
        return self.raw_env.get_scorecard()

    def get_object_types(self) -> tuple[str, ...]:
        """Return coarse object/state categories for LPP metadata."""
        return ("arc.frame", "arc.object", "arc.state", "arc.action")

    def get_action_types(self) -> tuple[str, ...]:
        """Return action type names for this ARC action space."""
        return tuple(action.name.lower() for action in self.action_space.actions)

    def get_action_values(self) -> tuple[int, ...]:
        """Return finite discrete action values available to LPP."""
        return tuple(int(action.value) for action in self.action_space.actions)

    def _format_observation(self, obs: Any) -> Any:
        if self.observation_format == "dict":
            return arc_observation_to_dict(obs)
        if self.observation_format == "processed":
            return preprocess_arc_observation(
                obs,
                game_id=self.raw_env.game_id,
                initial_state_variant=self.raw_env.last_initial_state_variant,
            )
        return obs

    def _make_info(self, obs: Any) -> dict[str, Any]:
        return {
            "game_id": getattr(obs, "game_id", self.raw_env.game_id),
            "seed": self.last_reset_seed,
            "requested_seed": self.last_requested_seed,
            "levels_completed": getattr(obs, "levels_completed", None),
            "win_levels": getattr(obs, "win_levels", None),
            "available_actions": getattr(obs, "available_actions", None),
            "scorecard": self.get_scorecard(),
            "initial_state_variant": self.raw_env.last_initial_state_variant,
        }

    def sample_initial_state_variant(
        self, seed: int, *, randomize_shape: bool = False
    ) -> Ls20InitialStateVariant | None:
        """Sample a deterministic local initial-state variant."""
        return self.raw_env.sample_initial_state_variant(
            seed, randomize_shape=randomize_shape
        )

    @staticmethod
    def _is_terminal(obs: Any) -> bool:
        state = getattr(obs, "state", None)
        state_name = str(getattr(state, "name", state))
        return state_name in {"WIN", "GAME_OVER"}


def list_environments(
    *,
    operation_mode: OperationMode | str = OperationMode.NORMAL,
    environments_dir: str = DEFAULT_ENVIRONMENTS_DIR,
    recordings_dir: str = DEFAULT_RECORDINGS_DIR,
    load_dotenv: bool = True,
) -> list[Any]:
    """List environments known to the official SDK."""
    arcade = create_arcade(
        operation_mode=operation_mode,
        environments_dir=environments_dir,
        recordings_dir=recordings_dir,
        load_dotenv=load_dotenv,
    )
    return arcade.get_environments()
