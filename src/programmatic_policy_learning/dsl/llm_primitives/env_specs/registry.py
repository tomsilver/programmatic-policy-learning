"""Registry for environment-specific LLM serialization behavior.

This layer centralizes the metadata that differs across environments
while keeping the rest of the LPP pipeline modality-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based import (
    continuous_hint_config,
    continuous_trajectory_serializer,
    grid_encoder,
    grid_hint_config,
    trajectory_serializer,
    transition_analyzer,
)

# pylint: disable-next=line-too-long
from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based.continuous_encoder import (
    ContinuousStateEncoder,
    ContinuousStateEncoderConfig,
)

DiscreteTrajectory = list[tuple[Any, Any, Any]]
ContinuousTrajectory = list[tuple[Any, Any, Any]]

_DISCRETE_TOKEN_MAPS: dict[str, dict[str, str]] = {
    "StopTheFall": {
        "empty": "stf.EMPTY",
        "red_token": "stf.RED",
        "static_token": "stf.STATIC",
        "falling_token": "stf.FALLING",
        "drawn_token": "stf.DRAWN",
        "advance_token": "stf.ADVANCE",
    },
    "TwoPileNim": {
        "empty": "tpn.EMPTY",
        "token": "tpn.TOKEN",
    },
    "ReachForTheStar": {
        "empty": "rfts.EMPTY",
        "agent": "rfts.AGENT",
        "star": "rfts.STAR",
        "left_arrow": "rfts.LEFT_ARROW",
        "right_arrow": "rfts.RIGHT_ARROW",
        "drawn": "rfts.DRAWN",
    },
    "Chase": {
        "empty": "ec.EMPTY",
        "agent": "ec.AGENT",
        "target": "ec.TARGET",
        "wall": "ec.WALL",
        "drawn": "ec.DRAWN",
        "left_arrow": "ec.LEFT_ARROW",
        "right_arrow": "ec.RIGHT_ARROW",
        "up_arrow": "ec.UP_ARROW",
        "down_arrow": "ec.DOWN_ARROW",
    },
    "CheckmateTactic": {
        "empty": "ct.EMPTY",
        "black_king": "ct.BLACK_KING",
        "white_king": "ct.WHITE_KING",
        "white_queen": "ct.WHITE_QUEEN",
        "highlighted_white_king": "ct.HIGHLIGHTED_WHITE_KING",
        "highlighted_white_queen": "ct.HIGHLIGHTED_WHITE_QUEEN",
    },
}


def _normalize_encoding_method(encoding_method: str) -> str:
    return str(encoding_method).replace("enc_", "")


@dataclass(frozen=True)
class EnvLLMSpec:
    """Environment-specific adapter for trajectory text and prompt metadata."""

    env_name: str
    action_mode: str

    def serialize_demonstrations(
        self,
        trajectories: list[list[tuple[Any, Any, Any]]],
        *,
        encoding_method: str,
        max_steps: int = 50,
    ) -> str:
        """Render expert demonstrations into prompt-ready text."""
        raise NotImplementedError

    def token_map(self) -> dict[str, str]:
        """Return a raw-token to canonical-token map when applicable."""
        return {}


@dataclass(frozen=True)
class GridEnvLLMSpec(EnvLLMSpec):
    """Discrete grid-world adapter."""

    def serialize_demonstrations(
        self,
        trajectories: list[list[tuple[Any, Any, Any]]],
        *,
        encoding_method: str,
        max_steps: int = 50,
    ) -> str:
        enc_method = str(encoding_method)
        enc_id = _normalize_encoding_method(enc_method)
        symbol_map = grid_hint_config.get_symbol_map(self.env_name)
        encoder = grid_encoder.GridStateEncoder(
            grid_encoder.GridStateEncoderConfig(
                symbol_map=symbol_map,
                empty_token="empty",
                coordinate_style="rc",
            )
        )
        analyzer = transition_analyzer.GenericTransitionAnalyzer()
        salient_tokens = grid_hint_config.SALIENT_TOKENS[self.env_name]

        all_traj_texts: list[str] = []
        for i, traj in enumerate(trajectories):
            if enc_method == "enc_1":
                text = trajectory_serializer.trajectory_to_diff_text(
                    traj,
                    encoder=encoder,
                    max_steps=max_steps,
                )
            else:
                text = trajectory_serializer.trajectory_to_text(
                    traj,
                    encoder=encoder,
                    analyzer=analyzer,
                    salient_tokens=salient_tokens,
                    encoding_method=enc_id,
                    max_steps=max_steps,
                )
            all_traj_texts.append(f"\n---[TRAJECTORY {i}]---\n{text}\n\n")
        return "\n\n".join(all_traj_texts)

    def token_map(self) -> dict[str, str]:
        mapping = _DISCRETE_TOKEN_MAPS.get(self.env_name)
        if mapping is None:
            raise KeyError(f"No token map configured for {self.env_name}")

        symbol_map = grid_hint_config.get_symbol_map(self.env_name)
        token_map: dict[str, str] = {}
        for token_name, raw_symbol in symbol_map.items():
            canonical_name = mapping.get(token_name)
            if canonical_name is None:
                raise KeyError(
                    f"Missing canonical token mapping for {self.env_name}: {token_name}"
                )
            token_map[raw_symbol] = canonical_name
        return token_map


@dataclass(frozen=True)
class KinderContinuousEnvLLMSpec(EnvLLMSpec):
    """Continuous KinDER adapter."""

    num_passages: int = 0

    def serialize_demonstrations(
        self,
        trajectories: list[list[tuple[Any, Any, Any]]],
        *,
        encoding_method: str,
        max_steps: int = 50,
    ) -> str:
        obs_field_names = continuous_hint_config.obs_field_names_for_kinder(
            self.env_name,
            self.num_passages,
        )
        encoder = ContinuousStateEncoder(
            ContinuousStateEncoderConfig(
                obs_field_names=obs_field_names,
                action_field_names=continuous_hint_config.ACTION_FIELD_NAMES[
                    self.env_name
                ],
                salient_indices=continuous_hint_config.salient_obs_indices_for_kinder(
                    self.env_name,
                    self.num_passages,
                ),
            )
        )
        enc_id = _normalize_encoding_method(encoding_method)

        all_traj_texts: list[str] = []
        for i, traj in enumerate(trajectories):
            text = continuous_trajectory_serializer.trajectory_to_text(
                traj,
                encoder=encoder,
                num_passages=self.num_passages,
                encoding_method=enc_id,
                env_name=self.env_name,
                max_steps=max_steps,
            )
            all_traj_texts.append(f"\n---[TRAJECTORY {i}]---\n{text}\n\n")
        return "\n\n".join(all_traj_texts)


def _resolve_motion2d_passages(
    env_name: str,
    env_specs: dict[str, Any] | None,
) -> int:
    if env_specs and "num_passages" in env_specs:
        return int(env_specs["num_passages"])
    if "-p" in env_name:
        try:
            return int(env_name.rsplit("-p", maxsplit=1)[1])
        except ValueError:
            return 0
    return 0


def get_env_llm_spec(
    env_name: str,
    env_specs: dict[str, Any] | None = None,
) -> EnvLLMSpec:
    """Return the LLM adapter for the given environment."""
    action_mode = str((env_specs or {}).get("action_mode", "discrete"))
    canonical_name = continuous_hint_config.canonicalize_env_name(
        env_name.split("-p")[0]
    )

    if action_mode == "continuous":
        try:
            continuous_hint_config.obs_field_names_for_kinder(
                canonical_name,
                _resolve_motion2d_passages(env_name, env_specs),
            )
        except ValueError as exc:
            raise ValueError(
                f"No LLM environment spec registered for env_name={env_name!r}, "
                f"action_mode={action_mode!r}"
            ) from exc
        return KinderContinuousEnvLLMSpec(
            env_name=canonical_name,
            action_mode=action_mode,
            num_passages=_resolve_motion2d_passages(env_name, env_specs),
        )
    if action_mode == "discrete":
        return GridEnvLLMSpec(env_name=env_name, action_mode=action_mode)
    raise ValueError(
        f"No LLM environment spec registered for env_name={env_name!r}, "
        f"action_mode={action_mode!r}"
    )
