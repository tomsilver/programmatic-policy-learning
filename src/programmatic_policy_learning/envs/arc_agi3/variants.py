"""Offline initial-state variants for official ARC-AGI-3 games."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from random import Random
from typing import Any

Position = tuple[int, int]

_LS20_PLAYER_TAG = "sfqyzhzkij"
_LS20_ROTATION_SWITCH_TAG = "rhsxkxzdjz"
_LS20_TARGET_TAG = "rjlbuycveu"
_LS20_WALL_TAG = "ihdgageizm"
_LS20_NUM_SHAPES = 6


@dataclass(frozen=True)
class Ls20InitialStateVariant:
    """Initial player/switch positions and shared shape for ``ls20`` level 0.

    Coordinates are ARC pixel coordinates for the top-left corner of a sprite.
    ``shape_index`` selects the same shape identity for the large rotating
    target and the small fixed reference. Their positions and rotations remain
    unchanged.
    """

    player_position: Position | None = None
    rotation_switch_position: Position | None = None
    shape_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON/pickle-friendly variant metadata."""
        data = asdict(self)
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in data.items()
            if value is not None
        }


def coerce_ls20_variant(
    value: Ls20InitialStateVariant | dict[str, Any] | None,
) -> Ls20InitialStateVariant | None:
    """Normalize a dataclass/dictionary variant and validate scalar fields."""
    if value is None or isinstance(value, Ls20InitialStateVariant):
        variant = value
    elif isinstance(value, dict):
        data = dict(value)
        for key in ("player_position", "rotation_switch_position"):
            if data.get(key) is not None:
                data[key] = _coerce_position(data[key], key)
        variant = Ls20InitialStateVariant(**data)
    else:
        raise TypeError(
            "initial_state_variant must be an Ls20InitialStateVariant, dict, or None."
        )

    if variant is None:
        return None
    if variant.shape_index is not None and not (
        0 <= variant.shape_index < _LS20_NUM_SHAPES
    ):
        raise ValueError("ls20 shape_index must be between 0 and 5.")
    return variant


def clone_local_clean_levels(sdk_env: Any) -> list[Any] | None:
    """Clone the official local wrapper's pristine levels, if available."""
    game = getattr(sdk_env, "_game", None)
    clean_levels = getattr(game, "_clean_levels", None)
    if game is None or clean_levels is None:
        return None
    return [level.clone() for level in clean_levels]


def prepare_local_initial_state(
    *,
    game_id: str,
    sdk_env: Any,
    baseline_levels: list[Any] | None,
    variant: Ls20InitialStateVariant | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Restore pristine local levels and apply a variant before SDK reset."""
    normalized = coerce_ls20_variant(variant)
    game = getattr(sdk_env, "_game", None)
    if game is not None and baseline_levels is not None:
        game._clean_levels = [level.clone() for level in baseline_levels]
    if normalized is None:
        return None
    if game_id.split("-", maxsplit=1)[0].lower() != "ls20":
        raise NotImplementedError(
            f"Initial-state variants are not registered for ARC game {game_id!r}."
        )

    if game is None or baseline_levels is None:
        raise RuntimeError(
            "ARC initial-state variants require OFFLINE mode and a locally cached game."
        )

    level = game._clean_levels[0]
    _apply_ls20_variant(level, normalized)
    return normalized.to_dict()


def sample_ls20_variant(
    *,
    baseline_levels: list[Any] | None,
    seed: int,
) -> Ls20InitialStateVariant | None:
    """Sample a deterministic, structurally valid level-0 ``ls20`` variant."""
    if seed == 0:
        return None
    if not baseline_levels:
        raise RuntimeError(
            "Random ls20 variants require OFFLINE mode and a locally cached game."
        )
    level = baseline_levels[0].clone()
    player = _one_sprite(level, _LS20_PLAYER_TAG)
    switch = _one_sprite(level, _LS20_ROTATION_SWITCH_TAG)
    target = _one_sprite(level, _LS20_TARGET_TAG)
    player_start = (player.x, player.y)
    switch_start = (switch.x, switch.y)
    target_position = (target.x, target.y)
    walkable = _ls20_walkable_positions(level, player)
    player_candidates = [
        position
        for position in walkable
        if position not in {player_start, target_position}
    ]
    if not player_candidates:
        raise RuntimeError("Could not find a new walkable cell for the ls20 player.")

    rng = Random(seed)
    player_position = rng.choice(player_candidates)
    switch_candidates = [
        position
        for position in walkable
        if position not in {switch_start, target_position, player_position}
    ]
    if not switch_candidates:
        raise RuntimeError("Could not find two walkable cells for an ls20 variant.")
    switch_position = rng.choice(switch_candidates)

    return Ls20InitialStateVariant(
        player_position=player_position,
        rotation_switch_position=switch_position,
    )


def add_random_ls20_shape(
    variant: Ls20InitialStateVariant | None,
    *,
    seed: int,
) -> Ls20InitialStateVariant | None:
    """Add a deterministic shared shape change for Stage-B experiments."""
    if seed == 0:
        return None
    rng = Random(seed)
    shape_index = rng.randrange(_LS20_NUM_SHAPES)
    if variant is None:
        return Ls20InitialStateVariant(shape_index=shape_index)
    return Ls20InitialStateVariant(
        player_position=variant.player_position,
        rotation_switch_position=variant.rotation_switch_position,
        shape_index=shape_index,
    )


def _apply_ls20_variant(level: Any, variant: Ls20InitialStateVariant) -> None:
    player = _one_sprite(level, _LS20_PLAYER_TAG)
    switch = _one_sprite(level, _LS20_ROTATION_SWITCH_TAG)

    if variant.player_position is not None:
        _validate_board_position(level, player, variant.player_position, "player")
        player.set_position(*variant.player_position)
    if variant.rotation_switch_position is not None:
        _validate_board_position(
            level,
            switch,
            variant.rotation_switch_position,
            "rotation switch",
        )
        switch.set_position(*variant.rotation_switch_position)
    if (
        variant.player_position is not None
        and variant.rotation_switch_position is not None
        and variant.player_position == variant.rotation_switch_position
    ):
        raise ValueError(
            "ls20 player and rotation switch cannot start on the same cell."
        )

    _validate_no_wall_overlap(level, player, "player")
    _validate_no_wall_overlap(level, switch, "rotation switch")
    _validate_not_on_target(level, player, "player")
    _validate_not_on_target(level, switch, "rotation switch")
    if variant.shape_index is not None:
        level._data["StartShape"] = variant.shape_index
        level._data["kvynsvxbpi"] = variant.shape_index


def _ls20_walkable_positions(level: Any, player: Any) -> list[Position]:
    width, height = level.grid_size
    step_x, step_y = player.width, player.height
    original_position = (player.x, player.y)
    positions: set[Position] = set()
    for y in range(player.y % step_y, height - player.height + 1, step_y):
        for x in range(player.x % step_x, width - player.width + 1, step_x):
            player.set_position(x, y)
            if not _overlaps_tag(level, player, _LS20_WALL_TAG):
                positions.add((x, y))

    component: list[Position] = []
    frontier = [original_position]
    visited: set[Position] = set()
    while frontier:
        position = frontier.pop()
        if position in visited or position not in positions:
            continue
        visited.add(position)
        component.append(position)
        x, y = position
        frontier.extend(
            (
                (x - step_x, y),
                (x + step_x, y),
                (x, y - step_y),
                (x, y + step_y),
            )
        )
    return sorted(component)


def _validate_board_position(
    level: Any,
    sprite: Any,
    position: Position,
    name: str,
) -> None:
    _validate_bounds(level, sprite, position, name)
    if position[0] % sprite.width != sprite.x % sprite.width:
        raise ValueError(
            f"ls20 {name} x={position[0]} is off the {sprite.width}-pixel movement grid."
        )
    if position[1] % sprite.height != sprite.y % sprite.height:
        raise ValueError(
            f"ls20 {name} y={position[1]} is off the {sprite.height}-pixel movement grid."
        )


def _validate_bounds(level: Any, sprite: Any, position: Position, name: str) -> None:
    width, height = level.grid_size
    x, y = position
    if x < 0 or y < 0 or x + sprite.width > width or y + sprite.height > height:
        raise ValueError(
            f"ls20 {name} position {position} is outside the {width}x{height} board."
        )


def _validate_no_wall_overlap(level: Any, sprite: Any, name: str) -> None:
    if _overlaps_tag(level, sprite, _LS20_WALL_TAG):
        raise ValueError(f"ls20 {name} position overlaps a wall.")


def _validate_not_on_target(level: Any, sprite: Any, name: str) -> None:
    if _overlaps_tag(level, sprite, _LS20_TARGET_TAG):
        raise ValueError(f"ls20 {name} position overlaps the fixed target cell.")


def _overlaps_tag(level: Any, sprite: Any, tag: str) -> bool:
    return any(
        sprite.collides_with(other, ignoreMode=True)
        for other in level.get_sprites_by_tag(tag)
        if other is not sprite
    )


def _one_sprite(level: Any, tag: str) -> Any:
    matches = level.get_sprites_by_tag(tag)
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one ls20 sprite tagged {tag!r}.")
    return matches[0]


def _coerce_position(value: Any, name: str) -> Position:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-item [x, y] coordinate.")
    return int(value[0]), int(value[1])
