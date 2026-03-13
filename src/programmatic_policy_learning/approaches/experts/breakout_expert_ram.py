from typing import Any, Callable
import numpy as np

ExpertPolicy = Callable[[Any], Any]


def make_breakout_expert_ram() -> ExpertPolicy:
    """Breakout expert for RAM observations (128-dim uint8)."""

    NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3

    # RAM indices (0-based) for Breakout (mila-iqia annotations)
    PLAYER_X_IDX = 72
    BALL_X_IDX = 99
    BALL_Y_IDX = 101

    steps_since_reset = 0
    prev_ball_x: int | None = None
    prev_ball_y: int | None = None

    def reset() -> None:
        nonlocal steps_since_reset, prev_ball_x, prev_ball_y
        steps_since_reset = 0
        prev_ball_x = None
        prev_ball_y = None

    def expert(obs: Any) -> int:
        nonlocal steps_since_reset, prev_ball_x, prev_ball_y
        steps_since_reset += 1

        # Serve logic: fire a few times right after reset / life loss
        if steps_since_reset <= 10:
            return FIRE

        if not isinstance(obs, np.ndarray) or obs.ndim != 1 or obs.shape[0] < 102:
            return FIRE

        ram = obs.astype(np.int32, copy=False)
        paddle_x = int(ram[PLAYER_X_IDX])
        ball_x = int(ram[BALL_X_IDX])
        ball_y = int(ram[BALL_Y_IDX])

        # If ball seems "not active" / just served, keep firing briefly
        # (ball_y can jump around on serve/death; this keeps it robust)
        if prev_ball_y is not None and ball_y > 240 and prev_ball_y < 50:
            reset()
            return FIRE

        # Basic tracking: keep paddle under ball
        # Optional small "lead" when ball is moving fast
        lead = 0
        if prev_ball_x is not None:
            vx = ball_x - prev_ball_x
            # clamp lead so we don't overreact
            if vx > 0:
                lead = 2
            elif vx < 0:
                lead = -2

        target_x = ball_x + lead

        prev_ball_x, prev_ball_y = ball_x, ball_y

        # Deadband to avoid jitter
        if target_x > paddle_x + 2:
            return RIGHT
        if target_x < paddle_x - 2:
            return LEFT
        return NOOP

    expert.reset = reset  # type: ignore[attr-defined]
    return expert
