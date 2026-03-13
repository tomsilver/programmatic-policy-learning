from typing import Any, Callable
import numpy as np

ExpertPolicy = Callable[[Any], Any]


def make_breakout_expert() -> ExpertPolicy:
    """Simple Breakout expert with a tiny amount of memory to ensure serving."""

    NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3
    steps_since_reset = 0

    def expert(obs: np.ndarray) -> int:
        nonlocal steps_since_reset
        steps_since_reset += 1

        # Always FIRE a few times right after reset to launch the ball
        if steps_since_reset <= 10:
            return FIRE

        if not isinstance(obs, np.ndarray) or obs.ndim != 3:
            return FIRE

        gray = obs.mean(axis=2)

        # Lower threshold: ball/paddle often aren't above 200 consistently
        ys, xs = np.where(gray > 150)

        if xs.size == 0:
            return FIRE

        # Paddle near bottom
        paddle_pixels = ys >= 185
        paddle_x = xs[paddle_pixels].mean() if np.any(paddle_pixels) else 80.0

        # Ball above paddle band (ignore top-most bricks a bit)
        ball_pixels = (ys < 175) & (ys > 30)
        if not np.any(ball_pixels):
            return FIRE

        ball_x = xs[ball_pixels].mean()

        if ball_x > paddle_x + 2:
            return RIGHT
        if ball_x < paddle_x - 2:
            return LEFT
        return NOOP

    def reset() -> None:
        nonlocal steps_since_reset
        steps_since_reset = 0

    # attach a reset hook (optional but convenient)
    expert.reset = reset  # type: ignore[attr-defined]
    return expert
