from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import ActionWrapper, ObservationWrapper
from gymnasium.spaces import Box, Discrete


# ============================================================
# RAM tracker (updated ONLY by action wrapper)
# ============================================================
@dataclass
class TrackState:
    ok: bool
    paddle_x: float | None
    paddle_y: float | None
    ball_x: float | None
    ball_y: float | None
    vx: float | None
    vy: float | None
    dx: float | None  # ball_x - paddle_x


class BreakoutRAMTracker:
    """
    Breakout RAM indices (as used here):
      paddle_x = ram[72]
      ball_x   = ram[99]
      ball_y   = ram[101]

    IMPORTANT: Call update() exactly once per env step.
    """

    def __init__(
        self,
        *,
        paddle_x_idx: int = 72,
        ball_x_idx: int = 99,
        ball_y_idx: int = 101,
        paddle_y_px: float = 190.0,  # kept for compatibility, not used for hit logic anymore
    ):
        self.paddle_x_idx = int(paddle_x_idx)
        self.ball_x_idx = int(ball_x_idx)
        self.ball_y_idx = int(ball_y_idx)
        self.paddle_y_px = float(paddle_y_px)

        self._prev_ball_x: float | None = None
        self._prev_ball_y: float | None = None

        # NEW: adaptive y range for RAM ball_y scale
        self.y_min: float | None = None
        self.y_max: float | None = None

        self.last = TrackState(False, None, None, None, None, None, None, None)

    def reset(self) -> None:
        self._prev_ball_x = None
        self._prev_ball_y = None
        self.y_min = None
        self.y_max = None
        self.last = TrackState(False, None, None, None, None, None, None, None)

    def _update_y_range(self, y: float) -> None:
        if self.y_min is None:
            self.y_min = y
            self.y_max = y
        else:
            self.y_min = float(min(self.y_min, y))
            self.y_max = float(max(self.y_max, y))

    def y_trigger(self, frac: float = 0.80, margin: float = 0.0) -> float | None:
        """
        Returns a "near bottom" threshold in the *current* RAM y scale,
        based on observed range this life.
        """
        if self.y_min is None or self.y_max is None:
            return None
        rng = max(1e-6, float(self.y_max - self.y_min))
        return float(self.y_min + frac * rng + margin)

    def update(self, obs_ram: Any) -> TrackState:
        obs = np.asarray(obs_ram)
        if not (isinstance(obs, np.ndarray) and obs.ndim == 1 and obs.size >= 128):
            self.last = TrackState(False, None, None, None, None, None, None, None)
            return self.last

        ram = obs.astype(np.float32, copy=False)
        paddle_x = float(ram[self.paddle_x_idx])
        ball_x = float(ram[self.ball_x_idx])
        ball_y = float(ram[self.ball_y_idx])
        paddle_y = float(self.paddle_y_px)

        # NEW: update RAM y range (for adaptive gating/hit detection)
        self._update_y_range(ball_y)

        vx = None if self._prev_ball_x is None else float(ball_x - self._prev_ball_x)
        vy = None if self._prev_ball_y is None else float(ball_y - self._prev_ball_y)

        self._prev_ball_x = ball_x
        self._prev_ball_y = ball_y

        dx = float(ball_x - paddle_x)

        self.last = TrackState(
            ok=True,
            paddle_x=paddle_x,
            paddle_y=paddle_y,
            ball_x=ball_x,
            ball_y=ball_y,
            vx=vx,
            vy=vy,
            dx=dx,
        )
        return self.last


# ============================================================
# Feature obs wrapper (READS tracker.last; does NOT update)
# ============================================================
class BreakoutHitPointFeatureObsWrapper(ObservationWrapper):
    """
    Features (6D) — unchanged from your version:
      0 dx_now/W
      1 vx/W
      2 vy/H
      3 ball_y/H
      4 paddle_x/W
      5 bias (1.0)

    This wrapper DOES NOT call tracker.update(). The action wrapper owns updates.
    """

    def __init__(self, env: gym.Env, tracker: BreakoutRAMTracker):
        super().__init__(env)
        self._tracker = tracker
        self._W = 160.0
        self._H = 210.0
        self.observation_space = Box(low=-2.0, high=2.0, shape=(6,), dtype=np.float32)

    def _feat(self) -> tuple[np.ndarray, bool]:
        st = self._tracker.last
        if not st.ok or st.paddle_x is None or st.ball_y is None:
            return np.zeros((6,), dtype=np.float32), False

        dx_now = float(st.dx) if st.dx is not None else 0.0
        vx = float(st.vx) if st.vx is not None else 0.0
        vy = float(st.vy) if st.vy is not None else 0.0
        ball_y = float(st.ball_y) if st.ball_y is not None else 0.0
        paddle_x = float(st.paddle_x)

        return (
            np.array(
                [
                    dx_now / self._W,
                    vx / self._W,
                    vy / self._H,
                    ball_y / self._H,
                    paddle_x / self._W,
                    1.0,
                ],
                dtype=np.float32,
            ),
            True,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        feat, ok = self._feat()
        info = dict(info)
        info["ram_ok"] = bool(ok)
        return feat, info

    def observation(self, obs: Any) -> np.ndarray:
        feat, _ = self._feat()
        return feat

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        feat, ok = self._feat()
        info = dict(info)
        info["ram_ok"] = bool(ok)
        return feat, float(rew), bool(term), bool(trunc), info


# ============================================================
# Residual hit-point control (owns tracker updates)
# ============================================================
class ResidualHitPointControlWrapper(ActionWrapper):
    """
    Your original wrapper, with ONE key fix:
      - replace hardcoded pixel y thresholds with adaptive RAM y_trigger()

    This makes:
      - preimpact window reachable (residual can actually act)
      - hit detection reachable (post-hit shaping can trigger)
    """

    NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3

    def __init__(
        self,
        env: gym.Env,
        base_policy: Callable[[Any], int],
        tracker: BreakoutRAMTracker,
        *,
        # OLD params kept, but y_min_for_residual is no longer used for gating
        y_min_for_residual: float = 165.0,  # kept for compatibility (ignored)
        gate_px: float = 20.0,              # slightly wider than 12 so residual has chances
        delta_x_max: float = 10.0,          # a bit more authority than 8
        move_deadband_px: float = 2.0,

        cooldown_steps: int = 8,            # shorter cooldown -> more learning chances
        cooldown_on_life_loss: bool = True,

        # OLD pixel params kept, but not used for hit detection anymore
        paddle_y_px: float = 190.0,
        hit_y_band: float = 14.0,

        posthit_window: int = 45,
        posthit_y_ref: float = 0.55,        # now treated as "fraction" target in normalized y (0..1)
        k_post_y: float = 0.10,
        k_post_vx: float = 0.03,

        k_u: float = 0.0007,

        # Serve handling
        serve_steps: int = 10,
        stuck_steps: int = 120,

        # NEW: adaptive y trigger config
        trigger_frac: float = 0.80,         # bottom 20% of observed y-range
        trigger_margin: float = 0.0,

        debug: bool = True,
    ):
        super().__init__(env)
        if not isinstance(env.action_space, Discrete) or int(env.action_space.n) < 4:
            raise TypeError("Expected Breakout-like Discrete action space with >=4 actions.")

        self._base = base_policy
        self._tracker = tracker

        self._W = 160.0
        self._H = 210.0

        self._gate_px = float(gate_px)
        self._delta_x_max = float(delta_x_max)
        self._move_deadband = float(move_deadband_px)

        self._cooldown_steps = int(cooldown_steps)
        self._cooldown_on_life_loss = bool(cooldown_on_life_loss)
        self._cooldown_left = 0

        self._posthit_window = int(posthit_window)
        self._posthit_left = 0
        self._posthit_y_ref_frac = float(posthit_y_ref)  # target y_norm fraction
        self._k_post_y = float(k_post_y)
        self._k_post_vx = float(k_post_vx)

        self._k_u = float(k_u)
        self._debug = bool(debug)

        # Serve + stuck handling
        self._serve_steps = int(serve_steps)
        self._serve_left = 0
        self._stuck_steps = int(stuck_steps)
        self._no_motion_left = self._stuck_steps
        self._prev_ball_pos: tuple[float, float] | None = None

        # Adaptive y trigger config
        self._trigger_frac = float(trigger_frac)
        self._trigger_margin = float(trigger_margin)

        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._last_obs_raw: Any | None = None
        self._prev_lives: int | None = None

        # debug counters
        self._dbg_total = 0
        self._dbg_used = 0
        self._dbg_hit = 0

    def reset(self, **kwargs):
        self._tracker.reset()
        obs, info = self.env.reset(**kwargs)
        self._last_obs_raw = obs

        _ = self._tracker.update(obs)

        self._cooldown_left = self._cooldown_steps
        self._posthit_left = 0

        lives = info.get("lives", None)
        self._prev_lives = int(lives) if lives is not None else None

        # start in serve mode
        self._serve_left = self._serve_steps

        # reset stuck detector
        self._no_motion_left = self._stuck_steps
        self._prev_ball_pos = None

        self._dbg_total = 0
        self._dbg_used = 0
        self._dbg_hit = 0

        if hasattr(self._base, "reset") and callable(getattr(self._base, "reset")):
            self._base.reset()  # type: ignore[attr-defined]

        return obs, info

    def _move_toward(self, target_x: float, paddle_x: float) -> int:
        err = float(target_x - paddle_x)
        if err > self._move_deadband:
            return self.RIGHT
        if err < -self._move_deadband:
            return self.LEFT
        return self.NOOP

    def _check_stuck(self, st: TrackState) -> None:
        if not st.ok or st.ball_x is None or st.ball_y is None:
            self._no_motion_left = self._stuck_steps
            self._prev_ball_pos = None
            return

        pos = (float(st.ball_x), float(st.ball_y))
        if self._prev_ball_pos is None:
            self._prev_ball_pos = pos
            self._no_motion_left = self._stuck_steps
            return

        if pos == self._prev_ball_pos:
            self._no_motion_left -= 1
        else:
            self._no_motion_left = self._stuck_steps
        self._prev_ball_pos = pos

        if self._no_motion_left <= 0:
            self._serve_left = max(self._serve_left, self._serve_steps)
            self._no_motion_left = self._stuck_steps

    def step(self, action: np.ndarray):
        if self._last_obs_raw is None:
            obs, info = self.reset()
            self._last_obs_raw = obs

        u = float(np.asarray(action, dtype=np.float32).reshape(1)[0])
        u = float(np.clip(u, -1.0, 1.0))
        delta_x = u * self._delta_x_max

        in_cooldown = self._cooldown_left > 0
        if in_cooldown:
            self._cooldown_left -= 1

        st = self._tracker.last
        self._check_stuck(st)

        base_a = int(self._base(self._last_obs_raw))
        base_a = int(np.clip(base_a, 0, int(self.env.action_space.n) - 1))

        used_residual = False

        # Serve mode forces FIRE
        if self._serve_left > 0:
            env_a = self.FIRE
            self._serve_left -= 1
        else:
            env_a = base_a

            # residual only if not FIRE
            if (
                (not in_cooldown)
                and st.ok
                and st.vy is not None
                and st.ball_x is not None
                and st.ball_y is not None
                and st.paddle_x is not None
                and base_a != self.FIRE
            ):
                vy = float(st.vy)
                ball_y = float(st.ball_y)
                dx_now = float(st.dx) if st.dx is not None else 0.0

                # NEW: adaptive bottom-of-screen trigger in RAM coordinates
                y_trig = self._tracker.y_trigger(frac=self._trigger_frac, margin=self._trigger_margin)

                preimpact_ok = (
                    (vy > 0.0)
                    and (y_trig is not None)
                    and (ball_y >= y_trig)
                    and (abs(dx_now) <= self._gate_px)
                )

                if preimpact_ok:
                    x_target = float(np.clip(float(st.ball_x) + delta_x, 0.0, self._W - 1.0))
                    env_a = self._move_toward(x_target, float(st.paddle_x))
                    used_residual = True

        # step env
        obs_next, rew_raw, term, trunc, env_info = self.env.step(int(env_a))
        self._last_obs_raw = obs_next

        st_next = self._tracker.update(obs_next)
        rew = float(rew_raw)

        # NEW: hit detection uses vy flip AND "near bottom" (adaptive), not paddle_y=190
        hit_detected = False
        if (
            st.ok
            and st.vy is not None
            and st_next.ok
            and st_next.vy is not None
            and st_next.ball_y is not None
        ):
            vy_prev = float(st.vy)
            vy_now = float(st_next.vy)

            y_trig_next = self._tracker.y_trigger(frac=self._trigger_frac, margin=self._trigger_margin)
            near_bottom = (y_trig_next is not None) and (float(st_next.ball_y) >= y_trig_next)

            if (vy_prev > 0.0) and (vy_now < 0.0) and near_bottom:
                hit_detected = True
                self._posthit_left = self._posthit_window

        # post-hit shaping (still simple, but now it can trigger)
        post_y_term = None
        post_vx_term = None
        if self._posthit_left > 0 and st_next.ok and st_next.ball_y is not None:
            if self._tracker.y_min is not None and self._tracker.y_max is not None:
                rng = max(1e-6, float(self._tracker.y_max - self._tracker.y_min))
                y_norm = (float(st_next.ball_y) - float(self._tracker.y_min)) / rng
                y_norm = float(np.clip(y_norm, 0.0, 1.0))
            else:
                y_norm = 0.5

            # reward: ball higher after hit => smaller y_norm
            y_term = float(np.clip(self._posthit_y_ref_frac - y_norm, -1.0, 1.0))
            rew += self._k_post_y * y_term
            post_y_term = y_term

            vx_term = 0.0
            if st_next.vx is not None:
                vx_term = float(abs(float(st_next.vx)) / max(1.0, self._W))
                vx_term = float(np.clip(vx_term, 0.0, 1.0))
                rew += self._k_post_vx * vx_term
            post_vx_term = vx_term

            self._posthit_left -= 1

        # effort penalty (only when residual actually used)
        if used_residual:
            rew -= self._k_u * abs(u)

        # life-loss handling: re-serve + reset tracker memory
        lives = env_info.get("lives", None) if isinstance(env_info, dict) else None
        if lives is not None:
            lives_i = int(lives)
            if self._prev_lives is not None and lives_i < self._prev_lives:
                self._serve_left = self._serve_steps

                # reset tracker so vx/vy and y-range don't carry garbage into new serve
                self._tracker.reset()
                _ = self._tracker.update(obs_next)

                if self._cooldown_on_life_loss:
                    self._cooldown_left = self._cooldown_steps
                self._posthit_left = 0

                self._no_motion_left = self._stuck_steps
                self._prev_ball_pos = None

                if hasattr(self._base, "reset") and callable(getattr(self._base, "reset")):
                    self._base.reset()  # type: ignore[attr-defined]

            self._prev_lives = lives_i

        # debug rates
        if self._debug:
            self._dbg_total += 1
            if used_residual:
                self._dbg_used += 1
            if hit_detected:
                self._dbg_hit += 1

        # include y_trigger in info so you can see if the window is reachable
        y_trig_dbg = self._tracker.y_trigger(frac=self._trigger_frac, margin=self._trigger_margin)

        info: dict[str, Any] = {
            "reward_raw": float(rew_raw),
            "reward_total": float(rew),
            "used_residual": bool(used_residual),
            "hit_detected": bool(hit_detected),
            "posthit_left": int(self._posthit_left),
            "in_cooldown": bool(in_cooldown),
            "cooldown_left": int(self._cooldown_left),
            "serve_left": int(self._serve_left),
            "u": float(u),
            "delta_x": float(delta_x),
            "base_action": int(base_a),
            "env_action": int(env_a),
            "ram_ok": bool(st_next.ok),
            "used_rate_dbg": (self._dbg_used / max(1, self._dbg_total)),
            "hit_rate_dbg": (self._dbg_hit / max(1, self._dbg_total)),
            "lives": int(self._prev_lives) if self._prev_lives is not None else None,
            "y_min": float(self._tracker.y_min) if self._tracker.y_min is not None else float("nan"),
            "y_max": float(self._tracker.y_max) if self._tracker.y_max is not None else float("nan"),
            "y_trigger": float(y_trig_dbg) if y_trig_dbg is not None else float("nan"),
        }
        if post_y_term is not None:
            info["posthit_y_term"] = float(post_y_term)
        if post_vx_term is not None:
            info["posthit_vx_term"] = float(post_vx_term)

        if isinstance(env_info, dict):
            info.update(env_info)

        return obs_next, float(rew), bool(term), bool(trunc), info
