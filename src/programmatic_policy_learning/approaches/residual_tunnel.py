# residual_tunnel.py
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
    Breakout RAM indices (common for ALE Breakout):
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
        paddle_y_px: float = 190.0,
    ):
        self.paddle_x_idx = int(paddle_x_idx)
        self.ball_x_idx = int(ball_x_idx)
        self.ball_y_idx = int(ball_y_idx)
        self.paddle_y_px = float(paddle_y_px)

        self._prev_ball_x: float | None = None
        self._prev_ball_y: float | None = None

        self.last = TrackState(False, None, None, None, None, None, None, None)

    def reset(self) -> None:
        self._prev_ball_x = None
        self._prev_ball_y = None
        self.last = TrackState(False, None, None, None, None, None, None, None)

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
class BreakoutTunnelFeatureObsWrapper(ObservationWrapper):
    """
    Features (8D), all normalized:
      0 dx_now/W
      1 vx/W
      2 vy/H
      3 ball_y/H
      4 ball_x/W
      5 paddle_x/W
      6 edge_dist_norm in [0,0.5]  (distance to nearest wall) / W
      7 bias = 1

    This wrapper DOES NOT call tracker.update(). The action wrapper owns updates.
    """

    def __init__(self, env: gym.Env, tracker: BreakoutRAMTracker, *, W: float = 160.0, H: float = 210.0):
        super().__init__(env)
        self._tracker = tracker
        self._W = float(W)
        self._H = float(H)
        self.observation_space = Box(low=-2.0, high=2.0, shape=(8,), dtype=np.float32)

    def _feat(self) -> tuple[np.ndarray, bool]:
        st = self._tracker.last
        if not st.ok or st.paddle_x is None or st.ball_x is None or st.ball_y is None:
            return np.zeros((8,), dtype=np.float32), False

        dx_now = float(st.dx) if st.dx is not None else 0.0
        vx = float(st.vx) if st.vx is not None else 0.0
        vy = float(st.vy) if st.vy is not None else 0.0
        ball_y = float(st.ball_y)
        ball_x = float(st.ball_x)
        paddle_x = float(st.paddle_x)

        edge_dist = min(ball_x, (self._W - 1.0) - ball_x)  # distance to nearest vertical wall
        edge_dist_n = float(edge_dist / max(1.0, self._W))

        return (
            np.array(
                [
                    dx_now / max(1.0, self._W),
                    vx / max(1.0, self._W),
                    vy / max(1.0, self._H),
                    ball_y / max(1.0, self._H),
                    ball_x / max(1.0, self._W),
                    paddle_x / max(1.0, self._W),
                    edge_dist_n,
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
# Residual tunnel control (owns tracker updates + shaping)
# ============================================================
class ResidualTunnelControlWrapper(ActionWrapper):
    """
    PPO outputs u in [-1,1] -> delta_x = u * delta_x_max.
    Pre-impact: if ball descending + near paddle and already aligned (|dx| <= gate_px),
    residual chooses to offset the contact point to steer the rebound.

    NEW tunnel shaping (dense, and aligned with real score):
      - Above-bricks shaping:
          +k_above each step ball is in the upper region (ball_y <= y_brick)
      - Side-edge shaping (only when above-bricks):
          +k_edge * edge_score, where edge_score increases as ball gets closer to wall
        This biases trajectories that climb and hug a side = tunnels.
    """

    NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3

    def __init__(
        self,
        env: gym.Env,
        base_policy: Callable[[Any], int],
        tracker: BreakoutRAMTracker,
        *,
        # Control gating
        y_min_for_residual: float = 165.0,
        gate_px: float = 20.0,
        delta_x_max: float = 6.0,
        move_deadband_px: float = 2.0,
        cooldown_steps: int = 8,
        cooldown_on_life_loss: bool = True,
        # Serve handling
        serve_steps: int = 10,
        stuck_steps: int = 120,
        # Hit detection
        paddle_y_px: float = 190.0,
        hit_y_band: float = 14.0,
        # Post-hit shaping (optional)
        posthit_window: int = 35,
        k_post_vx: float = 0.01,
        k_post_up: float = 0.02,
        # Tunnel shaping (dense)
        y_brick: float = 60.0,     # IMPORTANT: tune to your RAM scale (see notes)
        edge_band: float = 12.0,   # pixels from side considered "edge-ish"
        k_above: float = 0.02,     # per-step above-bricks reward
        k_edge: float = 0.05,      # additional per-step edge reward (only when above)
        # Effort penalty
        k_u: float = 0.0007,
        debug: bool = True,
    ):
        super().__init__(env)
        if not isinstance(env.action_space, Discrete) or int(env.action_space.n) < 4:
            raise TypeError("Expected Breakout-like Discrete action space with >=4 actions.")

        self._base = base_policy
        self._tracker = tracker

        self._W = 160.0
        self._H = 210.0

        self._y_min_for_residual = float(y_min_for_residual)
        self._gate_px = float(gate_px)
        self._delta_x_max = float(delta_x_max)
        self._move_deadband = float(move_deadband_px)

        self._cooldown_steps = int(cooldown_steps)
        self._cooldown_on_life_loss = bool(cooldown_on_life_loss)
        self._cooldown_left = 0

        self._serve_steps = int(serve_steps)
        self._serve_left = 0
        self._stuck_steps = int(stuck_steps)
        self._no_motion_left = self._stuck_steps
        self._prev_ball_pos: tuple[float, float] | None = None

        self._paddle_y = float(paddle_y_px)
        self._hit_y_band = float(hit_y_band)

        self._posthit_window = int(posthit_window)
        self._posthit_left = 0
        self._k_post_vx = float(k_post_vx)
        self._k_post_up = float(k_post_up)

        # tunnel shaping
        self._y_brick = float(y_brick)
        self._edge_band = float(edge_band)
        self._k_above = float(k_above)
        self._k_edge = float(k_edge)

        self._k_u = float(k_u)
        self._debug = bool(debug)

        # scalar residual action
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._last_obs_raw: Any | None = None
        self._prev_lives: int | None = None

        # debug counters
        self._dbg_total = 0
        self._dbg_used = 0
        self._dbg_hit = 0
        self._dbg_above = 0
        self._dbg_edge = 0

    def reset(self, **kwargs):
        self._tracker.reset()
        obs, info = self.env.reset(**kwargs)
        self._last_obs_raw = obs

        # prime tracker once
        _ = self._tracker.update(obs)

        self._cooldown_left = self._cooldown_steps
        self._posthit_left = 0

        lives = info.get("lives", None)
        self._prev_lives = int(lives) if lives is not None else None

        # serve mode
        self._serve_left = self._serve_steps

        # stuck detector
        self._no_motion_left = self._stuck_steps
        self._prev_ball_pos = None

        # reset base expert if it supports it
        if hasattr(self._base, "reset") and callable(getattr(self._base, "reset")):
            self._base.reset()  # type: ignore[attr-defined]

        # debug
        self._dbg_total = self._dbg_used = self._dbg_hit = self._dbg_above = self._dbg_edge = 0

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

    def _tunnel_shaping(self, st: TrackState) -> tuple[float, bool, bool, float]:
        """
        Returns (bonus, above, edge, edge_score).
        edge_score in [0,1], larger when closer to wall.
        """
        if (not st.ok) or st.ball_x is None or st.ball_y is None:
            return 0.0, False, False, 0.0

        ball_x = float(st.ball_x)
        ball_y = float(st.ball_y)

        above = ball_y <= self._y_brick
        if not above:
            return 0.0, False, False, 0.0

        # Above bricks bonus
        bonus = self._k_above

        # Edge bonus: only when above bricks
        dist_to_wall = min(ball_x, (self._W - 1.0) - ball_x)
        edge = dist_to_wall <= self._edge_band

        # Smooth score: 1 when at wall, 0 when outside edge_band
        edge_score = float(np.clip((self._edge_band - dist_to_wall) / max(1e-6, self._edge_band), 0.0, 1.0))
        if edge_score > 0.0:
            bonus += self._k_edge * edge_score

        return float(bonus), True, bool(edge), float(edge_score)

    def step(self, action: np.ndarray) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        if self._last_obs_raw is None:
            obs, info = self.reset()
            self._last_obs_raw = obs

        u = float(np.asarray(action, dtype=np.float32).reshape(1)[0])
        u = float(np.clip(u, -1.0, 1.0))
        delta_x = u * self._delta_x_max

        in_cooldown = self._cooldown_left > 0
        if in_cooldown:
            self._cooldown_left -= 1

        # state from last obs
        st = self._tracker.last

        # stuck detector
        self._check_stuck(st)

        # base action (expert)
        base_a = int(self._base(self._last_obs_raw))
        base_a = int(np.clip(base_a, 0, int(self.env.action_space.n) - 1))

        used_residual = False

        # Serve mode
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

                preimpact_ok = (vy > 0.0) and (ball_y >= self._y_min_for_residual) and (abs(dx_now) <= self._gate_px)
                if preimpact_ok:
                    # target contact point: ball_x + delta_x
                    x_target = float(np.clip(float(st.ball_x) + delta_x, 0.0, self._W - 1.0))
                    env_a = self._move_toward(x_target, float(st.paddle_x))
                    used_residual = True

        # step env
        obs_next, rew_raw, term, trunc, env_info = self.env.step(int(env_a))
        self._last_obs_raw = obs_next

        # update tracker ONCE with next obs
        st_next = self._tracker.update(obs_next)

        # start reward from raw
        rew_total = float(rew_raw)

        # tunnel shaping (dense)
        tunnel_bonus, above, edge, edge_score = self._tunnel_shaping(st_next)
        rew_total += float(tunnel_bonus)

        # hit detection: vy flips + -> - near paddle
        hit_detected = False
        if st.ok and st.vy is not None and st_next.ok and st_next.vy is not None and st_next.ball_y is not None:
            vy_prev = float(st.vy)
            vy_now = float(st_next.vy)
            near_paddle = abs(float(st_next.ball_y) - self._paddle_y) <= self._hit_y_band
            if (vy_prev > 0.0) and (vy_now < 0.0) and near_paddle:
                hit_detected = True
                self._posthit_left = self._posthit_window

        # post-hit shaping (very light): reward upward motion and some vx
        post_vx_term = None
        post_up_term = None
        if self._posthit_left > 0 and st_next.ok:
            vx_term = 0.0
            up_term = 0.0

            if st_next.vx is not None:
                vx_term = float(np.clip(abs(float(st_next.vx)) / max(1.0, self._W), 0.0, 1.0))
                rew_total += self._k_post_vx * vx_term

            if st_next.vy is not None:
                # after hit we want vy negative (going up)
                up_term = float(np.clip(-float(st_next.vy) / 6.0, 0.0, 1.0))
                rew_total += self._k_post_up * up_term

            post_vx_term = vx_term
            post_up_term = up_term
            self._posthit_left -= 1

        # effort penalty (only when residual used)
        if used_residual:
            rew_total -= self._k_u * abs(u)

        # life-loss handling: trigger serve mode again
        lives = env_info.get("lives", None) if isinstance(env_info, dict) else None
        if lives is not None:
            lives_i = int(lives)
            if self._prev_lives is not None and lives_i < self._prev_lives:
                self._serve_left = self._serve_steps
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

        # debug
        if self._debug:
            self._dbg_total += 1
            if used_residual:
                self._dbg_used += 1
            if hit_detected:
                self._dbg_hit += 1
            if above:
                self._dbg_above += 1
            if edge:
                self._dbg_edge += 1

        info: dict[str, Any] = {
            "reward_raw": float(rew_raw),
            "reward_total": float(rew_total),
            "used_residual": bool(used_residual),
            "hit_detected": bool(hit_detected),
            "serve_left": int(self._serve_left),
            "in_cooldown": bool(in_cooldown),
            "cooldown_left": int(self._cooldown_left),
            "u": float(u),
            "delta_x": float(delta_x),
            "base_action": int(base_a),
            "env_action": int(env_a),
            "ram_ok": bool(st_next.ok),
            "ball_x": float(st_next.ball_x) if st_next.ball_x is not None else float("nan"),
            "ball_y": float(st_next.ball_y) if st_next.ball_y is not None else float("nan"),
            "above_bricks": bool(above),
            "edge_score": float(edge_score),
            "tunnel_bonus": float(tunnel_bonus),
            "used_rate_dbg": (self._dbg_used / max(1, self._dbg_total)),
            "hit_rate_dbg": (self._dbg_hit / max(1, self._dbg_total)),
            "above_rate_dbg": (self._dbg_above / max(1, self._dbg_total)),
            "edge_rate_dbg": (self._dbg_edge / max(1, self._dbg_total)),
            "lives": int(self._prev_lives) if self._prev_lives is not None else None,
        }
        if post_vx_term is not None:
            info["post_vx_term"] = float(post_vx_term)
        if post_up_term is not None:
            info["post_up_term"] = float(post_up_term)

        if isinstance(env_info, dict):
            info.update(env_info)

        return obs_next, float(rew_total), bool(term), bool(trunc), info
