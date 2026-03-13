"""
Expert policies for LunarLanderContinuous-v3 (heuristic, not optimal).

This script contains:
1) A simple hand-written heuristic (callable policy factory) used as an "expert prior".
   - Threshold-based vertical control (vy) with a near-ground safety gate.
   - PD side control on (angle, ang_vel) plus damping on (x, vx).
   - Optional params dict for lightweight tuning / sweeps.

2) A ParametricPolicyBase version (tunable) of the SAME heuristic structure.

Obs: [x, y, vx, vy, angle, ang_vel, leg_l, leg_r]
Act: [main_throttle, side_throttle] in [-1, 1]
  - In practice, main_throttle < 0 is effectively "off", so we clamp main to [0, 1].
"""

from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from programmatic_policy_learning.approaches.structs import ParametricPolicyBase

Obs = NDArray[np.float32]
Act = NDArray[np.float32]


# ---------------------------------------------------------------------
# Simple hand-written heuristic (callable policy)
# ---------------------------------------------------------------------
def create_manual_lunarlander_continuous_policy(
    action_space: gym.spaces.Box,
    params: dict[str, float] | None = None,
) -> Callable[[Obs], Act]:
    """
    Create a heuristic policy for LunarLanderContinuous-v3.

    Params supported (all optional):
      Vertical:
        vy_hard, vy_soft, main_hard, main_soft,
        y_ground, vy_ground, main_ground
      Side:
        k_ang, k_w, k_x, k_vx,
        y_side_reduce, side_reduce
      Contact:
        contact_side_scale, contact_main_scale

    This matches the sweepable "baseline" family you were tuning.
    """
    assert isinstance(action_space, gym.spaces.Box)
    assert action_space.shape == (2,)

    low = action_space.low.astype(np.float32)
    high = action_space.high.astype(np.float32)

    p = params or {}

    # ---- defaults (use your best-known tuned baseline as starting point) ----
    # (These are the values from your eval_best JSON.)
    VY_HARD = float(p.get("vy_hard", -0.85))
    VY_SOFT = float(p.get("vy_soft", -0.55))
    MAIN_HARD = float(p.get("main_hard", 1.0))
    MAIN_SOFT = float(p.get("main_soft", 0.6))
    Y_GROUND = float(p.get("y_ground", 0.45))
    VY_GROUND = float(p.get("vy_ground", -0.22))
    MAIN_GROUND = float(p.get("main_ground", 0.9))

    K_ANG = float(p.get("k_ang", 1.6))
    K_W = float(p.get("k_w", 0.4))
    K_X = float(p.get("k_x", 0.3))
    K_VX = float(p.get("k_vx", 0.7))
    Y_SIDE_REDUCE = float(p.get("y_side_reduce", 0.35))
    SIDE_REDUCE = float(p.get("side_reduce", 0.4))

    # Contact dampening
    CONTACT_SIDE_SCALE = float(p.get("contact_side_scale", 0.3))
    CONTACT_MAIN_SCALE = float(p.get("contact_main_scale", 0.5))

    def policy(obs: Obs) -> Act:
        o = np.asarray(obs, dtype=np.float32)
        x, y, vx, vy, ang, ang_vel, leg_l, leg_r = [float(v) for v in o]

        # -----------------------------
        # Vertical control (main engine)
        # -----------------------------
        main = 0.0
        if vy < VY_HARD:
            main = MAIN_HARD
        elif vy < VY_SOFT:
            main = MAIN_SOFT

        # Near-ground safety gate
        if (y < Y_GROUND) and (vy < VY_GROUND):
            main = max(main, MAIN_GROUND)

        # IMPORTANT: main thrust effectively only works for >= 0
        main = float(np.clip(main, 0.0, 1.0))

        # -----------------------------
        # Horizontal / attitude control (side engine)
        # -----------------------------
        # PD to keep upright + damp lateral drift
        side = -K_ANG * ang - K_W * ang_vel
        side += -K_X * x - K_VX * vx

        # Reduce steering close to ground (prevents twitch / coupling)
        if y < Y_SIDE_REDUCE:
            side *= SIDE_REDUCE

        # -----------------------------
        # Contact dampening
        # -----------------------------
        if (leg_l > 0.5) or (leg_r > 0.5):
            side *= CONTACT_SIDE_SCALE
            main *= CONTACT_MAIN_SCALE

        side = float(np.clip(side, -1.0, 1.0))
        main = float(np.clip(main, 0.0, 1.0))

        act = np.array([main, side], dtype=np.float32)
        act = np.clip(act, low, high).astype(action_space.dtype)
        return act

    return policy


# ---------------------------------------------------------------------
# Parametric version (tunable heuristic) - MATCHES the simple heuristic above
# ---------------------------------------------------------------------
class LunarLanderContinuousParametricPolicy(ParametricPolicyBase):
    """
    A tunable heuristic "expert" for LunarLanderContinuous-v3.

    This parametric policy matches the SAME structure as
    create_manual_lunarlander_continuous_policy (threshold vertical + PD side).

    Parameters:
      Vertical:
        vy_hard, vy_soft, main_hard, main_soft,
        y_ground, vy_ground, main_ground
      Side:
        k_ang, k_w, k_x, k_vx,
        y_side_reduce, side_reduce
      Contact:
        contact_side_scale, contact_main_scale
    """

    def __init__(
        self,
        _env_description: Any | None = None,
        _observation_space: gym.spaces.Space | None = None,
        _action_space: gym.spaces.Box | None = None,
        _seed: int | None = None,
        *,
        init_params: dict[str, float] | None = None,
        param_bounds: dict[str, tuple[float, float]] | None = None,
        **_kwargs: dict[str, Any],
    ) -> None:
        if init_params is None:
            # Start from your current best-known config.
            init_params = {
                "vy_hard": -0.85,
                "vy_soft": -0.55,
                "main_hard": 1.0,
                "main_soft": 0.6,
                "y_ground": 0.45,
                "vy_ground": -0.22,
                "main_ground": 0.9,
                "k_ang": 1.6,
                "k_w": 0.4,
                "k_x": 0.3,
                "k_vx": 0.7,
                "y_side_reduce": 0.35,
                "side_reduce": 0.4,
                "contact_side_scale": 0.3,
                "contact_main_scale": 0.5,
            }

        if param_bounds is None:
            # Bounds chosen to keep the heuristic stable while allowing useful tuning.
            param_bounds = {
                # Vertical thresholds/gains
                "vy_hard": (-1.5, -0.2),
                "vy_soft": (-1.2, -0.05),
                "main_hard": (0.2, 1.0),
                "main_soft": (0.0, 1.0),
                "y_ground": (0.1, 1.2),
                "vy_ground": (-0.6, -0.02),
                "main_ground": (0.0, 1.0),
                # Side PD / damping
                "k_ang": (0.0, 4.0),
                "k_w": (0.0, 2.0),
                "k_x": (0.0, 1.0),
                "k_vx": (0.0, 1.5),
                "y_side_reduce": (0.05, 1.0),
                "side_reduce": (0.0, 1.0),
                # Contact dampening
                "contact_side_scale": (0.0, 1.0),
                "contact_main_scale": (0.0, 1.0),
            }

        super().__init__(init_params=init_params, param_bounds=param_bounds)
        self._action_space = _action_space

    def act(self, obs: Any) -> Any:
        assert isinstance(obs, np.ndarray)
        o = np.asarray(obs, dtype=np.float32)
        x, y, vx, vy, ang, ang_vel, leg_l, leg_r = [float(v) for v in o]

        # Extract params
        vy_hard = float(self._params["vy_hard"])
        vy_soft = float(self._params["vy_soft"])
        main_hard = float(self._params["main_hard"])
        main_soft = float(self._params["main_soft"])
        y_ground = float(self._params["y_ground"])
        vy_ground = float(self._params["vy_ground"])
        main_ground = float(self._params["main_ground"])

        k_ang = float(self._params["k_ang"])
        k_w = float(self._params["k_w"])
        k_x = float(self._params["k_x"])
        k_vx = float(self._params["k_vx"])
        y_side_reduce = float(self._params["y_side_reduce"])
        side_reduce = float(self._params["side_reduce"])

        contact_side_scale = float(self._params["contact_side_scale"])
        contact_main_scale = float(self._params["contact_main_scale"])

        # -----------------------------
        # Vertical control
        # -----------------------------
        main = 0.0
        if vy < vy_hard:
            main = main_hard
        elif vy < vy_soft:
            main = main_soft

        if (y < y_ground) and (vy < vy_ground):
            main = max(main, main_ground)

        main = float(np.clip(main, 0.0, 1.0))

        # -----------------------------
        # Side control
        # -----------------------------
        side = -k_ang * ang - k_w * ang_vel
        side += -k_x * x - k_vx * vx

        if y < y_side_reduce:
            side *= side_reduce

        # Contact dampening
        if (leg_l > 0.5) or (leg_r > 0.5):
            side *= contact_side_scale
            main *= contact_main_scale

        main = float(np.clip(main, 0.0, 1.0))
        side = float(np.clip(side, -1.0, 1.0))

        act = np.array([main, side], dtype=np.float32)

        # Clip to action space if provided; otherwise default
        if isinstance(self._action_space, gym.spaces.Box):
            low = self._action_space.low.astype(np.float32)
            high = self._action_space.high.astype(np.float32)
            act = np.clip(act, low, high).astype(self._action_space.dtype)
        else:
            act = np.clip(act, [0.0, -1.0], [1.0, 1.0]).astype(np.float32)

        return act