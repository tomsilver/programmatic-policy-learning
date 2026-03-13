# """
# Make two Pendulum GIFs for a presentation:

#   1) baseline expert only ("stupid approach" / no residual)
#   2) expert + trained residual (TD3 residual)

# Important: your sweep script saved only CSVs, not model checkpoints.
# CSV logs are not enough to render GIFs. This script trains once (expert variant),
# saves the trained residual model, then renders both GIFs.

# Outputs (in logs/ by default):
#   - logs/pendulum_expert_baseline.gif
#   - logs/pendulum_expert_plus_residual.gif
#   - logs/pendulum_expert_residual_td3_model.zip   (checkpoint)
# """

# from __future__ import annotations

# import os
# from pathlib import Path
# from typing import Callable, List

# import gymnasium as gym
# import imageio.v2 as imageio
# import numpy as np
# from gymnasium.spaces import Box

# from programmatic_policy_learning.approaches.experts.pendulum_experts import (
#     PendulumParametricPolicy,
# )
# from programmatic_policy_learning.approaches.residual_approach import ResidualApproach

# # -----------------------------
# # Config
# # -----------------------------
# ENV_ID = "Pendulum-v1"

# # Use one seed for clean, repeatable GIFs
# RUN_SEED = 0

# # Match your sweep defaults
# NUM_TRAIN_PHASES = 20
# TRAIN_STEPS_PER_PHASE = 500
# TOTAL_TRAIN_STEPS = NUM_TRAIN_PHASES * TRAIN_STEPS_PER_PHASE  # 10,000

# MAX_STEPS = 200  # Pendulum-v1 episode length is 200
# FPS = 30

# OUT_DIR = Path("logs")
# OUT_DIR.mkdir(parents=True, exist_ok=True)

# BASELINE_GIF = OUT_DIR / "pendulum_expert_baseline.gif"
# TRAINED_GIF = OUT_DIR / "pendulum_expert_plus_residual.gif"
# MODEL_PATH = OUT_DIR / "pendulum_expert_residual_td3_model.zip"

# # If True, re-train even if MODEL_PATH exists
# FORCE_RETRAIN = False


# def make_env(seed: int | None = None, render: bool = False) -> gym.Env:
#     env = gym.make(ENV_ID, render_mode="rgb_array" if render else None)
#     if seed is not None:
#         env.reset(seed=seed)
#     return env


# def build_pendulum_expert(action_space: Box) -> PendulumParametricPolicy:
#     return PendulumParametricPolicy(
#         _env_description=None,
#         _observation_space=None,
#         _action_space=action_space,
#         _seed=None,
#         init_params={"kp": 12.0, "kd": 3.0},
#         param_bounds={"kp": (-50.0, 50.0), "kd": (0.0, 20.0)},
#         min_torque=float(action_space.low[0]),
#         max_torque=float(action_space.high[0]),
#     )


# def build_residual(seed: int) -> ResidualApproach:
#     # Template env for spaces
#     env = make_env(seed=seed, render=False)
#     obs_space = env.observation_space
#     act_space = env.action_space
#     assert isinstance(act_space, Box)

#     expert = build_pendulum_expert(act_space)

#     def env_factory(instance_num: int) -> gym.Env:
#         return make_env(seed=seed + instance_num, render=False)

#     residual = ResidualApproach(
#         environment_description="Gymnasium Pendulum-v1",
#         observation_space=obs_space,
#         action_space=act_space,
#         seed=seed,
#         expert=expert,
#         env_factory=env_factory,
#         backend="sb3-td3",
#         total_timesteps=TOTAL_TRAIN_STEPS,
#         lr=1e-3,
#         noise_std=0.1,
#         verbose=1,
#         train_before_eval=False,  # we will call train() explicitly
#         train_env_instance=0,
#     )

#     env.close()
#     return residual


# def record_gif_with_policy(
#     policy_fn: Callable[[np.ndarray], np.ndarray],
#     out_path: Path,
#     seed: int,
#     *,
#     max_steps: int = MAX_STEPS,
#     fps: int = FPS,
# ) -> None:
#     env = make_env(seed=seed, render=True)
#     obs, _info = env.reset(seed=seed)

#     frames: List[np.ndarray] = []
#     for _ in range(max_steps):
#         frame = env.render()
#         if frame is not None:
#             frames.append(frame)

#         act = policy_fn(np.asarray(obs, dtype=np.float32))
#         obs, _r, term, trunc, _info = env.step(act)
#         if term or trunc:
#             break

#     env.close()

#     imageio.mimsave(str(out_path), frames, fps=fps)
#     print(f"[gif] wrote {out_path.resolve()}")


# def main() -> None:
#     # -----------------------------
#     # 1) Baseline expert-only GIF
#     # -----------------------------
#     tmp_env = make_env(seed=RUN_SEED, render=False)
#     assert isinstance(tmp_env.action_space, Box)
#     act_space: Box = tmp_env.action_space
#     expert = build_pendulum_expert(act_space)
#     tmp_env.close()

#     def expert_only_policy(obs: np.ndarray) -> np.ndarray:
#         # PendulumParametricPolicy has .act(obs)
#         a = expert.act(np.asarray(obs, dtype=np.float32))
#         # Ensure correct dtype/shape
#         return np.asarray(a, dtype=np.float32)

#     record_gif_with_policy(expert_only_policy, BASELINE_GIF, seed=RUN_SEED)

#     # -----------------------------
#     # 2) Trained residual GIF
#     # -----------------------------
#     residual = build_residual(seed=RUN_SEED)

#     if MODEL_PATH.exists() and not FORCE_RETRAIN:
#         print(f"[model] loading existing residual model: {MODEL_PATH}")
#         residual.load(str(MODEL_PATH))
#     else:
#         print(f"[train] training TD3 residual for {TOTAL_TRAIN_STEPS} timesteps...")
#         residual.train()
#         print(f"[model] saving residual model: {MODEL_PATH}")
#         residual.save(str(MODEL_PATH))

#     # Render using the public ResidualApproach API (reset/step/update)
#     def residual_policy(obs0: np.ndarray) -> np.ndarray:
#         # We drive via ResidualApproach methods to match your evaluation path.
#         # We'll maintain internal state by calling reset once per episode below.
#         raise RuntimeError("This should not be called directly.")

#     # Record an episode while stepping residual approach properly
#     env = make_env(seed=RUN_SEED, render=True)
#     obs, info = env.reset(seed=RUN_SEED)
#     residual.reset(np.asarray(obs, dtype=np.float32), info)

#     frames: List[np.ndarray] = []
#     for _ in range(MAX_STEPS):
#         frame = env.render()
#         if frame is not None:
#             frames.append(frame)

#         action = residual.step()
#         obs, rew, term, trunc, info = env.step(action)
#         residual.update(obs, float(rew), bool(term or trunc), info)

#         if term or trunc:
#             break

#     env.close()
#     imageio.mimsave(str(TRAINED_GIF), frames, fps=FPS)
#     print(f"[gif] wrote {TRAINED_GIF.resolve()}")


# if __name__ == "__main__":
#     main()

"""
Make two LunarLanderContinuous GIFs for a presentation:

  1) baseline expert only (no residual)
  2) expert + trained residual (TD3 residual)

Because sweep scripts typically save only CSVs (no checkpoints),
this script trains once (or loads a saved model) then renders both GIFs.

Outputs (in logs/ by default):
  - logs/lander_expert_baseline.gif
  - logs/lander_expert_plus_residual.gif
  - logs/lander_expert_residual_td3_model.zip
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from gymnasium.spaces import Box

from programmatic_policy_learning.approaches.residual_approach import ResidualApproach

# Your handwritten expert constructor (adjust import path if needed)
from programmatic_policy_learning.approaches.experts.lundar_lander_experts import (
    create_manual_lunarlander_continuous_policy,
)

# -----------------------------
# Config
# -----------------------------
ENV_ID = "LunarLanderContinuous-v3"

RUN_SEED = 0

# Match your sweep-ish defaults (tune if you want)
NUM_TRAIN_PHASES = 13
TRAIN_STEPS_PER_PHASE = 10_000
TOTAL_TRAIN_STEPS = NUM_TRAIN_PHASES * TRAIN_STEPS_PER_PHASE

MAX_STEPS = 1_000
FPS = 30

OUT_DIR = Path("logs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_GIF = OUT_DIR / "lander_expert_baseline.gif"
TRAINED_GIF = OUT_DIR / "lander_expert_plus_residual.gif"
MODEL_PATH = OUT_DIR / "lander_expert_residual_td3_model.zip"

FORCE_RETRAIN = False


def make_env(seed: int | None = None, render: bool = False) -> gym.Env:
    env = gym.make(ENV_ID, render_mode="rgb_array" if render else None)
    if seed is not None:
        env.reset(seed=seed)
    return env


def build_residual(seed: int) -> ResidualApproach:
    env = make_env(seed=seed, render=False)
    obs_space = env.observation_space
    act_space = env.action_space
    assert isinstance(act_space, Box), "LunarLanderContinuous must have a Box action space."

    # Base expert policy callable(obs)->act
    base_policy_fn = create_manual_lunarlander_continuous_policy(act_space)

    class _ExpertWrapper:
        # ResidualApproach will pick up .act(...)
        def act(self, obs):
            return np.asarray(base_policy_fn(np.asarray(obs, dtype=np.float32)), dtype=np.float32)

    expert = _ExpertWrapper()

    def env_factory(instance_num: int) -> gym.Env:
        return make_env(seed=seed + instance_num, render=False)

    residual = ResidualApproach(
        environment_description=f"Gymnasium {ENV_ID}",
        observation_space=obs_space,
        action_space=act_space,
        seed=seed,
        expert=expert,
        env_factory=env_factory,
        backend="sb3-td3",
        total_timesteps=TOTAL_TRAIN_STEPS,
        lr=1e-3,
        noise_std=0.1,
        verbose=1,
        train_before_eval=False,
        train_env_instance=0,
    )

    env.close()
    return residual, base_policy_fn


def record_gif_expert_only(
    base_policy_fn,
    out_path: Path,
    seed: int,
) -> None:
    env = make_env(seed=seed, render=True)
    obs, _info = env.reset(seed=seed)

    frames: List[np.ndarray] = []
    for _ in range(MAX_STEPS):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        act = np.asarray(base_policy_fn(np.asarray(obs, dtype=np.float32)), dtype=np.float32)
        act = np.clip(act, -1.0, 1.0).astype(np.float32)

        obs, _r, term, trunc, _info = env.step(act)
        if term or trunc:
            break

    env.close()
    imageio.mimsave(str(out_path), frames, fps=FPS)
    print(f"[gif] wrote {out_path.resolve()}")


def record_gif_residual(
    residual: ResidualApproach,
    out_path: Path,
    seed: int,
) -> None:
    env = make_env(seed=seed, render=True)
    obs, info = env.reset(seed=seed)
    residual.reset(np.asarray(obs, dtype=np.float32), info)

    frames: List[np.ndarray] = []
    for _ in range(MAX_STEPS):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action = residual.step()
        obs, rew, term, trunc, info = env.step(action)
        residual.update(obs, float(rew), bool(term or trunc), info)

        if term or trunc:
            break

    env.close()
    imageio.mimsave(str(out_path), frames, fps=FPS)
    print(f"[gif] wrote {out_path.resolve()}")


def main() -> None:
    residual, base_policy_fn = build_residual(seed=RUN_SEED)

    # 1) Baseline expert-only GIF
    record_gif_expert_only(base_policy_fn, BASELINE_GIF, seed=RUN_SEED)

    # 2) Train/load residual then GIF
    if MODEL_PATH.exists() and not FORCE_RETRAIN:
        print(f"[model] loading existing residual model: {MODEL_PATH}")
        residual.load(str(MODEL_PATH))
    else:
        print(f"[train] training TD3 residual for {TOTAL_TRAIN_STEPS} timesteps...")
        residual.train()
        print(f"[model] saving residual model: {MODEL_PATH}")
        residual.save(str(MODEL_PATH))

    record_gif_residual(residual, TRAINED_GIF, seed=RUN_SEED)


if __name__ == "__main__":
    main()