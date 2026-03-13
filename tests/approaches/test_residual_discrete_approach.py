from __future__ import annotations

from collections import deque
from typing import Any, Callable

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from programmatic_policy_learning.approaches.residual_discrete_approach import (
    BreakoutRAMTracker,
    BreakoutHitPointFeatureObsWrapper,
    ResidualHitPointControlWrapper,
)

ExpertPolicy = Callable[[Any], int]


def make_breakout_expert_ram(*, serve_steps: int = 10, deadband_px: float = 2.0) -> ExpertPolicy:
    """Simple RAM expert: serves, then tracks ball_x."""
    NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3
    steps = 0
    tracker = BreakoutRAMTracker()

    def reset() -> None:
        nonlocal steps
        steps = 0
        tracker.reset()

    def expert(obs_ram: Any) -> int:
        nonlocal steps
        steps += 1
        if steps <= serve_steps:
            return FIRE

        st = tracker.update(np.asarray(obs_ram))
        if not st.ok or st.paddle_x is None or st.ball_x is None:
            return NOOP

        dx = float(st.ball_x - st.paddle_x)
        if dx > deadband_px:
            return RIGHT
        if dx < -deadband_px:
            return LEFT
        return NOOP

    expert.reset = reset  # type: ignore[attr-defined]
    return expert


class TrainDebugCallback(BaseCallback):
    def __init__(self, print_every: int = 2000, window: int = 2000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.print_every = int(print_every)
        self.window = int(window)

        self._used = deque(maxlen=self.window)
        self._hit = deque(maxlen=self.window)
        self._raw = deque(maxlen=self.window)
        self._tot = deque(maxlen=self.window)
        self._ytr = deque(maxlen=self.window)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if not infos or not isinstance(infos, (list, tuple)) or not isinstance(infos[0], dict):
            return True
        info0 = infos[0]

        self._used.append(1.0 if info0.get("used_residual", False) else 0.0)
        self._hit.append(1.0 if info0.get("hit_detected", False) else 0.0)
        self._raw.append(float(info0.get("reward_raw", 0.0)))
        self._tot.append(float(info0.get("reward_total", 0.0)))

        ytr = info0.get("y_trigger", None)
        if ytr is not None:
            self._ytr.append(float(ytr))

        if self.num_timesteps % self.print_every == 0:
            used_rate = float(np.mean(self._used)) if self._used else 0.0
            hit_rate = float(np.mean(self._hit)) if self._hit else 0.0
            mean_raw = float(np.mean(self._raw)) if self._raw else 0.0
            mean_tot = float(np.mean(self._tot)) if self._tot else 0.0
            mean_ytr = float(np.mean(self._ytr)) if self._ytr else float("nan")

            print(
                f"[train] t={self.num_timesteps} used_rate={used_rate:.3f} hit_rate={hit_rate:.3f} "
                f"y_trigger={mean_ytr:.2f} mean_raw_rew={mean_raw:.4f} mean_total_rew={mean_tot:.4f}"
            )
        return True


def make_env(env_id: str, *, seed: int, render_mode: str | None = None) -> gym.Env:
    import ale_py

    gym.register_envs(ale_py)

    env = gym.make(env_id, obs_type="ram", render_mode=render_mode)
    env.reset(seed=seed)

    tracker = BreakoutRAMTracker()
    base = make_breakout_expert_ram(serve_steps=10, deadband_px=2.0)

    def base_np(obs_any: Any) -> int:
        return int(base(np.asarray(obs_any)))

    env_ctrl = ResidualHitPointControlWrapper(
        env,
        base_policy=base_np,
        tracker=tracker,
        # You can tune these, but these defaults are safer than the "aggressive" version
        gate_px=20.0,
        delta_x_max=10.0,
        cooldown_steps=8,
        posthit_window=45,
        k_post_y=0.10,
        k_post_vx=0.03,
        k_u=0.0007,
        trigger_frac=0.80,
        trigger_margin=0.0,
        serve_steps=10,
        debug=True,
    )

    env_feat = BreakoutHitPointFeatureObsWrapper(env_ctrl, tracker)
    return env_feat


def evaluate_and_gif(
    env_id: str,
    model: PPO,
    *,
    seed: int,
    num_episodes: int = 10,
    max_steps: int = 50_000,
    save_gif: bool = True,
    gif_path: str = "hitpoint_residual_ram_ep0.gif",
    gif_episode_index: int = 0,
    stride: int = 4,
    fps: int = 30,
) -> None:
    frames: list[np.ndarray] = []
    raw_returns: list[float] = []
    total_returns: list[float] = []

    for ep in range(num_episodes):
        env = make_env(
            env_id,
            seed=seed + ep,
            render_mode="rgb_array" if (save_gif and ep == gif_episode_index) else None,
        )
        obs, info = env.reset(seed=seed + ep)

        raw_ret = 0.0
        tot_ret = 0.0

        for t in range(max_steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)

            tot_ret += float(r)
            raw_ret += float(info.get("reward_raw", 0.0))

            if save_gif and ep == gif_episode_index and (t % stride == 0):
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            if term or trunc:
                break

        env.close()
        raw_returns.append(raw_ret)
        total_returns.append(tot_ret)
        print(f"Eval ep {ep+1}/{num_episodes}: raw_return={raw_ret:.1f} total_return={tot_ret:.2f}")

    if save_gif and frames:
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"Saved GIF to {gif_path} ({len(frames)} frames, fps={fps})")

    print("\nEval summary:")
    print(f"  Mean raw return:   {np.mean(raw_returns):.2f}")
    print(f"  Mean total return: {np.mean(total_returns):.2f}")


def main() -> None:
    env_id = "ALE/Breakout-v5"
    seed = 0

    total_timesteps = 100_000
    print_every = 2000

    vec_env = DummyVecEnv([lambda: make_env(env_id, seed=seed, render_mode=None)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=seed,
    )

    cb = TrainDebugCallback(print_every=print_every, window=print_every)
    print("Training hit-point residual (adaptive RAM y trigger)...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=cb)

    model.save("ppo_breakout_hitpoint_residual_ram")

    evaluate_and_gif(
        env_id,
        model,
        seed=seed + 123,
        num_episodes=10,
        save_gif=True,
        gif_path="hitpoint_residual_ram_ep0.gif",
        gif_episode_index=0,
        stride=4,
        fps=30,
    )

    vec_env.close()


if __name__ == "__main__":
    main()
