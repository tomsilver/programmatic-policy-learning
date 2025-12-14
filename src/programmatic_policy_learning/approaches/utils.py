"""Utils for Testing LPP Approach."""

from typing import Any

from programmatic_policy_learning.policies.lpp_policy import LPPPolicy


def run_single_episode(
    env: Any,
    policy: LPPPolicy,
    record_video: bool = False,
    video_out_path: str | None = None,
    max_num_steps: int = 100,
) -> float:
    """Run a single episode in the environment using the given policy."""

    if record_video:
        env.start_recording_video(video_out_path=video_out_path)

    obs, _ = env.reset()
    total_reward = 0.0
    for _ in range(max_num_steps):
        action = policy(obs)
        new_obs, reward, done, _, _ = env.step(action)
        total_reward += reward

        obs = new_obs

        if done:
            break
    env.close()

    return total_reward
