"""Tests for llm_ppl_approach.py with Motion2D environment."""

import tempfile
from pathlib import Path
from typing import Any

import kinder
import numpy as np
import pytest
from gymnasium.envs.registration import register, registry
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel, OrderedResponseModel
from prpl_llm_utils.structs import Response

from programmatic_policy_learning.approaches.llm_ppl_approach import LLMPPLApproach

runllms = pytest.mark.skipif("not config.getoption('runllms')")

_MOTION2D_P1 = "kinder/Motion2D-p1-v0"
_MOTION2D_P3 = "kinder/Motion2D-p3-v0"

MOTION2D_ENVIRONMENT_DESCRIPTION = """
    A 2D continuous motion planning environment (Motion2D from the KinDER
    benchmark).

    WORLD:
        - A 2.5 x 2.5 continuous world.
        - A circular robot starts on the left side, and a rectangular target region is on the right side.
        - Between the robot and target are vertical obstacle columns that span the environment from left to right.

    IMPORTANT:
        Each obstacle column is composed of TWO axis-aligned rectangles:
            (1) a bottom rectangle extending upward from the bottom of the world
            (2) a top rectangle extending downward from the top of the world
        These two rectangles share the same x-position and width, and together form a vertical wall
        with a single open gap between them.

        The only traversable region through each wall column is this vertical gap.

        - The number of such columns depends on the environment variant.
        - The x-positions of the columns increase from left to right between the robot and the target.

    OBSERVATION:
        - The observation is a flat numpy array (float32).

        - Robot:
            - obs[0], obs[1]: robot x, y position
            - obs[2]: robot theta (orientation in radians)
            - obs[3]: robot base radius (read from obs[3]; typically ~0.1)
            - The robot is a circle. The full circular body must fit through
              any passage — i.e., the robot center must be at least obs[3]
              away from every obstacle edge.

        - Target region:
            - obs[9], obs[10]: bottom-left corner (x, y)
            - obs[17], obs[18]: width and height

        - Obstacles:
            - Starting from obs[19], obstacles are listed sequentially
            - Each obstacle has 10 values:
                (x, y, theta, static, color_r, color_g, color_b, z_order, width, height)

            - Obstacles corresponding to a wall column appear in pairs:
                - the first rectangle is the bottom segment
                - the second rectangle is the top segment

            - For passage i:
                - bottom obstacle starts at index 19 + 20*i
                - top obstacle starts at index 19 + 20*i + 10

            - For a given column:
                - both rectangles share the same x-position and width
                - the wall x-position is obs[19 + 20*i] (same for both segments)
                - the bottom rectangle spans from its y up to y + height
                - the top rectangle starts at its y and extends upward

            - The open passage (gap) lies between:
                    (bottom_y + bottom_height) and (top_y)
            - Gap heights are on the order of 0.3–0.4 units. Given the robot
              diameter (~0.2), the clearance on each side is only ~0.05
              units. Precise y-alignment before entering the gap is necessary.

    ACTIONS:
        - A 5-dimensional continuous action:
        - action[0]: dx in [-0.05, 0.05]
        - action[1]: dy in [-0.05, 0.05]
        - action[2]: dtheta in [-pi/16, pi/16]
        - action[3]: darm in [-0.1, 0.1] (not needed, set to 0)
        - action[4]: vacuum in [0, 1] (not needed, set to 0)
        - The action array must be dtype float32.

    TASK:
        - Move the robot to the target region while avoiding obstacles.
        - Success occurs when the robot center lies inside the target region.

    REWARD:
        -1.0 per step until success.
"""


@pytest.fixture(scope="module", autouse=True)
def _register_envs() -> None:
    if _MOTION2D_P1 not in registry:
        register(
            id=_MOTION2D_P1,
            entry_point="kinder.envs.kinematic2d.motion2d:Motion2DEnv",
            kwargs={"num_passages": 1},
        )
    if _MOTION2D_P3 not in registry:
        register(
            id=_MOTION2D_P3,
            entry_point="kinder.envs.kinematic2d.motion2d:Motion2DEnv",
            kwargs={"num_passages": 3},
        )


def _make_motion2d_env(env_id: str, render_mode: str | None = None) -> Any:
    return kinder.make(env_id, render_mode=render_mode, allow_state_access=True)


def _save_video(frames: list[np.ndarray], path: Path, fps: int = 20) -> None:
    # pylint: disable=import-outside-toplevel
    from moviepy import ImageSequenceClip  # type: ignore[import-untyped]

    clean = [f[:, :, :3] if f.ndim == 3 and f.shape[2] == 4 else f for f in frames]
    clip = ImageSequenceClip(clean, fps=fps)
    clip.write_videofile(str(path), codec="libx264", logger=None)
    print(f"Video saved → {path}  ({len(clean)} frames)")


def test_llm_ppl_approach_motion2d() -> None:
    """Tests LLMPPLApproach on Motion2D with a mock LLM (constant action)."""
    env = _make_motion2d_env(_MOTION2D_P1)
    env.action_space.seed(123)
    constant_action = env.action_space.sample()

    cache_path = Path(tempfile.NamedTemporaryFile(suffix=".db").name)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    response = Response(
        f"""```python
def _policy(obs):
    import numpy as np
    return np.array({constant_action.tolist()}, dtype=np.float32)
```
""",
        {},
    )
    llm = OrderedResponseModel([response], cache)

    approach = LLMPPLApproach(
        MOTION2D_ENVIRONMENT_DESCRIPTION,
        env.observation_space,
        env.action_space,
        seed=123,
        llm=llm,
    )

    obs, info = env.reset(seed=123)
    approach.reset(obs, info)

    for _ in range(5):
        action = approach.step()
        assert env.action_space.contains(action)
        obs, reward, terminated, _, info = env.step(action)
        approach.update(obs, float(reward), terminated, info)

    env.close()


@runllms
def test_llm_ppl_approach_motion2d_with_real_llm(
    request: pytest.FixtureRequest,
) -> None:
    """Tests LLMPPLApproach on Motion2D-p1 with a real LLM. Saves a video if
    --runvisual.

    Run with:  pytest tests/approaches/test_motion2d_llm_ppl_approach.py --runllms -s
    With video: pytest tests/approaches/test_motion2d_llm_ppl_approach.py
        --runllms --runvisual -s
    """
    passages = 1
    synthesis_seed = 42
    eval_seed = 125
    save_video = request.config.getoption("--runvisual", default=False)

    env_id = _MOTION2D_P1
    render_mode = "rgb_array" if save_video else None
    env = _make_motion2d_env(env_id, render_mode=render_mode)

    cache_path = Path("llm_cache.db")
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    llm = OpenAIModel("gpt-5.2", cache)

    approach = LLMPPLApproach(
        MOTION2D_ENVIRONMENT_DESCRIPTION,
        env.observation_space,
        env.action_space,
        seed=123,
        llm=llm,
    )

    # Synthesize policy using a different seed than eval so the LLM
    # doesn't see the exact starting state it will be tested on.
    obs, info = env.reset(seed=synthesis_seed)
    approach.reset(obs, info)

    # Reset to the actual eval seed for the episode.
    obs, info = env.reset(seed=eval_seed)
    approach.reset(obs, info)

    print("Generated policy:\n", approach._policy)  # pylint: disable=protected-access

    frames: list[np.ndarray] = []
    total_reward = 0.0
    steps_taken = 0
    goal_reached = False

    for step in range(500):
        if save_video:
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))
        action = approach.step()
        assert env.action_space.contains(action)
        obs, reward, terminated, _, info = env.step(action)
        approach.update(obs, float(reward), terminated, info)
        total_reward += float(reward)
        steps_taken = step + 1
        if terminated:
            if save_video:
                frame = env.render()
                if frame is not None:
                    frames.append(np.asarray(frame))
            goal_reached = True
            break

    env.close()

    print()
    print("=" * 50)
    print(f"  Env:           {env_id}  (seed={eval_seed})")
    print(f"  Goal reached:  {goal_reached}")
    print(f"  Steps taken:   {steps_taken}")
    print(f"  Total reward:  {total_reward:.1f}")
    print("=" * 50)

    if save_video and frames:
        video_dir = Path("videos")
        video_dir.mkdir(exist_ok=True)
        _save_video(frames, video_dir / f"llm_ppl_p{passages}_s{eval_seed}.mp4")
