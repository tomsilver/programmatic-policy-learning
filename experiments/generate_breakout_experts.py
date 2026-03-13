# experiments/generate_breakout_llm_experts.py
"""
Generate an INTERPRETABLE, HIGH-LEVEL PROMPT FAMILY of LLM-based Breakout (RAM) experts.

Key design choices:
- We provide ONLY the environment interface + what signals exist (RAM indices + actions).
- We DO NOT specify numeric constants (no deadband_px, no y-thresholds, no timers).
- Each variant is a plain-English strategy brief. The LLM must choose any constants itself.

Outputs:
  experiments/breakout_llm_expert_<name>.py

Suggested variants focus on interpretable intent:
  - safe_tracker
  - center_hitter
  - right_pressure
  - left_pressure
  - tunnel_hunter_right
  - tunnel_hunter_left
  - adaptive_switch
  - patient_builder
  - aggressive_angle
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.code import SyntaxRepromptCheck, synthesize_python_function_with_llm
from prpl_llm_utils.models import OpenAIModel
from prpl_llm_utils.structs import Query

# -----------------------------
# Config
# -----------------------------
SEED = 0
MODEL_NAME = "gpt-4o-mini"
CACHE_PATH = "llm_cache.db"
OUT_DIR = Path("experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# If you only want a subset:
ENABLED_NAMES: set[str] | None = None
# Example:
# ENABLED_NAMES = {"safe_tracker", "tunnel_hunter_right"}


# -----------------------------
# Minimal contract (shared by all prompts)
# -----------------------------
BASE_CONTRACT = r"""
You are writing a deterministic, hand-crafted controller for:
  Gymnasium ALE/Breakout-v5 with obs_type="ram".

What you can observe:
- obs is a NumPy array of length 128 (RAM bytes).
- You may read these signals (common ALE Breakout indices):
    paddle_x = obs[72]
    ball_x   = obs[99]
    ball_y   = obs[101]

What actions mean:
- Action space is Discrete with:
    0 = NOOP
    1 = FIRE
    2 = RIGHT
    3 = LEFT

Your job:
- Write ONLY this function (and nothing else), with all logic inside it:

    def expert_policy(obs: "np.ndarray") -> int:

Hard requirements (must follow exactly):
1) Parse obs exactly like:
      obs = np.asarray(obs)
      if obs.ndim != 1 or obs.size < 128: return 1
      ram = obs.astype(np.int32, copy=False)
      paddle_x = int(ram[72]); ball_x = int(ram[99]); ball_y = int(ram[101])

2) Deterministic only. No randomness. No prints. No comments.

3) You MAY keep persistent state ONLY via function attributes
   (e.g., expert_policy._steps, expert_policy._prev_ball_x, etc.).
   All state MUST be lazily initialized inside expert_policy.
   No globals, no helper functions.

4) Serving must be handled:
   - Early after reset, you need to press FIRE for a short period to launch the ball.
   - If the game appears stuck (ball doesn't move for a long time), you may FIRE again briefly.

5) Return a valid int action in {0,1,2,3} every call.

Important constraint for this study:
- Do NOT hard-code specific numeric constants in your policy specification
  beyond what is necessary to serve at the start.
- If you need thresholds/timers/deadbands, choose them yourself implicitly in code,
  but do NOT mention or justify specific numbers in your reasoning (no comments anyway).
- Keep the policy simple and interpretable.

Output MUST be only the function definition of expert_policy.
Assume `np` is already imported.
"""


# -----------------------------
# Plain-English strategy briefs (no coding language, no constants)
# -----------------------------
SAFE_TRACKER = r"""
Strategy brief (interpretable intent): "Safety-first tracker"

Primary objective: keep the ball in play as reliably as possible.
- After serving, mostly stay under the ball horizontally.
- Avoid twitchy left-right jitter; move smoothly.
- Do not attempt fancy steering shots; prioritize catching the ball.
"""

CENTER_HITTER = r"""
Strategy brief (interpretable intent): "Center hitter / vertical control"

Objective: keep the ball's motion controlled and avoid extreme sideways bounces.
- After serving, track the ball like a normal player.
- When the ball is coming down toward the paddle, try to meet it in a way that produces
  a more 'straight' return (less sideways drift), while still being safe.
- If there's a conflict between control and safety, choose safety.
"""

RIGHT_PRESSURE = r"""
Strategy brief (interpretable intent): "Right pressure"

Objective: bias play toward the right side without losing the ball.
- After serving, track the ball like a normal player.
- When it is safe to do so, favor returning the ball toward the right side.
- Do not sacrifice catches: if the ball is hard to reach, focus on saving it.
"""

LEFT_PRESSURE = r"""
Strategy brief (interpretable intent): "Left pressure"

Objective: bias play toward the left side without losing the ball.
- After serving, track the ball like a normal player.
- When it is safe to do so, favor returning the ball toward the left side.
- Do not sacrifice catches: if the ball is hard to reach, focus on saving it.
"""

TUNNEL_HUNTER_RIGHT = r"""
Strategy brief (interpretable intent): "Tunnel hunter (right)"

Objective: pursue the classic Breakout tunneling idea on the right side.
- After serving, be safety-first near the paddle.
- When the ball is far away (up near the bricks), position yourself in a way that supports
  repeated play on the right side.
- When the ball is coming down and you are well-positioned, try to return it in a way that
  encourages right-side repeat hits.
- If in doubt, catch the ball; tunneling is secondary to survival.
"""

TUNNEL_HUNTER_LEFT = r"""
Strategy brief (interpretable intent): "Tunnel hunter (left)"

Objective: pursue the classic Breakout tunneling idea on the left side.
- After serving, be safety-first near the paddle.
- When the ball is far away (up near the bricks), position yourself in a way that supports
  repeated play on the left side.
- When the ball is coming down and you are well-positioned, try to return it in a way that
  encourages left-side repeat hits.
- If in doubt, catch the ball; tunneling is secondary to survival.
"""

ADAPTIVE_SWITCH = r"""
Strategy brief (interpretable intent): "Adaptive side switcher"

Objective: try a side-focused plan, but switch if it seems to be failing.
- Start by favoring one side (choose deterministically).
- If you lose a life or the ball seems to repeatedly escape your control, switch to favoring
  the opposite side for a while.
- Always prioritize catching the ball near the paddle.
"""

PATIENT_BUILDER = r"""
Strategy brief (interpretable intent): "Patient builder"

Objective: play conservatively until opportunities arise.
- Emphasize stable returns and long rallies.
- Only apply side-bias or steering when you have strong control (you feel 'set').
- If control is poor, return to simple tracking.
"""

AGGRESSIVE_ANGLE = r"""
Strategy brief (interpretable intent): "Aggressive angles"

Objective: seek faster brick-breaking by creating more sideways motion.
- Still serve properly and avoid losing immediately.
- When the ball is coming down and you are in control, try to create angled returns
  that sweep across bricks.
- If the ball is low and dangerous, prioritize saving it over aggression.
"""


@dataclass(frozen=True)
class PromptVariant:
    name: str
    strategy_brief: str


def build_prompt_family() -> list[PromptVariant]:
    return [
        PromptVariant("safe_tracker", SAFE_TRACKER),
        PromptVariant("center_hitter", CENTER_HITTER),
        PromptVariant("right_pressure", RIGHT_PRESSURE),
        PromptVariant("left_pressure", LEFT_PRESSURE),
        PromptVariant("tunnel_hunter_right", TUNNEL_HUNTER_RIGHT),
        PromptVariant("tunnel_hunter_left", TUNNEL_HUNTER_LEFT),
        PromptVariant("adaptive_switch", ADAPTIVE_SWITCH),
        PromptVariant("patient_builder", PATIENT_BUILDER),
        PromptVariant("aggressive_angle", AGGRESSIVE_ANGLE),
    ]


# -----------------------------
# Helpers
# -----------------------------
def build_llm_model() -> OpenAIModel:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    cache = SQLite3PretrainedLargeModelCache(Path(CACHE_PATH))
    return OpenAIModel(model_name=MODEL_NAME, cache=cache)


def synthesize_and_write(llm: OpenAIModel, prompt: str, out_path: Path) -> None:
    """Synthesize expert_policy from `prompt` and write it to out_path."""
    full_prompt = prompt + f"\n# Generation seed tag: {SEED}\n"
    query = Query(full_prompt)

    cand = synthesize_python_function_with_llm(
        "expert_policy",
        llm,
        query,
        reprompt_checks=[SyntaxRepromptCheck()],
    )

    # Extract source string from SynthesizedPythonFunction
    src = None
    for _, value in vars(cand).items():
        if isinstance(value, str) and "def expert_policy" in value:
            src = value
            break

    if src is None:
        raise RuntimeError(
            "Could not extract source code from SynthesizedPythonFunction. "
            "No attribute contained 'def expert_policy'."
        )

    out_path.write_text("import numpy as np\n\n" + src + "\n")
    print(f"[generate_breakout_experts] Wrote {out_path.resolve()}")


def main() -> None:
    np.random.seed(SEED)
    llm = build_llm_model()

    family = build_prompt_family()
    if ENABLED_NAMES is not None:
        family = [v for v in family if v.name in ENABLED_NAMES]
        if not family:
            raise RuntimeError(f"ENABLED_NAMES={ENABLED_NAMES} matched no variants.")

    for v in family:
        prompt = BASE_CONTRACT + "\n" + v.strategy_brief
        out_path = OUT_DIR / f"breakout_llm_expert_{v.name}.py"
        synthesize_and_write(llm, prompt, out_path)


if __name__ == "__main__":
    main()