# experiments/generate_lunarlander_experts.py
"""
Generate exactly ONE LLM-based LunarLanderContinuous expert and save it:

  1) experiments/lander_llm_expert.py

No candidate search / selection logic: this does exactly one LLM call.
"""

import os
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

# -----------------------------
# Prompt
# -----------------------------

LUNARLANDER_PROMPT = """
You are writing a hand-crafted controller for the Gymnasium LunarLanderContinuous-v3 environment.

Environment details:
- Observation: obs is a NumPy array of shape (8,) =
    [x, y, vx, vy, angle, ang_vel, leg_l, leg_r]
- Action: a NumPy array of shape (2,) =
    [main_engine, side_engine]
  with BOTH values clipped to [-1.0, 1.0]
- Goal: land softly and stably near the center (x ~ 0), upright (angle ~ 0), low velocity.

Rules:
- Deterministic only (no randomness).
- No print statements.
- No comments.
- Assume `np` (NumPy) is already imported.
- Output MUST be only the function definition of expert_policy.

Write ONLY the following function, with all logic inside it:

    def expert_policy(obs: "np.ndarray") -> "np.ndarray":

Hard requirements:
1) Parse obs exactly like:
      obs = np.asarray(obs, dtype=np.float32)
      x = float(obs[0]); y = float(obs[1])
      vx = float(obs[2]); vy = float(obs[3])
      angle = float(obs[4]); angvel = float(obs[5])
      leg_l = float(obs[6]); leg_r = float(obs[7])

2) Use a simple piecewise controller with at least TWO regimes based on altitude y:
      near_ground = y < Y_THRESH
   Choose Y_THRESH yourself.

3) Control intent (you decide gains, keep it conservative):
   - Main engine should counteract downward velocity and control descent rate,
     especially when near_ground.
   - Side engine should help reduce angle error and move x toward 0.
   - Add damping terms on vx, vy, and angvel to reduce oscillations.

4) Always clip:
      main = float(np.clip(main, -1.0, 1.0))
      side = float(np.clip(side, -1.0, 1.0))
      return np.array([main, side], dtype=np.float32)

Output constraints:
- Return shape must be (2,) float32 every time.
- Must always return finite values.
"""

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
    """Synthesize expert_policy from `prompt` and write it to out_path.

    We append a seed tag to reduce accidental cache collisions.
    """
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
    print(f"[generate_lunarlander_experts] Wrote {out_path.resolve()}")


def main() -> None:
    np.random.seed(SEED)
    llm = build_llm_model()

    synthesize_and_write(
        llm,
        LUNARLANDER_PROMPT,
        OUT_DIR / "lander_llm_expert.py",
    )


if __name__ == "__main__":
    main()