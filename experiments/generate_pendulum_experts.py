"""
Generate exactly TWO LLM-based Pendulum experts and save them:

  1) experiments/pendulum_llm_expert_basic.py
  2) experiments/pendulum_llm_expert_structured.py

No candidate search / selection logic: this does exactly two LLM calls.
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
# Prompts
# -----------------------------

BASIC_PROMPT = """
You are writing a hand-crafted controller for the Gymnasium Pendulum-v1 environment.

Environment details:
- Observation: obs is a NumPy array of shape (3,) = [cos(theta), sin(theta), theta_dot]
- Action: a NumPy array of shape (1,) containing a single torque value in [-2.0, 2.0]
- Goal: keep the pendulum upright (theta near 0) and stable.
- No randomness, no print statements, no comments.

Write ONLY the following function, with all logic inside it:

    def expert_policy(obs: "np.ndarray") -> "np.ndarray":

Requirements:
- obs is a NumPy array of shape (3,).
- Reconstruct theta using: theta = np.arctan2(y, x) where x = obs[0], y = obs[1].
- You may use any deterministic control law (PD control and/or swing-up logic).
- Compute a torque scalar, clip it to [-2.0, 2.0] using np.clip,
  and return np.array([torque], dtype=np.float32).
- The function must be deterministic.
- Assume `np` (NumPy) is already imported.
- Do NOT include anything other than the function definition.
"""

STRUCTURED_PROMPT = """
You are writing a hand-crafted controller for the Gymnasium Pendulum-v1 environment.

Environment details:
- Observation: obs is a NumPy array of shape (3,) = [cos(theta), sin(theta), theta_dot]
- Action: a NumPy array of shape (1,) containing a single torque value in [-2.0, 2.0]
- Goal: keep the pendulum upright (theta near 0) and stable.
- No randomness, no print statements, no comments.

Write ONLY the following function, with all logic inside it:

    def expert_policy(obs: "np.ndarray") -> "np.ndarray":

Hard requirements (match this overall STRUCTURE closely):
1) Parse obs exactly like:
      obs = np.asarray(obs, dtype=np.float32)
      x = float(obs[0]); y = float(obs[1]); angvel = float(obs[2])
      theta = float(np.arctan2(y, x))

2) Define two boolean regime tests (you must choose the numeric thresholds yourself):
      is_hanging_down = abs(abs(theta) - np.pi) < HANG_THRESH
      is_near_top     = abs(theta) < TOP_THRESH

3) Piecewise control logic must follow this pattern:
   - If is_hanging_down:
        * If abs(angvel) > VEL_SMALL:
              torque = 2.0 * np.sign(angvel)
          else:
              torque = 2.0 * np.sign(theta)
   - Elif is_near_top:
        * Use PD stabilization:
              torque = -kp_top * theta - kd_top * angvel
   - Else:
        * Use a different PD-like law (different gains than near-top):
              torque = -kp_mid * theta - kd_mid * angvel
        * If abs(angvel) > VEL_FAST:
              torque = torque + BOOST * np.sign(angvel)

4) Clip and return exactly like:
      torque = float(np.clip(torque, -2.0, 2.0))
      return np.array([torque], dtype=np.float32)

Tuning freedom (do NOT hard-code numbers from any reference):
- You must choose reasonable constants:
    HANG_THRESH, TOP_THRESH, VEL_SMALL, VEL_FAST, BOOST,
    kp_top, kd_top, kp_mid, kd_mid
- But you MUST keep:
    * the hanging-down torques at magnitude 2.0
    * the same regime structure and sign logic above
    * the action clipped to [-2.0, 2.0]

Additional rules:
- Deterministic only.
- Assume np is already imported.
- Output MUST be only the function definition of expert_policy.
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
    print(f"[generate_pendulum_experts] Wrote {out_path.resolve()}")


def main() -> None:
    np.random.seed(SEED)
    llm = build_llm_model()

    synthesize_and_write(
        llm,
        BASIC_PROMPT,
        OUT_DIR / "pendulum_llm_expert_basic.py",
    )

    synthesize_and_write(
        llm,
        STRUCTURED_PROMPT,
        OUT_DIR / "pendulum_llm_expert_structured.py",
    )


if __name__ == "__main__":
    main()
