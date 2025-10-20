"""Compact iterative LLM parametric policy tuner for Pendulum-v1.

Usage:
  python tests/approaches/run_param_policy_grid_iterative.py \
    --env Pendulum-v1 --episodes 10 --steps 200 \
    --samples 25 --iterations 8 --target_score -300 --diag
"""
import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypedDict, cast

import gymnasium as gym
import numpy as np
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel


class CompiledLLMPolicy:
    """Compiled policy for once LLM provides python code."""

    def __init__(self, fn: Callable[[Any, Dict[str, float]], Any]):
        self.fn = fn


class LLMParametricPolicy:
    """Class for the LLM parameteric policy setting bounds and parameters."""

    def __init__(
        self,
        compiled: CompiledLLMPolicy,
        init_params: Dict[str, float],
        bounds: Dict[str, Tuple[float, float]],
        action_space,
    ):
        self.fn = compiled.fn
        self._params = dict(init_params)
        self._bounds = dict(bounds)
        self.action_space = action_space

    def bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return bounds."""
        return dict(self._bounds)

    def set_params(self, new_params: Dict[str, float]) -> None:
        """Set parameters."""
        for k, v in new_params.items():
            lo, hi = self._bounds[k]
            self._params[k] = float(np.clip(v, lo, hi))

    def act(self, obs: Any) -> np.ndarray:
        """Return action given value and like clipping limits."""
        p = {"threshold": 0.5, "pd_angle": 12.0, "pd_vel": 2.5, **self._params}
        a = self.fn(obs, p)
        a = np.asarray(a, dtype=np.float32).ravel()
        if isinstance(self.action_space, gym.spaces.Box):
            low = np.asarray(self.action_space.low, dtype=np.float32).ravel()
            high = np.asarray(self.action_space.high, dtype=np.float32).ravel()
            a = np.clip(a, low, high)
        else:
            a = np.clip(a, -2.0, 2.0)
        return a.astype(np.float32)


def base_prompt() -> str:
    """Base prompt for LLMs starting prompt."""
    return """Return ONLY Python code for:

def policy(obs, params):
    # obs = [cos(theta), sin(theta), theta_dot]
    # Output: np.array([torque], dtype=np.float32) in [-2, 2]
    # Strategy: hard swing-up, PD near upright.
    import numpy as np  

    cos_t, sin_t, theta_dot = float(obs[0]), float(obs[1]), float(obs[2])
    theta = np.arctan2(sin_t, cos_t)

    threshold = params.get('threshold', 0.5)
    pd_angle  = params.get('pd_angle', 12.0)
    pd_vel    = params.get('pd_vel', 2.5)

    if abs(theta) < threshold:
        u = -pd_angle * theta - pd_vel * theta_dot
    else:
        u = 2.0 if theta_dot >= 0.0 else -2.0

    return np.array([float(np.clip(u, -2.0, 2.0))], dtype=np.float32)
"""


def refine_prompt(
    best_score: float,
    best_params: Dict[str, float],
) -> str:
    """Refining prompt for after base one."""
    return f"""Improve the controller based on this feedback and return ONLY Python code:
Current best avg return: {best_score:.1f}
Current best params: {best_params}
Notes:
- Keep hard (bang bang method) for swing-up (full ±2.0 by velocity sign).
- Balance with PD near top; try slightly higher gains (pd_angle≈12-18, pd_vel≈2-3) and threshold 0.4-0.6.
- Keep it short, no imports except numpy (caller strips them).

def policy(obs, params):
    # obs=[cos, sin, theta_dot]; output np.array([torque], dtype=np.float32)
"""


def _extract_text(obj: Any) -> str:
    """Extracting text from LLM."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for k in ("text", "content"):
            if k in obj and isinstance(obj[k], str):
                return obj[k]
        return obj["choices"][0]["message"]["content"]  # type: ignore[index]
    if hasattr(obj, "content"):
        return obj.content  # type: ignore[return-value, attr-defined]
    if hasattr(obj, "text"):
        return obj.text  # type: ignore[return-value, attr-defined]
    if isinstance(obj, (list, tuple)) and obj:
        return _extract_text(obj[0])
    return str(obj)


def _strip_code_fences(text: str) -> str:
    """Stripping text to only actual python code."""
    if "```" not in text:
        return text.strip()
    parts = text.split("```")
    body = parts[1] if len(parts) >= 3 else parts[0]
    if body.startswith("python"):
        body = body[len("python") :].lstrip("\n")
    return body.strip()


def call_llm_return_code(llm: Any, prompt: str) -> str:
    """Combining extract and stripping to get code from LLM."""
    for m in ("complete", "generate", "chat", "query", "run_query"):
        if hasattr(llm, m):
            fn = getattr(llm, m)
            out = (
                fn(prompt) if m != "chat" else fn([{"role": "user", "content": prompt}])
            )
            return _strip_code_fences(_extract_text(out))
    raise TypeError(f"{type(llm).__name__} has no known query method")


def compile_policy_from_code(code_str: str) -> CompiledLLMPolicy:
    """Compiling LLM policy code."""
    safe_lines = [
        ln
        for ln in code_str.splitlines()
        if not ln.strip().startswith(("import ", "from "))
    ]
    safe_code = "\n".join(safe_lines)
    g = {"np": np, "math": math}
    l: Dict[str, Any] = {}
    exec(safe_code, g, l)  # pylint: disable=exec-used
    return CompiledLLMPolicy(fn=l["policy"])  # type: ignore[index]


def evaluate_policy(
    env_id: str,
    policy: LLMParametricPolicy,
    episodes: int = 10,
    seed: int = 0,
    max_steps: int = 200,
    diag: bool = False,
) -> float:
    """Go through a bunch of iterations and episodes to check return values."""
    env = gym.make(env_id)
    rng = np.random.RandomState(seed)
    total = 0.0
    for _ in range(episodes):
        # ensure randint bounds are ints for mypy
        obs, _ = env.reset(seed=int(rng.randint(0, 1_000_000_000)))
        ret = 0.0
        for _t in range(max_steps):
            action = policy.act(obs)
            obs, rew, terminated, truncated, _ = env.step(action)
            ret += float(rew)
            if terminated or truncated:
                break
        total += ret
    cast(Any, env).close()  # gym close() is untyped; cast silences mypy
    if diag:
        print(f"[diag] avg_return: {total / episodes:.1f}")
    return float(total / episodes)


def random_sample_grid(
    bounds: Dict[str, Tuple[float, float]], n_samples: int, seed: int
) -> Iterator[Dict[str, float]]:
    """Samples random values to put into new function created by LLM."""
    rng = np.random.RandomState(seed)
    keys = list(bounds.keys())
    for _ in range(n_samples):
        yield {k: float(rng.uniform(*bounds[k])) for k in keys}


class IterResult(TypedDict):
    """Tracking iter results."""

    params: Dict[str, float]
    score: float


class HistoryItem(IterResult):
    """Tracking history."""

    iteration: int


def main() -> None:
    """Main method."""
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="Pendulum-v1")
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--samples", type=int, default=25)
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--target_score", type=float, default=-400.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", type=str, default="results/pendulum_iterative_best.json")
    p.add_argument("--diag", action="store_true")
    args = p.parse_args()

    init_params = {"pd_angle": 12.0, "pd_vel": 2.5, "threshold": 0.5}
    bounds = {"pd_angle": (8.0, 20.0), "pd_vel": (1.5, 4.0), "threshold": (0.3, 0.7)}
    _param_names = list(init_params.keys())  # kept, but unused; silence linter

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    llm = OpenAIModel(
        args.model, SQLite3PretrainedLargeModelCache(Path("llm_cache.db"))
    )

    history: List[HistoryItem] = []
    overall_best: Optional[HistoryItem] = None
    current_policy: Optional[CompiledLLMPolicy] = None

    for it in range(1, args.iterations + 1):
        print(f"\n=== Iteration {it}/{args.iterations} ===")

        # pylint: disable=unsubscriptable-object
        if overall_best is None or (
            history and history[-1]["iteration"] == overall_best["iteration"]
        ):
            if overall_best is None:
                prompt = base_prompt()
            else:
                prompt = refine_prompt(overall_best["score"], overall_best["params"])
            print(" Querying LLM for policy...")
            code = call_llm_return_code(llm, prompt)
            compiled = compile_policy_from_code(code)
            current_policy = compiled
            print(" Policy compiled")
        else:
            print(" Reusing best policy code (no new LLM query)")
            assert current_policy is not None
            compiled = current_policy

        print(" Querying LLM for policy...")
        code = call_llm_return_code(llm, prompt)  # type: ignore[name-defined]
        compiled = compile_policy_from_code(code)
        print(" Policy compiled")

        env = gym.make(args.env)
        env.action_space.seed(args.seed)
        policy = LLMParametricPolicy(compiled, init_params, bounds, env.action_space)

        print(f"Sampling {args.samples} param sets...")
        best_iter: Optional[IterResult] = None
        for i, params in enumerate(
            random_sample_grid(policy.bounds(), args.samples, args.seed + it)
        ):
            policy.set_params(params)
            score = evaluate_policy(
                args.env,
                policy,
                episodes=args.episodes,
                seed=args.seed + it,
                max_steps=args.steps,
                diag=args.diag,
            )
            if (best_iter is None) or (
                score > best_iter["score"]
            ):  # pylint: disable=unsubscriptable-object
                best_iter = {"params": dict(params), "score": float(score)}
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{args.samples}...")

        # mypy: best_iter not None here because samples >= 1
        assert best_iter is not None
        print(f"Best this iter: {best_iter['score']:.1f} with {best_iter['params']}")
        history.append(
            {
                "iteration": it,
                "params": best_iter["params"],
                "score": best_iter["score"],
            }
        )

        if (overall_best is None) or (
            best_iter["score"] > overall_best["score"]
        ):  # pylint: disable=unsubscriptable-object
            overall_best = {
                "iteration": it,
                "params": best_iter["params"],
                "score": best_iter["score"],
            }
            print("🌟 New overall best")

        if (
            overall_best["score"] >= args.target_score
        ):  # pylint: disable=unsubscriptable-object
            print(
                f"🎉 Target reached: {overall_best['score']:.1f} ≥ {args.target_score}"
            )
            break

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_best": overall_best,
                "history": history,
                "target_score": args.target_score,
            },
            f,
            indent=2,
        )

    print("\n=== FINAL ===")
    assert overall_best is not None
    print(
        f"Best score:  {overall_best['score']:.1f} (iter {overall_best['iteration']})"  # pylint: disable=unsubscriptable-object
    )
    print(
        f"Best params: {overall_best['params']}"
    )  # pylint: disable=unsubscriptable-object
    print(f"Saved to:    {args.out}")


if __name__ == "__main__":
    main()
