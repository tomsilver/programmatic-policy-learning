"""Evaluate and visualize a local feature-based ARC-AGI-3 policy."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from programmatic_policy_learning.data.demo_io import load_demo_record
from programmatic_policy_learning.envs.arc_agi3 import (
    ArcAgi3GymEnv,
    preprocess_arc_observation,
)
from programmatic_policy_learning.visualization.arc_agi3_plp_trace import (
    generate_arc_lpp_decision_trace,
)

FeatureFn = Callable[[Any, int], bool]
FeatureLookup = Callable[[str, Any, int], bool]

DEFAULT_FEATURES = (
    "src/programmatic_policy_learning/dsl/llm_primitives/prompts/"
    "py_feature_gen/arc_agi3_ls20_offline_features.json"
)
DEFAULT_POLICY = "experiments/local_policies/arc_agi3_ls20_policy.py"


class LocalFeaturePolicy:
    """Adapter exposing a local policy through the LPP visualization interface."""

    def __init__(
        self,
        *,
        feature_functions: dict[str, FeatureFn],
        policy_module: ModuleType,
        candidate_actions: Sequence[int] = (1, 2, 3, 4),
    ) -> None:
        self.feature_functions = feature_functions
        self.policy_module = policy_module
        self.candidate_actions = [int(action) for action in candidate_actions]
        self.map_program = inspect.getsource(policy_module.accepts)
        self.map_posterior = 0.0

    def feature(self, feature_id: str, state: Any, action: int) -> bool:
        """Evaluate one named feature."""
        try:
            function = self.feature_functions[feature_id]
        except KeyError as exc:
            raise KeyError(f"Unknown feature id {feature_id!r}.") from exc
        return bool(function(state, int(action)))

    def accepted_actions(self, state: Any) -> list[int]:
        """Return all actions accepted by the local policy."""
        return [
            action
            for action in self.candidate_actions
            if bool(self.policy_module.accepts(state, action, self.feature))
        ]

    def __call__(self, state: Any) -> int:
        accepted = self.accepted_actions(state)
        if not accepted:
            available = (
                state.get("available_actions", []) if isinstance(state, dict) else []
            )
            fallback = [
                int(getattr(action, "value", action))
                for action in available
                if int(getattr(action, "value", action)) in self.candidate_actions
            ]
            if not fallback:
                raise RuntimeError("No accepted or available action for local policy.")
            return fallback[0]

        chooser = getattr(self.policy_module, "choose_action", None)
        if callable(chooser):
            chosen = int(chooser(state, accepted, self.feature))
            if chosen not in accepted:
                raise ValueError(
                    f"choose_action returned {chosen}, which is not in {accepted}."
                )
            return chosen
        return accepted[0]

    def explain_finite_discrete_decision(self, state: Any) -> dict[str, Any]:
        """Return action decisions in the format expected by the HTML renderer."""
        accepted = self.accepted_actions(state)
        chosen = self(state)
        rows = []
        for action in self.candidate_actions:
            active = [
                feature_id
                for feature_id, function in self.feature_functions.items()
                if bool(function(state, action))
            ]
            rows.append(
                {
                    "action": action,
                    "probability": 1.0 if action == chosen else 0.0,
                    "map_accepts": action in accepted,
                    "active_features": active,
                }
            )
        return {
            "chosen_action": chosen,
            "map_program": self.map_program,
            "map_posterior": self.map_posterior,
            "actions": rows,
        }


def load_feature_functions(path: Path) -> dict[str, FeatureFn]:
    """Compile feature functions from an offline feature payload."""
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    functions: dict[str, FeatureFn] = {}
    for feature in payload["features"]:
        feature_id = str(feature["id"])
        source = str(feature["source"]).replace("\\n", "\n")
        namespace: dict[str, Any] = {}
        exec(  # pylint: disable=exec-used
            compile(ast.parse(source), f"<{feature_id}>", "exec"),
            namespace,
        )
        function_name = str(feature.get("name", feature_id))
        function = namespace.get(function_name)
        if not callable(function):
            function = next(
                (
                    value
                    for key, value in namespace.items()
                    if key != "__builtins__" and callable(value)
                ),
                None,
            )
        if not callable(function):
            raise ValueError(f"No callable feature found for {feature_id}.")
        functions[feature_id] = function
    return functions


def load_policy_module(path: Path) -> ModuleType:
    """Import a local policy Python file."""
    spec = importlib.util.spec_from_file_location("arc_agi3_local_policy", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local policy from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "accepts", None)):
        raise ValueError("Local policy must define accepts(state, action, feature).")
    return module


def evaluate_demonstrations(
    *,
    policy: LocalFeaturePolicy,
    demos_dir: Path,
    demo_glob: str,
) -> None:
    """Print acceptance and deterministic-choice metrics on saved demos."""
    paths = sorted(demos_dir.glob(demo_glob))
    if not paths:
        raise FileNotFoundError(f"No demonstrations match {demos_dir / demo_glob}.")

    expert_accepted = 0
    chosen_matches = 0
    total = 0
    accepted_histogram: Counter[int] = Counter()

    for demo_index, path in enumerate(paths):
        record = load_demo_record(path)
        print(f"\nDemo {demo_index}: {path.name}")
        for step_index, (observation, expert_action) in enumerate(
            record.trajectory.steps
        ):
            state = preprocess_arc_observation(
                observation,
                game_id=str(record.metadata.get("game_id", "ls20")),
            )
            expert = int(getattr(expert_action, "value", expert_action))
            accepted = policy.accepted_actions(state)
            chosen = policy(state)
            expert_accepted += int(expert in accepted)
            chosen_matches += int(chosen == expert)
            accepted_histogram[len(accepted)] += 1
            total += 1
            player = state.get("player") or {}
            print(
                f"  step={step_index:02d} player={player.get('center')} "
                f"expert={expert} accepted={accepted} chosen={chosen}"
            )

    print("\nDemonstration summary")
    print(f"  states: {total}")
    print(f"  expert-action acceptance: {expert_accepted}/{total}")
    print(f"  deterministic chosen-action match: {chosen_matches}/{total}")
    print(
        f"  accepted-action count histogram: {dict(sorted(accepted_histogram.items()))}"
    )


def main() -> int:
    """Evaluate a local policy on demonstrations and an optional live rollout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-file", default=DEFAULT_POLICY)
    parser.add_argument("--features-json", default=DEFAULT_FEATURES)
    parser.add_argument(
        "--demos-dir",
        default="manual_demos/arc_agi3/ls20",
    )
    parser.add_argument("--demo-glob", default="*shape_match*.pkl")
    parser.add_argument("--demo-only", action="store_true")
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument(
        "--stop-after-levels",
        type=int,
        default=1,
        help="stop the live trace after this many completed levels; use 0 to disable",
    )
    parser.add_argument("--env-dir", default="environment_files")
    parser.add_argument("--recordings-dir", default="recordings")
    parser.add_argument(
        "--html-output",
        default="logs/arc_agi3_local_policy_trace.html",
    )
    args = parser.parse_args()

    feature_path = Path(args.features_json)
    policy_path = Path(args.policy_file)
    policy = LocalFeaturePolicy(
        feature_functions=load_feature_functions(feature_path),
        policy_module=load_policy_module(policy_path),
    )
    evaluate_demonstrations(
        policy=policy,
        demos_dir=Path(args.demos_dir),
        demo_glob=args.demo_glob,
    )

    if args.demo_only:
        return 0

    env = ArcAgi3GymEnv(
        game_id=args.game,
        observation_format="processed",
        operation_mode="offline",
        environments_dir=args.env_dir,
        recordings_dir=args.recordings_dir,
        render_mode=None,
        seed=args.seed,
    )
    report_path = generate_arc_lpp_decision_trace(
        env=env,
        policy=policy,
        output_path=args.html_output,
        max_steps=args.max_steps,
        reset_seed=args.seed,
        feature_json_path=feature_path,
        stop_after_levels=(
            args.stop_after_levels if args.stop_after_levels > 0 else None
        ),
    )
    print(f"\nLive rollout HTML: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
