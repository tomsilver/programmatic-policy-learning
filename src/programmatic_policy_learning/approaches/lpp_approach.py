"""An approach that learns a logical programmatic policy from data."""

import ast
import logging
import random
import signal
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Generator, List, Sequence, Tuple, TypeVar, cast

import numpy as np
from gymnasium.spaces import Space

# from omegaconf import DictConfig
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel
from scipy.special import logsumexp

from programmatic_policy_learning.approaches.base_approach import BaseApproach
from programmatic_policy_learning.approaches.utils import (
    convert_dir_lists_to_tuples,
    load_hint_text,
    run_single_episode,
    sample_transition_example,
)
from programmatic_policy_learning.data.collect import get_demonstrations
from programmatic_policy_learning.data.dataset import run_all_programs_on_demonstrations
from programmatic_policy_learning.dsl.generators.grammar_based_generator import (
    Grammar,
    GrammarBasedProgramGenerator,
)

# from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based import (
#     grid_hint_config,
# )
from programmatic_policy_learning.dsl.llm_primitives.feature_generator import (
    LLMFeatureGenerator,
)
from programmatic_policy_learning.dsl.llm_primitives.feature_priors import (
    compute_feature_log_probs,
)
from programmatic_policy_learning.dsl.llm_primitives.llm_generator import (
    LLMPrimitivesGenerator,
)
from programmatic_policy_learning.dsl.llm_primitives.py_feature_generator import (
    PyFeatureGenerator,
)
from programmatic_policy_learning.dsl.llm_primitives.utils import (
    JSONStructureRepromptCheck,
)
from programmatic_policy_learning.dsl.primitives_sets.grid_v1 import (
    GridInput,
    LocalProgram,
    create_grammar,
    get_dsl_functions_dict,
    make_ablated_dsl,
    make_dsl,
)
from programmatic_policy_learning.dsl.state_action_program import (
    StateActionProgram,
    set_dsl_functions,
)
from programmatic_policy_learning.learning.decision_tree_learner import learn_plps
from programmatic_policy_learning.learning.particles_utils import select_particles
from programmatic_policy_learning.learning.plp_likelihood import compute_likelihood_plps
from programmatic_policy_learning.learning.prior_calculation import priors_from_features
from programmatic_policy_learning.policies.lpp_policy import LPPPolicy

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")
EnvFactory = Callable[[int | None], Any]


class CustomTimeoutError(Exception):
    """Custom exception raised when a time limit is exceeded.

    This exception is used to indicate that a block of code has exceeded
    the allowed time limit set by the `time_limit` context manager.
    """


@contextmanager
def time_limit(seconds: int) -> Generator[None, None, None]:
    """Context manager to enforce a time limit on a block of code.

    Args:
        seconds (int): The maximum number of seconds to allow the block to run.

    Raises:
        CustomTimeoutError: If the time limit is exceeded.
    """

    def handler(signum: int, frame: FrameType | None) -> None:
        """Signal handler to raise a CustomTimeoutError when the alarm is
        triggered.

        Args:
            signum (int): The signal number.
            frame (FrameType | None): The current stack frame (unused).
        """
        raise CustomTimeoutError(f"Timed out after {seconds} seconds")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


REPO_ROOT = Path(__file__).resolve().parents[1]
HINTS_ROOT = (
    REPO_ROOT
    / "dsl"
    / "llm_primitives"
    / "hint_generation"
    / "llm_based"
    / "final_hints"
)


def build_py_feature_functions(
    feature_programs: list[str],
    dsl_functions: dict[str, Any],
) -> dict[str, Any]:
    """Build a dict of feature function names to callables from source
    strings."""
    functions: dict[str, Any] = {}
    for source in feature_programs:

        tree = ast.parse(source)
        func_names = [
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        ]
        if not func_names:
            raise ValueError("Expected at least one function definition in feature.")

        module_globals = dict(dsl_functions)
        exec(source, module_globals)  # pylint: disable=exec-used
        for name in func_names:
            fn = module_globals.get(name)
            if not callable(fn):
                raise ValueError(f"Feature function '{name}' is not callable.")
            functions[name] = fn
    return functions


def get_program_set(
    num_programs: int,
    base_class_name: str,  # pylint: disable=unused-argument
    env_factory: EnvFactory,
    env_specs: dict[str, Any] | None = None,
    start_symbol: int = 0,
    program_generation: dict[str, Any] | None = None,
    outer_feedback: str | None = None,
    seed: int = 0,
) -> tuple[list, list, dict]:
    """Enumerate programs from the grammar and return programs + prior log-
    probs.

    This helper creates the DSL and the grammar-based generator, then
    samples `num_programs` programs from the generator. It returns a tuple of
    (programs, program_prior_log_probs).
    """
    if program_generation is None:
        raise ValueError(
            "program_generation configuration is required for LPP approach."
        )
    strategy = program_generation["strategy"]

    # Define strategies as a dictionary
    strategies = {
        "fixed_grid_v1": lambda: _generate_with_fixed_grid_v1(env_specs, start_symbol),
        "dsl_generator": lambda: _generate_with_dsl_generator(
            program_generation, env_specs, start_symbol, env_factory, outer_feedback
        ),
        "grid_v1_ablated": lambda: _generate_with_ablated_grid_v1(
            program_generation, env_specs, start_symbol
        ),
        "offline_loader": lambda: _generate_with_offline_loader(
            program_generation, env_specs, start_symbol
        ),
    }
    if strategy == "feature_generator":
        cache_path = Path("feature_cache.db")
        cache = SQLite3PretrainedLargeModelCache(cache_path)
        llm_client = OpenAIModel("gpt-4.1", cache)
        prompt_path = program_generation["feature_generator_prompt"]
        num_features = program_generation["num_features"]
        env = env_factory(0)
        object_types = env.get_object_types()
        generator = LLMFeatureGenerator(llm_client)
        hint_text = load_hint_text(
            base_class_name,
            program_generation["encoding_method"],
            program_generation["hint_structured"],
            HINTS_ROOT,
        )

        features, payload = generator.generate(
            prompt_path=prompt_path,
            object_types=object_types,
            hint_text=hint_text,
            num_features=num_features,
        )
        features = convert_dir_lists_to_tuples(features)

        program_prior_log_probs = compute_feature_log_probs(payload, object_types)
        dsl_fns = get_dsl_functions_dict()
        return features, program_prior_log_probs, dsl_fns
    if strategy == "py_feature_gen":
        cache_path = Path("py_feature_cache.db")
        cache = SQLite3PretrainedLargeModelCache(cache_path)
        llm_client = OpenAIModel("gpt-4.1", cache)
        prompt_path = program_generation["py_feature_gen_prompt"]
        batch_prompt_path = program_generation.get("py_feature_gen_batch_prompt")
        num_features = program_generation["num_features"]
        num_batches = program_generation.get("num_batches")
        env = env_factory(0)
        object_types = env.get_object_types()
        # object_types = grid_hint_config.SALIENT_TOKENS[base_class_name]
        py_generator = PyFeatureGenerator(llm_client)
        hint_text = load_hint_text(
            base_class_name,
            program_generation["encoding_method"],
            program_generation["hint_structured"],
            HINTS_ROOT,
        )

        st, at, st1 = sample_transition_example(
            env_factory, base_class_name, "1", max_steps=40
        )
        features, payload = py_generator.generate(
            prompt_path=prompt_path,
            batch_prompt_path=batch_prompt_path,
            object_types=object_types,
            hint_text=hint_text,
            num_features=num_features,
            num_batches=num_batches,
            state_t_example=st,
            action_example=at,
            state_t1_example=st1,
            _seed=seed,
            reprompt_checks=[JSONStructureRepromptCheck(required_fields=["features"])],
            offline_json_path=program_generation["offline_json_path"],
        )
        # program_prior_log_probs = [-4.0] * len(features)
        dsl_fns = get_dsl_functions_dict()
        dsl_fns.update(
            build_py_feature_functions(features, dsl_fns)
        )  # pylint: disable=exec-used
        out_dict = priors_from_features(features)
        program_prior_log_probs = out_dict["logprobs"]
        return features, program_prior_log_probs, dsl_fns

    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Call the appropriate strategy
    program_generator, dsl_fns = strategies[strategy]()
    # Generate programs using the shared helper function
    programs, program_prior_log_probs = _generate_programs(
        program_generator, num_programs
    )
    return programs, program_prior_log_probs, dsl_fns


def _generate_with_fixed_grid_v1(
    env_specs: dict[str, Any] | None, start_symbol: int
) -> tuple[GrammarBasedProgramGenerator, dict[str, Any]]:
    """Generate programs using the fixed grid_v1 DSL."""

    dsl = make_dsl()
    dsl_dict = get_dsl_functions_dict()

    program_generator = GrammarBasedProgramGenerator(
        cast(
            Callable[[dict[str, Any]], Grammar[LocalProgram, GridInput, Any]],
            create_grammar,
        ),
        dsl,
        env_spec=env_specs if env_specs is not None else {},
        start_symbol=start_symbol,
    )
    return program_generator, dsl_dict


def _generate_with_ablated_grid_v1(
    program_generation: dict[str, Any],
    env_specs: dict[str, Any] | None,
    start_symbol: int,
) -> tuple[GrammarBasedProgramGenerator, dict[str, Any]]:
    """Generate programs using the ablated grid_v1 DSL."""
    removed_primitive = program_generation["removed_primitive"]
    dsl = make_ablated_dsl(removed_primitive)
    dsl_dict = get_dsl_functions_dict(removed_primitive)
    program_generator = GrammarBasedProgramGenerator(
        cast(
            Callable[[dict[str, Any]], Grammar[LocalProgram, GridInput, Any]],
            create_grammar,
        ),
        dsl,
        env_spec=env_specs if env_specs is not None else {},
        start_symbol=start_symbol,
        removed_primitive=removed_primitive,
    )
    return program_generator, dsl_dict


def _generate_with_dsl_generator(
    program_generation: dict[str, Any],
    env_specs: dict[str, Any] | None,
    start_symbol: int,
    env_factory: EnvFactory,
    outer_feedback: str | None,
) -> tuple[GrammarBasedProgramGenerator, dict[str, Any]]:
    """Generate programs using the DSL generator."""
    cache_path = Path("llm_cache.db")
    # cache_path = Path(tempfile.NamedTemporaryFile(suffix=".db").name)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    llm_client = OpenAIModel("gpt-4.1", cache)
    prompt_path = program_generation["dsl_generator_prompt"]
    with open(
        prompt_path,
        "r",
        encoding="utf-8",
    ) as file:
        prompt = file.read()
    removed_primitive = program_generation["removed_primitive"]
    generator = LLMPrimitivesGenerator(llm_client, removed_primitive)
    if env_specs is None:
        raise ValueError("env_specs cannot be None when indexing.")
    _, new_dsl_dict, dsl = generator.generate_and_process_grammar(
        prompt,
        env_specs["object_types"],
        env_factory,  # type: ignore
        "full",
        outer_feedback,
    )

    program_generator = GrammarBasedProgramGenerator(
        generator.create_grammar,
        dsl,
        env_spec=env_specs if env_specs is not None else {},
        start_symbol=start_symbol,
    )

    return program_generator, new_dsl_dict


def _generate_with_offline_loader(
    program_generation: dict[str, Any],
    env_specs: dict[str, Any] | None,
    start_symbol: int,
) -> tuple[GrammarBasedProgramGenerator, dict[str, Any]]:
    """Generate programs using the offline loader."""
    run_id = program_generation["offline_loader_run_id"]
    removed_primitive = program_generation["removed_primitive"]
    generator = LLMPrimitivesGenerator(None, removed_primitive)
    _, new_dsl_dict, dsl = generator.offline_loader(run_id)
    program_generator = GrammarBasedProgramGenerator(
        generator.create_grammar,
        dsl,
        env_spec=env_specs if env_specs is not None else {},
        start_symbol=start_symbol,
    )
    return program_generator, new_dsl_dict


def _generate_programs(
    program_generator: GrammarBasedProgramGenerator, num_programs: int
) -> tuple[list[Any], list[float]]:
    """Shared logic for generating programs."""
    logging.info(f"Generating {num_programs} programs")

    programs: list[Any] = []
    program_prior_log_probs = []
    gen = program_generator.generate_programs()
    for _ in range(num_programs):
        try:
            program, prior = next(gen)
        except StopIteration:
            logging.info(
                f"Generator exhausted early — only produced {len(programs)} programs."
            )
            break
        programs.append(program)
        program_prior_log_probs.append(prior)
    return programs, program_prior_log_probs


def _format_one_example(
    s: np.ndarray,
    a: tuple[int, int],
    *,
    label: int,
    idx: int,
) -> str:
    rows = []
    for r in range(s.shape[0]):
        row = ", ".join(repr(str(x)) for x in s[r])
        rows.append(f"[{row}]")
    grid = "\n".join(f"  {row}" for row in rows)
    r, c = int(a[0]), int(a[1])
    cell = s[r, c] if 0 <= r < s.shape[0] and 0 <= c < s.shape[1] else "OOB"
    return f"- idx={idx} label={label} action=({r}, {c}) cell={cell}\n[\n{grid}\n]"


def build_collision_repair_prompt(
    pos_indices: List[int],
    neg_indices: List[int],
    examples: List[Tuple[np.ndarray, Tuple[int, int]]],
    *,
    env_name: str | None = None,
    existing_feature_summary: str | None = None,
    max_per_label: int = 5,
) -> str:
    if not pos_indices or not neg_indices:
        raise ValueError(
            "Need at least 1 positive and 1 negative from the SAME feature-key bucket."
        )

    pos_indices = pos_indices[:max_per_label]
    neg_indices = neg_indices[:max_per_label]

    pos_blocks = []
    for idx in pos_indices:
        s, a = examples[idx]
        pos_blocks.append(_format_one_example(s, a, label=1, idx=idx))

    neg_blocks = []
    for idx in neg_indices:
        s, a = examples[idx]
        neg_blocks.append(_format_one_example(s, a, label=0, idx=idx))

    env_line = f"ENV: {env_name}\n\n" if env_name else ""
    feat_line = ""
    if existing_feature_summary:
        feat_line = f"EXISTING FEATURES (summary):\n{existing_feature_summary}\n\n"

    prompt = f"""
SYSTEM:
You are an expert feature-library designer for Logical Programmatic Policies (LPP) in grid-based games.
Your task is to REPAIR representational failures in an existing feature set.

{env_line}CONTEXT:
- Observation s is a 2D grid (list of lists of tokens).
- Action a is a clicked cell coordinate (row, col).
- Each feature is a Python function f(s, a) -> bool.
- Features must generalize across board sizes and positions.
- Features depend ONLY on (s, a). No history.

{feat_line}COLLISION EVIDENCE:
All examples below produce IDENTICAL feature vectors under the current feature set (same feature-key),
yet the expert labels differ. Therefore, the current features are provably insufficient.

POSITIVE EXAMPLES (label = 1):
{chr(10).join(pos_blocks)}

NEGATIVE EXAMPLES (label = 0):
{chr(10).join(neg_blocks)}

TASK:
Propose new boolean feature functions that distinguish positives from negatives WITHIN THIS BUCKET.

Each proposed feature must:
1) Be True for most positives and False for most negatives (or vice versa) within this bucket
2) Capture a meaningful semantic distinction (safety, reachability, blocking, threat, progress, etc.)
3) Be board-size invariant (no hard-coded coordinates)
4) Use ONLY (s, a)
5) Return a boolean

DO NOT:
- Hard-code exact positions
- Memorize these examples
- Use random logic
- Output anything other than JSON

OUTPUT FORMAT (STRICT JSON ONLY):

{{
  "features": [
    {{
      "id": "f_new_1",
      "name": "short_descriptive_name",
      "source": "def f_new_1(s, a):\\n    <python code>\\n"
    }}
  ]
}}
""".strip()
    # print(prompt)
    # input("HHMMM")
    return prompt


class LogicProgrammaticPolicyApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that learns a logical programmatic policy from data."""

    def __init__(
        self,
        environment_description: str,  # env_id
        observation_space: Space[_ObsType],
        action_space: Space[_ActType],
        seed: int,
        expert: BaseApproach,
        env_factory: Any,
        base_class_name: str,
        demo_numbers: tuple[int, ...] = (1, 2),
        program_generation_step_size: int = 10,
        num_programs: int = 100,
        num_dts: int = 5,
        max_num_particles: int = 10,
        max_demo_length: int | float = np.inf,
        env_specs: dict[str, Any] | None = None,
        start_symbol: int = 0,
        program_generation: dict[str, Any] | None = None,
    ) -> None:
        """LPP APProach."""
        super().__init__(environment_description, observation_space, action_space, seed)
        self.seed_num = seed
        self.configure_rng()
        self._policy: LPPPolicy | None = None
        self.env_factory = env_factory
        self.base_class_name = base_class_name
        self.expert = expert
        self.demo_numbers = demo_numbers
        self.program_generation_step_size = program_generation_step_size
        self.num_programs = num_programs
        self.num_dts = num_dts
        self.max_num_particles = max_num_particles
        self.max_demo_length = max_demo_length
        self.env_specs = env_specs if env_specs is not None else {}
        self.start_symbol = start_symbol
        self.program_generation = program_generation

    def configure_rng(self) -> None:
        """Seed Python/NumPy RNGs for deterministic rollouts."""
        random.seed(self.seed_num)
        np.random.seed(self.seed_num)

    def reset(self, *args: Any, **kwargs: Any) -> None:
        super().reset(*args, **kwargs)
        self._policy = self._train_policy()
        self._timestep = 0

    def _train_policy(self) -> LPPPolicy:
        """Train the logical programmatic policy using demonstrations."""

        def constant_feature_cols(X_csr):
            n = X_csr.shape[0]  # num examples
            # nnz per column = how many rows have a nonzero entry in that feature
            col_nnz = np.asarray(X_csr.getnnz(axis=0)).ravel()
            all_zero = np.where(col_nnz == 0)[0]
            all_one = np.where(col_nnz == n)[0]  # only valid if X is truly binary 0/1
            return all_zero, all_one, col_nnz

        outer_feedback = None
        programs, program_prior_log_probs, dsl_functions = get_program_set(
            self.num_programs,
            self.base_class_name,
            self.env_factory,
            env_specs=self.env_specs,
            start_symbol=self.start_symbol,
            program_generation=self.program_generation,
            outer_feedback=outer_feedback,
            seed=self.seed_num,
        )

        logging.info("Programs Generation is Done.")
        logging.info(len(programs))

        demonstrations, demo_dict = get_demonstrations(
            self.env_factory, self.expert, demo_numbers=self.demo_numbers
        )

        if self.program_generation is None:
            raise ValueError("program_generation config is required.")
        # n = self.program_generation["num_features"]
        total_features = len(programs)
        new = [f"f{i}(s, a)" for i in range(1, total_features + 1)]  # check space
        programs_sa: list[StateActionProgram] = [StateActionProgram(p) for p in new]
        # programs_sa: list[StateActionProgram] = [StateActionProgram(p) for p in programs]
        X, y, examples = run_all_programs_on_demonstrations(
            self.base_class_name,
            self.demo_numbers,
            programs_sa,
            demo_dict,
            dsl_functions,
            data_imbalance=(self.program_generation or {}).get("data_imbalance"),
            return_examples=True,
        )
        print(len(examples))

        if X is None:
            raise ValueError(
                "X is None. Ensure the program execution results are valid."
            )
        print("prev shape", X.shape)
        print(len(y))
        all_zero, all_one, col_nnz = constant_feature_cols(X)
        print("n_examples:", X.shape[0], "n_features:", X.shape[1])
        print("#all-zero features:", len(all_zero), "indices:", all_zero[:30])
        print("#all-one features:", len(all_one), "indices:", all_one[:30])
        remove = np.unique(np.concatenate([all_zero, all_one]))
        if remove.size > 0:
            keep_mask = np.ones(X.shape[1], dtype=bool)
            keep_mask[remove] = False
            X = X[:, keep_mask]
            programs_sa = [p for i, p in enumerate(programs_sa) if keep_mask[i]]
            if program_prior_log_probs is not None:
                program_prior_log_probs = [
                    lp for i, lp in enumerate(program_prior_log_probs) if keep_mask[i]
                ]
            print("Filtered constant features. New X shape:", X.shape)

        print("after shape", X.shape)

        collision_groups = self._log_feature_collisions(
            X,
            y,
            examples,
        )
        if collision_groups and examples is not None:
            best_group = max(collision_groups, key=lambda g: g["max_occur"])
            prompt = build_collision_repair_prompt(
                pos_indices=best_group["pos"],
                neg_indices=best_group["neg"],
                examples=examples,
                env_name=self.base_class_name,
                existing_feature_summary=None,
                max_per_label=5,
            )
            logging.info("Collision repair prompt built (len=%d).", len(prompt))

        n = X.shape[0]
        print("N", n)
        freq = col_nnz / n  # fraction of examples where feature is on

        rare = np.where(freq <= 0.05)[0]  # almost always 0
        common = np.where(freq >= 0.95)[0]  # almost always 1
        print("Almost-always-0:", len(rare))
        print("Almost-always-1:", len(common))
        self._assert_features_fire(X, programs_sa)

        y_bool: list[bool] = list(y.astype(bool).flatten()) if y is not None else []

        pos = sum(y_bool)
        neg = len(y_bool) - pos
        logging.info(
            "y: n=%d pos=%d (%.2f%%) neg=%d",
            len(y_bool),
            pos,
            100 * pos / len(y_bool),
            neg,
        )

        plps, plp_priors = learn_plps(
            X,
            y_bool,
            programs_sa,
            program_prior_log_probs,
            num_dts=self.num_dts,
            program_generation_step_size=self.program_generation_step_size,
            dsl_functions=dsl_functions,
        )

        print(len(plps))

        filtered: list[tuple[StateActionProgram, float]] = []
        for plp, prior in zip(plps, plp_priors):
            if str(plp).strip() == "False":
                continue
            filtered.append((plp, prior))
        if filtered:
            plps, plp_priors = zip(*filtered)
            plps, plp_priors = list(plps), list(plp_priors)
        else:
            plps, plp_priors = [], []

        # IGNORE = self._log_plp_violation_counts(plps, demonstrations, dsl_functions, top_k=50)
        # for each in IGNORE:
        #     print(each)
        #     print("****")
        # print(len(IGNORE))
        # plps = IGNORE
        likelihoods = compute_likelihood_plps(plps, demonstrations, dsl_functions)
        logging.info(f"LIKELIHOODS: {likelihoods}")
        particles = []
        particle_log_probs = []
        for plp, prior, likelihood in zip(plps, plp_priors, likelihoods):
            particles.append(plp)
            particle_log_probs.append(prior + likelihood)
        print(len(likelihoods))

        logging.info(particle_log_probs)
        map_idx = np.argmax(particle_log_probs).squeeze()
        print(map_idx)
        # map_idx = 0
        logging.info(f"MAP program ({particle_log_probs[map_idx]}):")
        logging.info(particles[map_idx])

        top_particles, top_particle_log_probs = select_particles(
            particles, particle_log_probs, self.max_num_particles
        )
        policy: LPPPolicy
        if len(top_particle_log_probs) > 0:
            top_particle_log_probs = np.array(top_particle_log_probs) - logsumexp(
                top_particle_log_probs
            )
            top_particle_probs = np.exp(top_particle_log_probs)
            logging.info("top_particle_probs: %s", top_particle_probs)
            policy = LPPPolicy(top_particles, top_particle_probs)
            policy.map_program = str(particles[map_idx])
            policy.map_posterior = particle_log_probs[map_idx]
        else:
            logging.info("no nontrivial particles found")
            policy = LPPPolicy([StateActionProgram("False")], [1.0])

        return policy

    @staticmethod
    def _log_feature_collisions(
        X: Any,
        y: np.ndarray | None,
        examples: list[tuple[np.ndarray, tuple[int, int]]] | None,
    ) -> list[dict[str, Any]]:
        """Log collisions where identical feature vectors have different
        labels."""
        if y is None:
            logging.info("Collision check skipped: y is None.")
            return []
        if X is None or X.shape[0] == 0:
            logging.info("Collision check skipped: empty X.")
            return []

        labels = y.astype(int).flatten().tolist()
        collisions: list[tuple[int, int, int]] = []  # (idx_prev, idx_cur, label_prev)
        seen: dict[bytes, tuple[int, int]] = {}  # key -> (index, label)

        for i in range(X.shape[0]):
            row = X.getrow(i)
            dense = row.toarray().ravel().astype(np.uint8)
            key = dense.tobytes()
            label = labels[i]
            if key in seen:
                prev_idx, prev_label = seen[key]
                if prev_label != label:
                    collisions.append((prev_idx, i, prev_label))
            else:
                seen[key] = (i, label)

        if collisions:
            logging.info("Feature collisions found: %d", len(collisions))
            for prev_idx, cur_idx, prev_label in collisions:
                logging.info(
                    "Collision: row %d(label=%d) vs row %d(label=%d)",
                    prev_idx,
                    prev_label,
                    cur_idx,
                    labels[cur_idx],
                )
            return LogicProgrammaticPolicyApproach._group_collision_indices(
                collisions, labels
            )
        else:
            logging.info("No feature collisions found.")
            return []

    @staticmethod
    def _group_collision_indices(
        collisions: list[tuple[int, int, int]],
        labels: list[int],
    ) -> list[dict[str, Any]]:
        """Group collisions into pos/neg index lists and compute max_occur."""
        groups: dict[int, dict[str, set[int]]] = {}
        for prev_idx, cur_idx, prev_label in collisions:
            cur_label = labels[cur_idx]
            if prev_label == cur_label:
                continue
            if prev_label == 1:
                pos_idx, neg_idx = prev_idx, cur_idx
            else:
                pos_idx, neg_idx = cur_idx, prev_idx
            entry = groups.setdefault(pos_idx, {"pos": set(), "neg": set()})
            entry["pos"].add(pos_idx)
            entry["neg"].add(neg_idx)

        out: list[dict[str, Any]] = []
        for pos_idx, data in groups.items():
            pos_list = sorted(data["pos"])
            neg_list = sorted(data["neg"])
            max_occur = max(len(pos_list), len(neg_list))
            out.append({"pos": pos_list, "neg": neg_list, "max_occur": max_occur})
        return out

    @staticmethod
    def _log_plp_violation_counts(
        plps: list[StateActionProgram],
        demonstrations: Any,
        dsl_functions: dict[str, Any],
        top_k: int = 10,
    ) -> list[StateActionProgram]:
        """Log how many demo steps each PLP fails (False on expert action)."""
        set_dsl_functions(dsl_functions)
        counts: list[tuple[int, StateActionProgram]] = []
        total_steps = len(demonstrations.steps)
        print(total_steps)
        print(len(plps))

        for plp in plps:
            violations = 0
            all_obs = []
            all_acts = []
            for obs, action in demonstrations.steps:
                try:
                    if not plp(obs, action):
                        violations += 1
                        all_obs.append(obs)
                        all_acts.append(action)
                except Exception:  # pylint: disable=broad-exception-caught
                    print("EXCEPTION")
                    violations += 1
            counts.append((violations, plp, all_obs, all_acts))

        counts.sort(key=lambda item: item[0])
        logging.info("PLP violation counts (lower is better):")
        for violations, plp, obs_all, act_all in counts[:top_k]:
            rate = (violations / total_steps) if total_steps else 0.0
            logging.info(
                "violations=%d/%d (%.2f%%) | plp=%s",
                violations,
                total_steps,
                100.0 * rate,
                plp,
            )
            for idx, item in enumerate(obs_all):
                print(item)
                actionn = act_all[idx]
                print(actionn)
                print(item[actionn])
        return [plp for _, plp, _, _ in counts[:top_k]]

    @staticmethod
    def _assert_features_fire(X: Any, programs: list[StateActionProgram]) -> None:
        """Assert that every feature fires at least once across all
        examples."""
        if X is None:
            raise AssertionError("X is None; cannot validate feature coverage.")
        if X.shape[1] == 0:
            raise AssertionError("No features found in X.")
        totals = np.asarray(X.sum(axis=0)).ravel()
        dead_idxs = np.where(totals == 0)[0].tolist()
        if dead_idxs:
            dead = [str(programs[i]) for i in dead_idxs]  #:20
            print(f"{len(dead_idxs)} features never fire. Examples: {dead}")

    def test_policy_on_envs(
        self,
        base_class_name: str,
        test_env_nums: Sequence[int] = range(11, 20),
        max_num_steps: int = 50,
        record_videos: bool = True,
        video_format: str | None = "mp4",
    ) -> list[bool]:
        """Train the logical programmatic policy using demonstrations."""
        accuracies = []
        for i in test_env_nums:
            env = self.env_factory(i)
            video_out_path = f"/tmp/lfd_{base_class_name}.{video_format}"
            assert self._policy is not None, "Policy must be trained before testing."
            result = (
                run_single_episode(
                    env,
                    self._policy,
                    max_num_steps=max_num_steps,
                    record_video=record_videos,
                    video_out_path=video_out_path,
                )
                > 0
            )
            accuracies.append(result)
            env.close()
        return accuracies

    def _get_action(self) -> _ActType:
        assert self._policy is not None, "Call reset() first."
        assert self._last_observation is not None
        # Use the logical policy to select an action
        return self._policy(self._last_observation)
