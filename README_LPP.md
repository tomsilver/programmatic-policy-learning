# Programmatic Policy Learning (LPP-Focused README)

This repository contains multiple programmatic policy learning approaches.  
This README is focused on the LPP work and the recent contributions around:

- LPP reimplementation and integration into this codebase
- GGG strategic grid environments
- LLM/VLM-assisted feature and hint generation

LPP here refers to a logical programmatic policy pipeline inspired by the Few-Shot Bayesian imitation learning line of work, where candidate logical features/programs are learned from demonstrations and composed into executable policies.

## Installation

Requirements:
- Python `>=3.10` and `<=3.12`
- `OPENAI_API_KEY` set if you run LLM/VLM components

Recommended:

```bash
uv pip install -e ".[develop]"
```

Sanity check:

```bash
./run_ci_checks.sh
```

## What Happens When `approach=lpp`

Hydra instantiates `LogicProgrammaticPolicyApproach` from `experiments/conf/approach/lpp.yaml` via `experiments/run_experiment.py`.

At a high level, LPP does this:

1. Generate candidate programs/features (`fixed_grid_v1`, `dsl_generator`, `feature_generator`, `py_feature_gen`, or offline loader path).
2. Collect expert demonstrations from selected environment instances.
3. Build a classification dataset over `(state, action)` positives/negatives.
4. Evaluate all candidate programs on demos to build sparse feature matrix `X`.
5. Learn decision trees and convert paths to logical formulas (PLPs).
6. Score PLPs by prior + likelihood, keep top particles, build `LPPPolicy`.

Primary implementation:
- `src/programmatic_policy_learning/approaches/lpp_approach.py`

## Running LPP

Single run:

```bash
python3 experiments/run_experiment.py env=ggg_stf approach=lpp seed=0
```

Multi-run with Hydra sweep:

```bash
python3 experiments/run_experiment.py -m env=ggg_stf approach=lpp seed='range(0,5)'
```

Run with offline LLM JSON payload (through py-feature generation path):

```bash
python3 experiments/run_experiment.py env=ggg_stf approach=lpp seed=0 \
  approach.program_generation.loading.offline=1 \
  approach.program_generation.loading.offline_json_path=test_stf.json
```

Sweep version of offline run:

```bash
python3 experiments/run_experiment.py -m env=ggg_stf approach=lpp seed='range(0,5)' \
  approach.program_generation.loading.offline=1 \
  approach.program_generation.loading.offline_json_path=test_stf.json
```

## `lpp.yaml` Guide

File: `experiments/conf/approach/lpp.yaml`

Key fields:
- `_target_`: LPP approach class Hydra instantiates.
- `demo_numbers`: training demonstration env indices.
- `program_generation_step_size`: feature growth schedule during DT learning.
- `num_programs`: max generated programs for grammar-based modes.
- `num_dts`: number of decision trees per batch.
- `max_num_particles`: posterior particles kept for policy.
- `program_generation.strategy`: core generation mode.
- `program_generation.py_feature_gen_prompt`: prompt template for Python features.
- `program_generation.py_feature_gen_batch_prompt`: follow-up batch prompt template.
- `program_generation.num_features`: number of candidate features to ask for.
- `program_generation.loading.offline`: `1` loads JSON from disk, `0` queries LLM.
- `program_generation.loading.offline_json_path`: JSON path used when offline.

Current main path is `strategy: py_feature_gen`.

## Repository Structure (Onboarding View)

Top-level:
- `experiments/`: Hydra configs and experiment entrypoint.
- `src/`: main implementation.
- `tests/`: mirrored test structure by subsystem.
- `data/`: static assets (for example maze files).

### `experiments/`

- `experiments/run_experiment.py`: Hydra entrypoint, environment loading, approach instantiation, evaluation loop.
- `experiments/conf/config.yaml`: global defaults and Hydra run/sweep output config.
- `experiments/conf/env/ggg_*.yaml`: GGG task configs.

GGG environments in this repo:
- `ggg_stf` -> `StopTheFall`
- `ggg_nim` -> `TwoPileNim`
- `ggg_rfts` -> `ReachForTheStar`
- `ggg_chase` -> `Chase`
- `ggg_checkmate` -> `CheckmateTactic`

### `src/programmatic_policy_learning/approaches/`

- `lpp_approach.py`: core LPP pipeline and training loop.
- Hydra creates this class when `approach=lpp`.

### `src/programmatic_policy_learning/data/`

- `collect.py`: collects expert demonstrations into trajectories.
- `dataset.py`: builds positive/negative `(s,a)` examples and evaluates candidate programs over demos.
- Includes data-imbalance support (for example downsampling negatives).

### `src/programmatic_policy_learning/dsl/` (most important subsystem)

#### `primitives_sets/`

- `grid_v1.py`: fixed DSL primitives and grammar source for grid tasks.
- Defines DSL primitives, grammar nonterminals, and object-type-aware value productions.

#### `llm_primitives/` (major contribution area)

- `baselines/`
- `hint_generation/`
- `prompts/`
- `dsl_evaluator.py`
- `feature_generator.py` (legacy feature generation path)
- `py_feature_generator.py` (current main feature generation path)

#### `llm_primitives/baselines/`

LLM baseline:
- `src/programmatic_policy_learning/dsl/llm_primitives/baselines/llm_based/CaP_baseline.py`
- CLI args are defined in `_parse_cli_args`.
- Example command:

```bash
python -m programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based.CaP_baseline \
  --env StopTheFall --encodings 4 --seeds 0
```

Note:
- The current `main()` includes a `manual_eval(...)` debug call before batch flow. If you want full batch execution, remove/disable that debug call.

VLM baseline:
- `src/programmatic_policy_learning/dsl/llm_primitives/baselines/vlm_based/video_frames_hint_extractor.py`
- Update `env_name` (and paths if needed) in `__main__`, then run:

```bash
python src/programmatic_policy_learning/dsl/llm_primitives/baselines/vlm_based/video_frames_hint_extractor.py
```

#### `llm_primitives/prompts/`

Prompt directories:
- `prompts/dsl_generation/`
- `prompts/feature_generation/` (deprecated)
- `prompts/py_feature_gen/` (current)

Current work uses `py_feature_gen` prompts to request executable Python feature functions from an LLM.

#### `llm_primitives/hint_generation/`

LLM-based:
- `hint_generation/llm_based/hint_extractor.py`
- `hint_generation/llm_based/hint_aggregator.py`
- Purpose: extract compact hint text from demonstrations first, then use hints in later generation stages (instead of embedding long demos directly in one big prompt).

Run extractor:

```bash
python -m programmatic_policy_learning.dsl.llm_primitives.hint_generation.llm_based.hint_extractor
```

Run aggregator:

```bash
python -m programmatic_policy_learning.dsl.llm_primitives.hint_generation.llm_based.hint_aggregator
```

VLM-based:
- `hint_generation/vlm_based/video_frames_hint_extractor.py`
- Edit `env_name` in `__main__` and run:

```bash
python src/programmatic_policy_learning/dsl/llm_primitives/hint_generation/vlm_based/video_frames_hint_extractor.py
```

#### `dsl_evaluator.py`

Used in DSL-generation experiments to partially evaluate candidate DSLs during generation before sending to downstream stages.

#### `feature_generator.py` (legacy)

Legacy feature generation path (older prompting setup). Kept for experiments and backward comparison.

#### `py_feature_generator.py` (current)

Main current contribution path for LPP feature generation.

- Builds prompts from templates.
- Queries LLM (or loads offline JSON).
- Parses feature source code.
- Returns Python feature functions consumed by LPP.

Triggered in `lpp_approach.py` when:
- `program_generation.strategy == "py_feature_gen"`

If you want offline evaluation, set in config:

```yaml
loading:
  offline: 1
  offline_json_path: "test_stf.json"
```

### `src/programmatic_policy_learning/dsl/state_action_program.py`

`StateActionProgram` wraps program strings to:
- provide clean `str`/`repr`
- support pickling
- lazily compile cached callable forms
- reduce redundant evals

### `src/programmatic_policy_learning/envs/`

- `registry.py`: provider registry and fallback loader.
- `providers/ggg_provider.py`: GGG adapter + object type exposure.
- `providers/prbench_provider.py`: PRBench adapter.
- `providers/maze_provider.py`: custom maze environment provider.

Provider logic:
- If `provider` is set in env config, `EnvRegistry` routes to the provider function.
- Otherwise it falls back to `gymnasium.make(**make_kwargs)`.

### `src/programmatic_policy_learning/learning/`

Short descriptions:
- `decision_tree_learner.py`: train decision trees and extract logical PLPs.
- `plp_likelihood.py`: compute PLP log-likelihood on demonstrations.
- `particles_utils.py`: select top particles by posterior score.
- `prior_calculation.py`: prior-related utilities.

### `src/programmatic_policy_learning/policies/`

- `lpp_policy.py`: policy wrapper used by LPP in RL-style env interaction (`reset`, `step`, cached action probabilities, MAP/sampling behavior).

### `tests/`

Tests mirror source subsystems:
- `tests/approaches/`
- `tests/data/`
- `tests/dsl/`
- `tests/env/`
- `tests/learning/`
- `tests/policies/`

## Environment Provider Notes

Use provider-backed envs when env construction needs custom logic or external repos.

Current providers in `EnvRegistry`:
- `ggg`
- `prbench`
- `maze`

For new provider integration:
1. Add env yaml under `experiments/conf/env/`.
2. Add provider function in `src/programmatic_policy_learning/envs/providers/`.
3. Register provider in `src/programmatic_policy_learning/envs/registry.py`.

## Contribution Summary (LPP Track)

This LPP track contributes:
- end-to-end `lpp_approach` integration in Hydra experiment flow
- GGG-focused programmatic imitation pipeline
- LLM/VLM hint and feature generation modules
- py-feature generation workflow with offline replay support
- supporting DSL, policy, and dataset pipeline updates
