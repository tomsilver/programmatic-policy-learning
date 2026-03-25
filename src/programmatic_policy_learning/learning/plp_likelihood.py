"""Fast likelihood computation for Programmatic Logical Policies (PLPs)."""

import inspect
import logging
import multiprocessing
from importlib import import_module
from typing import Any, TypeVar

import cloudpickle
import numpy as np

from programmatic_policy_learning.data.demo_types import Trajectory
from programmatic_policy_learning.dsl.state_action_program import (
    StateActionProgram,
    set_dsl_functions,
)
from programmatic_policy_learning.utils.grid_validation import require_grid_state_action

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")



def _split_dsl(dsl: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Return (base, module_map) — base is pickleable; module_map is
    name→import_path.

    Ensure `out_of_bounds` is included in the DSL.
    """
    base, module_map = {}, {}
    for k, v in dsl.items():
        if inspect.ismodule(v):
            module_map[k] = v.__name__
        else:
            base[k] = v
    base.pop("__builtins__", None)

    return base, module_map


# Global worker state
_WORKER_PLPS = None
_WORKER_CANDIDATE_ACTIONS = None


def likelihood_worker_init(
    dsl_blob: bytes,
    module_map: dict[str, str],
    plp_batch: list[str],
    candidate_actions: list[Any] | None = None,
) -> None:
    """Set up the worker once.

    Loads DSL, reimports modules, and compiles the given PLP batch.
    """
    base = cloudpickle.loads(dsl_blob)
    for name, modpath in module_map.items():
        base[name] = import_module(modpath)
    set_dsl_functions(base)

    global _WORKER_CANDIDATE_ACTIONS, _WORKER_PLPS  # pylint: disable=global-statement
    _WORKER_PLPS = [eval("lambda s, a: " + p, base) for p in plp_batch]
    _WORKER_CANDIDATE_ACTIONS = candidate_actions


def _compute_likelihood_worker(
    demonstrations: Trajectory[_ObsType, _ActType],
) -> list[float]:
    """Compute log-likelihoods for all precompiled PLPs on the given
    demonstrations."""
    global _WORKER_CANDIDATE_ACTIONS, _WORKER_PLPS  # pylint: disable=global-variable-not-assigned
    assert _WORKER_PLPS is not None, "Worker not initialized with PLPs."

    results: list[float] = []
    for plp in _WORKER_PLPS:
        ll = 0.0

        # hyperparams
        # Prevents -inf when expert action is disallowed; models expert noise.
        eps = 1e-4
        beta = 2.0  # >1 penalizes permissiveness more strongly

        for obs, action in demonstrations.steps:
            try:
                obs_grid, action_grid = require_grid_state_action(
                    obs, action, context="_compute_likelihood_worker"
                )
            except TypeError:
                # Continuous/non-grid fallback:
                # enumerate the fixed candidate catalog used at inference time.
                probe_actions = (
                    list(_WORKER_CANDIDATE_ACTIONS)
                    if _WORKER_CANDIDATE_ACTIONS is not None
                    else [action]
                )
                expert_allowed = plp(obs, action)
                size = 0
                for probe_action in probe_actions:
                    if plp(obs, probe_action):
                        size += 1
                size_eff = max(1, size)
                disallowed = max(1, len(probe_actions) - size)
                if expert_allowed:
                    ll += np.log(1.0 - eps) - beta * np.log(size_eff)
                else:
                    ll += np.log(eps) - np.log(disallowed)
                continue

            rows, cols = obs_grid.shape[:2]

            size = 0
            for r in range(rows):
                for c in range(cols):
                    if plp(obs_grid, (r, c)):
                        size += 1

            N = rows * cols  # total actions

            # Numerical safety (avoid log(0))
            size = max(0, min(size, N))

            expert_allowed = plp(obs_grid, action_grid)
            # Likelihood model:
            # - If expert action is allowed: (1-eps) * Uniform(allowed set)
            # - Else: eps * Uniform(disallowed set)
            if expert_allowed:
                # ensure size>=1
                size_eff = max(1, size)
                ll += np.log(1.0 - eps) - beta * np.log(size_eff)
            else:
                disallowed = max(1, N - size)
                ll += np.log(eps) - np.log(disallowed)
        results.append(ll)

    # for plp in _WORKER_PLPS: #PREVIOUS APPROACH
    #     ll = 0.0
    #     valid = True
    #     for obs, action in demonstrations.steps:
    #         if not plp(obs, action):
    #             ll = -np.inf
    #             valid = False
    #             break

    #         rows, cols = obs.shape[:2]
    #         size = 1
    #         for r in range(rows):
    #             for c in range(cols):
    #                 if (r, c) == action:
    #                     continue
    #                 if plp(obs, (r, c)):
    #                     size += 1
    #         ll += np.log(1.0 / size)
    #     results.append(ll if valid else -np.inf)
    return results


def compute_likelihood_plps(
    plps: list[StateActionProgram],
    demonstrations: Trajectory[_ObsType, _ActType],
    dsl_functions: dict[str, Any],
    candidate_actions: list[Any] | None = None,
    plp_interval: int = 100,
    num_workers: int | None = None,
) -> list[float]:
    """Compute log-likelihoods for PLPs given demonstrations.

    Uses fork-based multiprocessing and precompiled PLPs for speed.
    """
    logging.info(f"Computing likelihoods for {len(plps)} PLPs...")

    # Prepare DSL serialization
    base_dsl, module_map = _split_dsl(dsl_functions)
    dsl_blob = cloudpickle.dumps(base_dsl)
    plp_strs = [
        (p.program if isinstance(p, StateActionProgram) else str(p)) for p in plps
    ]

    num_plps = len(plp_strs)

    try:
        ctx: multiprocessing.context.ForkContext = multiprocessing.get_context(
            "fork"
        )  # linux
    except (ValueError, RuntimeError):
        ctx: multiprocessing.context.DefaultContext = (  # type: ignore[no-redef]
            multiprocessing.get_context()
        )  # macOS/Windows fallback (spawn)

    worker_count = max(1, num_workers or multiprocessing.cpu_count())
    likelihoods_all: list[float] = []

    for p_start in range(0, num_plps, plp_interval):
        p_end = min(p_start + plp_interval, num_plps)
        plp_batch = plp_strs[p_start:p_end]

        with ctx.Pool(
            processes=worker_count,
            initializer=likelihood_worker_init,
            initargs=(dsl_blob, module_map, plp_batch, candidate_actions),
        ) as pool:

            # Each worker runs likelihood for its PLP batch on the same demonstrations
            results_iter = pool.imap(
                _compute_likelihood_worker, [demonstrations], chunksize=1
            )
            batch_lls = list(results_iter)[0]
            likelihoods_all.extend(batch_lls)

    return likelihoods_all
