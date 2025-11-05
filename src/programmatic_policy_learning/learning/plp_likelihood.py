"""Fast likelihood computation for Programmatic Logical Policies (PLPs)."""

import inspect
import logging
import multiprocessing
from importlib import import_module
from typing import Any

import cloudpickle
import numpy as np

from programmatic_policy_learning.data.demo_types import Trajectory
from programmatic_policy_learning.dsl.state_action_program import (
    StateActionProgram,
    set_dsl_functions,
)


def _split_dsl(dsl: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Return (base, module_map) — base is pickleable; module_map is
    name→import_path."""
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


def likelihood_worker_init(
    dsl_blob: bytes, module_map: dict[str, str], plp_batch: list[str]
) -> None:
    """Set up the worker once.

    Loads DSL, reimports modules, and compiles the given PLP batch.
    """
    base = cloudpickle.loads(dsl_blob)
    for name, modpath in module_map.items():
        base[name] = import_module(modpath)
    set_dsl_functions(base)

    global _WORKER_PLPS  # pylint: disable=global-statement
    _WORKER_PLPS = [eval("lambda s, a: " + p, base) for p in plp_batch]

def _compute_likelihood_worker(
    demonstrations: Trajectory[np.ndarray, tuple[int, int]],
) -> list[float]:
    """Compute log-likelihoods for all precompiled PLPs efficiently."""
    global _WORKER_PLPS
    assert _WORKER_PLPS is not None, "Worker not initialized."

    steps = demonstrations.steps
    obs_list = [obs for obs, _ in steps]
    act_list = [act for _, act in steps]

    results = []
    for plp in _WORKER_PLPS:
        ll = 0.0
        valid = True

        # precompute all plp(obs, r, c)
        obs_masks = [
            np.array([[plp(obs, (r, c)) for c in range(obs.shape[1])]
                      for r in range(obs.shape[0])], dtype=bool)
            for obs in obs_list
        ]

        for mask, act in zip(obs_masks, act_list):
            if not mask[act]:
                ll = -np.inf
                valid = False
                break
            size = int(mask.sum())
            ll += -np.log(size)

        results.append(ll if valid else -np.inf)

    return results

# def _compute_likelihood_worker(
#     demonstrations: Trajectory[np.ndarray, tuple[int, int]],
# ) -> list[float]:
#     """Compute log-likelihoods for all precompiled PLPs on given demos."""
#     global _WORKER_PLPS
#     assert _WORKER_PLPS is not None, "Worker not initialized."

#     steps = demonstrations.steps
#     # Precompute coordinates excluding each action
#     coords = [
#         [(r, c) for r in range(obs.shape[0]) for c in range(obs.shape[1]) if (r, c) != act]
#         for obs, act in steps
#     ]

#     results = []
#     for plp in _WORKER_PLPS:
#         ll = 0.0
#         for (obs, act), cell_list in zip(steps, coords):
#             if not plp(obs, act):
#                 ll = -np.inf
#                 break
#             # Count valid alternative actions
#             size = 1 + sum(plp(obs, rc) for rc in cell_list)
#             ll += -np.log(size)
#         results.append(ll)
#     return results

# def _compute_likelihood_worker(
#     demonstrations: Trajectory[np.ndarray, tuple[int, int]],
# ) -> list[float]:
#     """Compute log-likelihoods for all precompiled PLPs on the given
#     demonstrations."""
#     global _WORKER_PLPS  # pylint: disable=global-variable-not-assigned
#     assert _WORKER_PLPS is not None, "Worker not initialized with PLPs."

#     results: list[float] = []
#     for plp in _WORKER_PLPS:
#         ll = 0.0
#         valid = True
#         for obs, action in demonstrations.steps:
#             if not plp(obs, action):
#                 ll = -np.inf
#                 valid = False
#                 break

#             rows, cols = obs.shape[:2]
#             size = 1
#             for r in range(rows):
#                 for c in range(cols):
#                     if (r, c) == action:
#                         continue
#                     if plp(obs, (r, c)):
#                         size += 1
#             ll += np.log(1.0 / size)
#         results.append(ll if valid else -np.inf)
#     return results


def compute_likelihood_plps(
    plps: list[StateActionProgram],
    demonstrations: Trajectory[np.ndarray, tuple[int, int]],
    dsl_functions: dict[str, Any],
    plp_interval: int = 100,
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

    num_workers = max(1, multiprocessing.cpu_count())
    likelihoods_all: list[float] = []

    for p_start in range(0, num_plps, plp_interval):
        p_end = min(p_start + plp_interval, num_plps)
        plp_batch = plp_strs[p_start:p_end]

        with ctx.Pool(
            processes=num_workers,
            initializer=likelihood_worker_init,
            initargs=(dsl_blob, module_map, plp_batch),
        ) as pool:
            # Each worker runs likelihood for its PLP batch on the same demonstrations
            results_iter = pool.imap(
                _compute_likelihood_worker, [demonstrations], chunksize=1
            )
            batch_lls = list(results_iter)[0]
            likelihoods_all.extend(batch_lls)

    return likelihoods_all
