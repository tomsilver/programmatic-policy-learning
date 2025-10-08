"""Functions for computing log likelihoods of programmatic logical policies
(PLPs) given demonstrations."""

import multiprocessing
from functools import partial

import numpy as np

from programmatic_policy_learning.data.demo_types import Trajectory
from programmatic_policy_learning.dsl.state_action_program import StateActionProgram


def compute_likelihood_single_plp(
    demonstrations: Trajectory[np.ndarray, tuple[int, int]], plp: StateActionProgram
) -> float:
    """Compute the log likelihood of a single PLP given demonstrations.

    Parameters
    ----------
    demonstrations : [(np.ndarray, (int, int))]
        State, action pairs.
    plp : StateActionProgram

    Returns
    -------
    likelihood : float
        The log likelihood.
    """
    ll = 0.0

    for obs, action in demonstrations.steps:
        if not plp(obs, action):
            return -np.inf

        size = 1
        for r in range(obs.shape[0]):
            for c in range(obs.shape[1]):
                if (r, c) == action:
                    continue
                if plp(obs, (r, c)):
                    size += 1

        ll += np.log(1.0 / size)

    return ll


def compute_likelihood_plps(
    plps: list[StateActionProgram],
    demonstrations: Trajectory[np.ndarray, tuple[int, int]],
) -> list[float]:
    """Compute log likelihoods for a list of PLPs given demonstrations.

    See compute_likelihood_single_plp.
    """
    num_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_workers)

    fn = partial(compute_likelihood_single_plp, demonstrations)
    likelihoods = pool.map(fn, plps)
    pool.close()

    return likelihoods
