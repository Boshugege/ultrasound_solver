"""Target-response phase alignment."""

import numpy as np

from algorithm.baselines.common import wrap


def phases_for_targets(problem):
    if not problem.target_indices:
        raise ValueError("Response alignment requires target points")
    if problem.target_response is not None:
        return wrap(np.angle(np.sum(problem.target_response.conj(), axis=0)))
    response = np.zeros(problem.size, dtype=np.complex128)
    for index in problem.target_indices:
        row = np.ravel_multi_index(index, problem.target.target.shape, order="F")
        response += np.conjugate(problem.basis.basis[row, :])
    return wrap(np.angle(response))


def solve(problem, options):
    final = problem.evaluate(phases_for_targets(problem))
    return problem.result(final, "analytic")
