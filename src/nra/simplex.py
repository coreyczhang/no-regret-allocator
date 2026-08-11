"""Probability-simplex utilities shared by the learners.

The allocation weight vector always lives on the probability simplex

    Delta_N = { w in R^N : w_i >= 0, sum_i w_i = 1 },

i.e. long-only, fully invested. Several learners take an unconstrained step and
then need to project back onto Delta_N; that Euclidean projection is
implemented here with the classic sort-based algorithm of

    Wang & Carreira-Perpinan (2013), "Projection onto the probability simplex",
    which is equivalent to the projection step in
    Duchi, Shalev-Shwartz, Singer & Chandra (2008), "Efficient projections
    onto the l1-ball for learning in high dimensions".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def euclidean_projection_simplex(v: FloatArray) -> FloatArray:
    """Return the Euclidean projection of ``v`` onto the probability simplex.

    Solves ``argmin_{w in Delta_N} ||w - v||_2`` in ``O(N log N)`` via sorting.

    Args:
        v: A 1-D vector of length ``N``.

    Returns:
        The projected vector ``w`` on the simplex (non-negative, sums to 1).
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    n = v.shape[0]
    if n == 0:
        raise ValueError("Cannot project a length-0 vector onto the simplex.")
    # Sort in descending order.
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1, dtype=np.float64)
    # Largest rho such that u[rho] - (cumsum(u)[rho] - 1) / (rho + 1) > 0.
    cond = u - cssv / ind > 0
    rho = np.nonzero(cond)[0][-1]
    theta = cssv[rho] / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w


def is_on_simplex(w: FloatArray, atol: float = 1e-9) -> bool:
    """Check that ``w`` is a valid simplex point: non-negative and sums to 1."""
    w = np.asarray(w, dtype=np.float64)
    return bool(np.all(w >= -atol) and abs(float(w.sum()) - 1.0) <= atol)


def uniform_weights(n: int) -> FloatArray:
    """Return the uniform allocation over ``n`` strategies."""
    if n <= 0:
        raise ValueError("n must be positive.")
    return np.full(n, 1.0 / n, dtype=np.float64)
