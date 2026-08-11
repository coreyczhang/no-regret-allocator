"""Exponentiated Gradient / Multiplicative Weights.

The classical multiplicative-weights update applied to online portfolio
selection. After observing per-strategy returns ``r``, each weight is scaled by
a factor exponential in that strategy's return and the vector is renormalized:

    w_i  <-  w_i * exp(eta * r_i)   then   w <- w / sum(w).

This is the Exponentiated Gradient (EG) algorithm of Helmbold, Schapire, Singer
& Warmuth (1998), which realizes the multiplicative-weights / Hedge strategy on
the simplex and is the practical online cousin of Cover's (1991) Universal
Portfolios. It reacts quickly to a strategy that starts outperforming (weight
grows geometrically), which is powerful but also makes it the highest-turnover
of the no-regret learners here. With a learning rate on the order of
``eta ~ sqrt(8 ln N / T)`` its regret against the best fixed allocation is
``O(sqrt(T ln N))`` — sublinear, hence "no regret".
"""

from __future__ import annotations

import numpy as np

from nra.learners.base import FloatArray, Learner


class ExponentiatedGradient(Learner):
    """Multiplicative-weights allocation update.

    Args:
        n: Number of strategies.
        eta: Learning rate (>= 0). ``eta = 0`` freezes the weights, so a learner
            initialized uniformly stays uniform.
        initial_weights: Optional starting allocation; defaults to uniform.
    """

    def __init__(
        self,
        n: int,
        eta: float = 0.05,
        initial_weights: FloatArray | None = None,
    ) -> None:
        if eta < 0:
            raise ValueError("eta must be non-negative.")
        super().__init__(n, initial_weights)
        self.eta = float(eta)

    @property
    def name(self) -> str:
        return f"ExpGradient(eta={self.eta:g})"

    def _step(self, r: FloatArray) -> FloatArray:
        # Multiply by exp(eta * r); subtract the max exponent first for numerical
        # stability. The shift cancels under renormalization, so weights are
        # unchanged in exact arithmetic.
        exponent = self.eta * r
        exponent -= exponent.max()
        w = self._weights * np.exp(exponent)
        total = w.sum()
        if total <= 0.0 or not np.isfinite(total):
            # Degenerate case (should not happen for finite returns); fall back
            # to the current weights rather than emitting an invalid vector.
            return self._weights.copy()
        return w / total
