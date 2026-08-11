"""Uniform (equal-weight) baseline.

Holds ``1/N`` in every strategy every period, ignoring realized returns. This is
the constant-rebalanced portfolio at equal weights and a natural sanity-check
benchmark: any adaptive learner worth its complexity should be competitive with
it, and a good one should exploit dispersion across strategies to beat it.
"""

from __future__ import annotations

from nra.learners.base import FloatArray, Learner


class Uniform(Learner):
    """Equal-weight allocation, held constant every period."""

    def _step(self, r: FloatArray) -> FloatArray:
        # Ignore the observed returns; rebalance back to the initial weights.
        return self._initial_weights.copy()
