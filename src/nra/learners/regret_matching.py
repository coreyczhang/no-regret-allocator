"""Regret Matching (Hart & Mas-Colell, 2000).

The general-purpose adaptive procedure of

    Hart & Mas-Colell (2000), "A Simple Adaptive Procedure Leading to
    Correlated Equilibrium", Econometrica 68(5).

Reframed for capital allocation: treat each strategy as an action. Had the
learner placed *all* its capital on strategy ``i`` last period it would have
earned ``r_i``; it actually earned ``w . r``. The instantaneous regret for not
having done so is ``g_i = r_i - w . r``, accumulated over time into
``R_i = sum_t g_i^{(t)}``. The learner then allocates next period in proportion
to the positive part of its cumulative regret,

    w_i  =  [R_i]^+ / sum_j [R_j]^+ ,

falling back to uniform weights when no regret is positive. Hart & Mas-Colell
prove this drives the cumulative positive regret per action to be sublinear in
``T`` — the no-regret property — with no learning-rate tuning required, which is
why this learner exposes no ``eta``.
"""

from __future__ import annotations

import numpy as np

from nra.learners.base import FloatArray, Learner
from nra.simplex import uniform_weights


class RegretMatching(Learner):
    """Allocate in proportion to positive cumulative regret.

    Args:
        n: Number of strategies.
        initial_weights: Optional starting allocation; defaults to uniform.
    """

    def __init__(self, n: int, initial_weights: FloatArray | None = None) -> None:
        super().__init__(n, initial_weights)
        self._cumulative_regret: FloatArray = np.zeros(self.n, dtype=np.float64)

    def _reset_state(self) -> None:
        self._cumulative_regret = np.zeros(self.n, dtype=np.float64)

    def _step(self, r: FloatArray) -> FloatArray:
        # Regret for each pure strategy vs. the realized portfolio return.
        realized = float(self._weights @ r)
        self._cumulative_regret += r - realized

        positive = np.maximum(self._cumulative_regret, 0.0)
        total = positive.sum()
        if total <= 0.0:
            # No strategy is regretted so far: default to uniform.
            return uniform_weights(self.n)
        return positive / total
