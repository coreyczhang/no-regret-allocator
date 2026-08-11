"""Naive "chase last period's winner" baseline.

Each period it dumps all capital into whichever single strategy had the highest
realized return in the *previous* period. This is a deliberately naive heuristic:
it is greedy, has maximal turnover, and provably has linear regret on adversarial
inputs (an environment can flip the winner every period so the chaser is always
one step behind). It exists to show that the no-regret learners add real value
over the obvious "just ride the hot hand" rule.
"""

from __future__ import annotations

import numpy as np

from nra.learners.base import FloatArray, Learner


class FollowTheWinner(Learner):
    """Put full weight on the previous period's best-performing strategy."""

    def _step(self, r: FloatArray) -> FloatArray:
        w = np.zeros(self.n, dtype=np.float64)
        # np.argmax breaks ties toward the lowest index, which is deterministic.
        w[int(np.argmax(r))] = 1.0
        return w
