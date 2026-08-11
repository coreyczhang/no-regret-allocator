"""Online Gradient Descent on the simplex.

The learner wants to maximize its cumulative portfolio return. Over a single
period with realized returns ``r``, the reward of weight vector ``w`` is the
linear function ``w . r``, whose gradient with respect to ``w`` is simply ``r``.
Online Gradient Descent (Zinkevich, 2003) therefore takes an ascent step in the
direction of ``r`` and projects the result back onto the probability simplex:

    w  <-  Proj_simplex( w + eta * r ).

Compared with multiplicative weights, the additive step plus Euclidean
projection makes OGD move more gently and rebalance less aggressively (lower
turnover), at the cost of adapting more slowly to a newly dominant strategy.
For a bounded gradient, ``eta ~ 1/sqrt(T)`` yields ``O(sqrt(T))`` regret against
the best fixed allocation.
"""

from __future__ import annotations

from nra.learners.base import FloatArray, Learner
from nra.simplex import euclidean_projection_simplex


class OnlineGradientDescent(Learner):
    """Projected online gradient ascent on portfolio return.

    Args:
        n: Number of strategies.
        eta: Step size (>= 0). ``eta = 0`` freezes the weights.
        initial_weights: Optional starting allocation; defaults to uniform.
    """

    def __init__(
        self,
        n: int,
        eta: float = 0.1,
        initial_weights: FloatArray | None = None,
    ) -> None:
        if eta < 0:
            raise ValueError("eta must be non-negative.")
        super().__init__(n, initial_weights)
        self.eta = float(eta)

    @property
    def name(self) -> str:
        return f"OGD(eta={self.eta:g})"

    def _step(self, r: FloatArray) -> FloatArray:
        # Ascent step on the reward w . r (gradient = r), then project.
        return euclidean_projection_simplex(self._weights + self.eta * r)
