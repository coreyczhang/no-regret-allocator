"""Online no-regret learning algorithms for capital allocation.

Every learner shares the :class:`~nra.learners.base.Learner` interface: it holds
a weight vector on the probability simplex and, given the realized per-strategy
returns of a period, produces the weights to use for the next period.
"""

from nra.learners.base import Learner
from nra.learners.uniform import Uniform
from nra.learners.follow_the_winner import FollowTheWinner
from nra.learners.exp_gradient import ExponentiatedGradient
from nra.learners.ogd import OnlineGradientDescent
from nra.learners.regret_matching import RegretMatching

__all__ = [
    "Learner",
    "Uniform",
    "FollowTheWinner",
    "ExponentiatedGradient",
    "OnlineGradientDescent",
    "RegretMatching",
]
