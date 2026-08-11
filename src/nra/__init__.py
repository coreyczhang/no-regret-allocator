"""No-Regret Allocator.

An online no-regret learning framework for adaptively allocating capital across
a set of candidate trading strategies. The learner observes each strategy's
realized return every period and reweights capital so that its cumulative
performance provably approaches that of the best fixed-weight allocation in
hindsight, without knowing in advance which strategy will do best.

This is a research / educational implementation. Nothing here is investment
advice, and the bundled toy strategies are illustrative, not trading ideas.
"""

from nra.simplex import euclidean_projection_simplex, is_on_simplex

__all__ = ["euclidean_projection_simplex", "is_on_simplex", "__version__"]

__version__ = "0.1.0"
