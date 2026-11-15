"""Cross-sectional equity momentum.

Rank a cross-section of stocks by their trailing return, go long the winners and
short the losers, and measure whether the edge is real and holds up — with
decile monotonicity, the information coefficient, the t-stat of the long/short
spread, turnover, drawdowns, and net-of-cost results.
"""

from nra.alphas.momentum.evaluate import MomentumResult, run_momentum
from nra.alphas.momentum.portfolio import (
    decile_returns,
    long_short_weights,
    quantile_buckets,
)
from nra.alphas.momentum.risk import volatility_scale
from nra.alphas.momentum.signal import (
    forward_returns,
    momentum_signal,
    to_monthly_prices,
)

__all__ = [
    "to_monthly_prices",
    "momentum_signal",
    "forward_returns",
    "quantile_buckets",
    "long_short_weights",
    "decile_returns",
    "volatility_scale",
    "MomentumResult",
    "run_momentum",
]
