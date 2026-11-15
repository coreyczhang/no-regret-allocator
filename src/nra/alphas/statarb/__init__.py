"""Cointegration-based statistical-arbitrage pairs trading.

Find cointegrated stock pairs, model the spread, and trade its mean reversion —
with the discipline that makes or breaks a pairs study: **pairs are selected and
hedge ratios estimated strictly in-sample, then traded out-of-sample**, so the
reported edge is not an artifact of having picked the pairs with hindsight.

Pipeline:
- :mod:`~nra.alphas.statarb.selection` — within-sector correlation screen then
  Engle-Granger cointegration test (in-sample only).
- :mod:`~nra.alphas.statarb.hedge` — static OLS and dynamic Kalman-filter hedge
  ratios.
- :mod:`~nra.alphas.statarb.spread` — spread z-score and the Ornstein-Uhlenbeck
  half-life of mean reversion.
- :mod:`~nra.alphas.statarb.signals` — z-score entry/exit/stop and pair PnL.
- :mod:`~nra.alphas.statarb.evaluate` — the in-sample/out-of-sample backtest.
"""

from nra.alphas.statarb.evaluate import StatArbResult, run_statarb
from nra.alphas.statarb.hedge import kalman_hedge_ratio, ols_hedge_ratio
from nra.alphas.statarb.selection import CandidatePair, find_cointegrated_pairs
from nra.alphas.statarb.signals import generate_positions, pair_returns
from nra.alphas.statarb.spread import compute_spread, half_life, zscore

__all__ = [
    "CandidatePair",
    "find_cointegrated_pairs",
    "ols_hedge_ratio",
    "kalman_hedge_ratio",
    "compute_spread",
    "zscore",
    "half_life",
    "generate_positions",
    "pair_returns",
    "StatArbResult",
    "run_statarb",
]
