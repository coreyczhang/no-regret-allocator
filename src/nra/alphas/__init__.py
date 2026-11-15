"""Rigorous standalone alpha-research strategies.

Two self-contained strategies, each evaluated on its own terms before being
wrapped (via :mod:`nra.adapters`) as a return stream the no-regret allocator can
allocate over:

- :mod:`nra.alphas.momentum` — cross-sectional equity momentum (long winners,
  short losers), judged by decile monotonicity, information coefficient, and the
  t-stat of the long/short spread.
- :mod:`nra.alphas.statarb` — cointegration-based pairs trading, judged with a
  strict in-sample-selection / out-of-sample-trading split.

Both are research/educational implementations, not investment advice.
"""
