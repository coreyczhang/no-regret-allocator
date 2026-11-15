"""Transaction-cost model for the momentum portfolio.

Costs are charged proportional to turnover — the L1 change in weights from one
rebalance to the next — at a flat per-unit-traded rate. This is the standard
first-order cost model; it deliberately ignores market impact and shorting/borrow
costs, which are called out as limitations in the README.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def turnover_series(weights: pd.DataFrame) -> pd.Series:
    """L1 turnover entering each rebalance, ``sum_i |w_t - w_{t-1}|``.

    The first period's turnover is the cost of putting the initial book on.
    """
    prev = weights.shift(1).fillna(0.0)
    return (weights - prev).abs().sum(axis=1)


def apply_costs(
    gross_returns: pd.Series, weights: pd.DataFrame, cost_bps: float = 10.0
) -> pd.Series:
    """Subtract turnover-proportional costs from gross returns.

    Args:
        gross_returns: Per-period gross portfolio returns.
        weights: Per-period weights used to compute turnover.
        cost_bps: Cost per unit of turnover, in basis points (10 bps = 0.1%).

    Returns:
        Net per-period returns (``gross - cost_bps/1e4 * turnover``).
    """
    turnover = turnover_series(weights).reindex(gross_returns.index).fillna(0.0)
    cost = (cost_bps / 1e4) * turnover
    return (gross_returns - cost).astype(np.float64)
