"""Monthly cross-sectional momentum signal, with no look-ahead.

The classic Jegadeesh & Titman (1993) construction, at monthly frequency:

- Resample daily adjusted closes to month-end prices.
- The formation signal at month ``t`` is the **12-1** momentum: the return over
  months ``t-12 .. t-1``, i.e. ``P_{t-1} / P_{t-12} - 1``. Skipping the most
  recent month (``skip=1``) avoids the well-known short-term reversal that
  contaminates raw 12-month momentum.
- The holding return earned by a portfolio formed at month ``t`` is month
  ``t+1``'s return, aligned back to ``t`` with ``forward_returns`` purely for
  convenient indexing — the signal uses only information available at formation.
"""

from __future__ import annotations

import pandas as pd


def to_monthly_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Resample daily prices to month-end adjusted closes."""
    return prices.resample("ME").last()


def momentum_signal(
    prices: pd.DataFrame, lookback: int = 12, skip: int = 1
) -> pd.DataFrame:
    """Cross-sectional 12-1 momentum signal at monthly frequency.

    Args:
        prices: Daily adjusted close prices (date index, one column per name).
        lookback: Formation window in months.
        skip: Most-recent months to skip (1 removes short-term reversal).

    Returns:
        DataFrame of signal values indexed by month end, one column per name.
        The signal at month ``t`` depends only on prices through month ``t-skip``.
    """
    if lookback <= skip:
        raise ValueError("lookback must exceed skip.")
    monthly = to_monthly_prices(prices)
    return monthly.shift(skip) / monthly.shift(lookback) - 1.0


def forward_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Next-month simple returns, aligned to the formation month.

    The value at month ``t`` is month ``t+1``'s realized return, so a portfolio
    formed from ``momentum_signal`` at ``t`` is scored against the return it
    actually earns over the following month.
    """
    monthly = to_monthly_prices(prices)
    return monthly.pct_change(fill_method=None).shift(-1)
