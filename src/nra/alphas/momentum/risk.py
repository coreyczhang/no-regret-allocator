"""Volatility-targeting overlay — risk-managed momentum.

Momentum's defining failure mode is the *momentum crash*: in the panicked
rebound out of a bear market (spring 2009 is the canonical case) the short leg —
beaten-down high-beta losers — rips higher and the strategy suffers a sudden,
severe loss. Daniel & Moskowitz (2016) show these crashes are partly forecastable
from the strategy's own realized volatility, so scaling exposure to target a
constant volatility sharply reduces the left tail.

This overlay scales each period's strategy return by
``target / trailing_realized_vol`` (exposure capped), using only past volatility.
It cannot create a premium that is not there, but it reliably tames the crash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def volatility_scale(
    returns: pd.Series,
    target_annual_vol: float = 0.12,
    window: int = 6,
    cap: float = 2.0,
    periods_per_year: int = 12,
) -> pd.Series:
    """Scale a return stream toward a constant target volatility.

    Args:
        returns: Per-period strategy returns.
        target_annual_vol: Desired annualized volatility.
        window: Trailing window (periods) for the realized-vol estimate.
        cap: Maximum gross exposure (leverage) allowed.
        periods_per_year: Periods per year, for de-annualizing the target.

    Returns:
        The risk-managed return stream (exposure uses only past volatility).
    """
    per_period_target = target_annual_vol / np.sqrt(periods_per_year)
    trailing_vol = returns.rolling(window).std().shift(1)
    exposure = (per_period_target / trailing_vol).clip(upper=cap).fillna(0.0)
    return (exposure * returns).astype(np.float64)
