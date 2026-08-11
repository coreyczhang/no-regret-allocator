"""Cross-sectional momentum.

Ranks assets by their trailing ``lookback``-period return and holds the top
fraction at equal weight for the next period, betting that recent relative
winners keep winning. The trailing signal is lagged one period so the positions
use only past information.
"""

from __future__ import annotations

import pandas as pd

from nra.strategies.base import Strategy


class Momentum(Strategy):
    """Hold the recent relative winners.

    Args:
        lookback: Number of periods in the trailing-return signal.
        top_fraction: Fraction of assets to hold long each period.
    """

    def __init__(self, lookback: int = 63, top_fraction: float = 0.5) -> None:
        if lookback < 1:
            raise ValueError("lookback must be >= 1.")
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0, 1].")
        self.lookback = int(lookback)
        self.top_fraction = float(top_fraction)

    @property
    def name(self) -> str:
        return f"Momentum(L={self.lookback})"

    def compute_returns(self, prices: pd.DataFrame) -> pd.Series:
        asset_returns = self._asset_returns(prices)
        # Trailing return, lagged one period to avoid look-ahead.
        signal = prices.pct_change(self.lookback).shift(1)
        returns = self._cross_sectional_returns(
            signal, asset_returns, self.top_fraction, prefer_high=True
        )
        return self._named(returns)
