"""Cross-sectional mean reversion.

The mirror image of the momentum rule: rank assets by their trailing
short-horizon return and hold the recent *losers* at equal weight, betting that
short-term underperformers bounce back. The signal is lagged one period.
"""

from __future__ import annotations

import pandas as pd

from nra.strategies.base import Strategy


class MeanReversion(Strategy):
    """Hold the recent relative losers.

    Args:
        lookback: Number of periods in the trailing-return signal (short).
        top_fraction: Fraction of assets to hold long each period.
    """

    def __init__(self, lookback: int = 5, top_fraction: float = 0.5) -> None:
        if lookback < 1:
            raise ValueError("lookback must be >= 1.")
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0, 1].")
        self.lookback = int(lookback)
        self.top_fraction = float(top_fraction)

    @property
    def name(self) -> str:
        return f"MeanReversion(L={self.lookback})"

    def compute_returns(self, prices: pd.DataFrame) -> pd.Series:
        asset_returns = self._asset_returns(prices)
        signal = prices.pct_change(self.lookback).shift(1)
        returns = self._cross_sectional_returns(
            signal, asset_returns, self.top_fraction, prefer_high=False
        )
        return self._named(returns)
