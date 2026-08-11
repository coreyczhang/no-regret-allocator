"""Volatility-targeting overlay on the equal-weight basket.

Takes the equal-weight portfolio of all assets and scales exposure up or down to
aim for a constant target volatility: when recent realized volatility is high,
exposure is cut; when it is calm, exposure rises toward fully invested. Exposure
is capped at 1 (long-only, no leverage) with the unused fraction earning nothing
(held as cash). The volatility estimate is lagged one period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nra.strategies.base import Strategy


class VolatilityTarget(Strategy):
    """Scale an equal-weight basket toward a constant target volatility.

    Args:
        target_annual_vol: Desired annualized volatility of the strategy.
        window: Rolling window (periods) for the realized-volatility estimate.
        periods_per_year: Periods per year, for annualization (252 ~ daily).
    """

    def __init__(
        self,
        target_annual_vol: float = 0.10,
        window: int = 20,
        periods_per_year: int = 252,
    ) -> None:
        if target_annual_vol <= 0:
            raise ValueError("target_annual_vol must be positive.")
        if window < 2:
            raise ValueError("window must be >= 2.")
        self.target_annual_vol = float(target_annual_vol)
        self.window = int(window)
        self.periods_per_year = int(periods_per_year)

    @property
    def name(self) -> str:
        return f"VolTarget({self.target_annual_vol:g})"

    def compute_returns(self, prices: pd.DataFrame) -> pd.Series:
        asset_returns = self._asset_returns(prices)
        basket = asset_returns.mean(axis=1)

        per_period_target = self.target_annual_vol / np.sqrt(self.periods_per_year)
        # Trailing realized vol, lagged so exposure uses only past information.
        trailing_vol = basket.rolling(self.window).std().shift(1)
        exposure = (per_period_target / trailing_vol).clip(upper=1.0)

        returns = (exposure * basket).dropna()
        return self._named(returns)
