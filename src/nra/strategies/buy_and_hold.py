"""Equal-weight buy-and-hold.

Invests equal dollar amounts across all assets on day one and never rebalances.
The per-period return is that of the (weight-drifting) equal-dollar portfolio.
It is the passive benchmark of the strategy set.
"""

from __future__ import annotations

import pandas as pd

from nra.strategies.base import Strategy


class BuyAndHold(Strategy):
    """Buy an equal-dollar basket of all assets and hold it."""

    def compute_returns(self, prices: pd.DataFrame) -> pd.Series:
        # Growth of $1 in each asset, then an equal-dollar basket of those.
        growth = prices / prices.iloc[0]
        basket_value = growth.mean(axis=1)
        returns = basket_value.pct_change().iloc[1:]
        return self._named(returns)
