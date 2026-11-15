"""Adapt each alpha into a return stream the no-regret allocator can allocate over.

This is the glue between the two layers. Each alpha is wrapped as a
:class:`~nra.strategies.base.Strategy` that emits a single per-period return
series; :func:`monthly_alpha_panel` resamples them to a common monthly frequency
and aligns them into the panel the learners in :mod:`nra.learners` consume via
:func:`nra.backtest.engine.run_backtest`.

The allocator therefore sits *on top of* the momentum and stat-arb research —
closing the loop from the original project (which allocated over toy strategies)
to allocating over the real alphas built here.
"""

from __future__ import annotations

import pandas as pd

from nra.alphas.momentum import run_momentum
from nra.alphas.statarb import run_statarb
from nra.strategies.base import Strategy


class MomentumStrategy(Strategy):
    """Risk-managed cross-sectional momentum, as a monthly return stream."""

    def __init__(self, **kwargs: object) -> None:
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return "Momentum"

    def compute_returns(self, prices: pd.DataFrame) -> pd.Series:
        result = run_momentum(prices, **self._kwargs)  # type: ignore[arg-type]
        return result.ls_risk_managed.rename(self.name)


class PairsStrategy(Strategy):
    """Out-of-sample stat-arb pairs portfolio, as a daily return stream."""

    def __init__(self, sector_map: dict[str, str], method: str = "ols") -> None:
        self._sector_map = sector_map
        self._method = method

    @property
    def name(self) -> str:
        return "Pairs"

    def compute_returns(self, prices: pd.DataFrame) -> pd.Series:
        result = run_statarb(prices, self._sector_map, method=self._method)
        return result.portfolio_oos.rename(self.name)


class MarketStrategy(Strategy):
    """Equal-weight long-only equity market, as a daily return stream.

    Included as a directional sleeve so the allocator chooses among genuinely
    different return profiles, not just two market-neutral alphas.
    """

    @property
    def name(self) -> str:
        return "Market"

    def compute_returns(self, prices: pd.DataFrame) -> pd.Series:
        return prices.pct_change(fill_method=None).mean(axis=1).rename(self.name)


def _to_monthly(returns: pd.Series) -> pd.Series:
    """Compound a (possibly daily) return series to month-end frequency.

    Idempotent on an already-monthly series, so it is safe to apply uniformly.
    """
    if returns.empty:
        return returns
    return (1.0 + returns).resample("ME").prod() - 1.0


def monthly_alpha_panel(
    prices: pd.DataFrame,
    sector_map: dict[str, str],
    strategies: list[Strategy] | None = None,
) -> pd.DataFrame:
    """Build the aligned monthly return panel of the alphas for the allocator.

    Args:
        prices: Daily equity prices.
        sector_map: Ticker -> sector, for the pairs strategy.
        strategies: Strategies to include; defaults to momentum, pairs, market.

    Returns:
        DataFrame of monthly returns over the window where every stream is live
        (one column per strategy), ready for ``run_backtest``.
    """
    if strategies is None:
        strategies = [
            MomentumStrategy(),
            PairsStrategy(sector_map, method="ols"),
            MarketStrategy(),
        ]
    monthly = [_to_monthly(s.compute_returns(prices)) for s in strategies]
    panel = pd.concat(monthly, axis=1)
    return panel.dropna(how="any")
