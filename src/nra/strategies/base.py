"""Common interface for the toy strategies.

A :class:`Strategy` maps a price panel to a per-period return series. To avoid
look-ahead bias every strategy forms its positions from information available
*strictly before* the period whose return it collects: positions decided at the
close of day ``t-1`` earn the asset returns realized over day ``t``.
"""

from __future__ import annotations

import abc

import numpy as np
import pandas as pd


class Strategy(abc.ABC):
    """Abstract base class for a toy trading strategy.

    Subclasses implement :meth:`compute_returns`, returning a return series
    aligned to (a subset of) the input price index.
    """

    @property
    def name(self) -> str:
        """Human-readable strategy name (used as the return series' name)."""
        return type(self).__name__

    @abc.abstractmethod
    def compute_returns(self, prices: pd.DataFrame) -> pd.Series:
        """Return the strategy's realized per-period simple returns.

        Args:
            prices: DataFrame of adjusted close prices indexed by date, one
                column per asset.

        Returns:
            A Series of per-period simple returns named after the strategy.
        """

    # --- Shared helpers -------------------------------------------------------

    @staticmethod
    def _asset_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Per-asset simple returns, first (NaN) row dropped."""
        return prices.pct_change().iloc[1:]

    @staticmethod
    def _cross_sectional_returns(
        signal: pd.DataFrame,
        asset_returns: pd.DataFrame,
        top_fraction: float,
        prefer_high: bool,
    ) -> pd.Series:
        """Long-only cross-sectional portfolio return from a ranking signal.

        Each period the assets are ranked by ``signal``; the top
        ``top_fraction`` (by count, at least one) are held at equal weight and
        earn that period's asset returns. ``signal`` must already be lagged so
        that no future information leaks into the weights.

        Args:
            signal: Per-asset ranking signal, indexed like ``asset_returns``.
            asset_returns: Per-asset simple returns for the periods to hold.
            top_fraction: Fraction of assets to hold long (in ``(0, 1]``).
            prefer_high: If True, hold the highest-signal assets (momentum);
                if False, hold the lowest (mean reversion).

        Returns:
            Per-period portfolio return over the rows where the signal is
            defined for every asset.
        """
        n_assets = signal.shape[1]
        k = max(1, int(round(top_fraction * n_assets)))
        # Align, then keep only rows where every asset has a defined signal.
        sig = signal.reindex(asset_returns.index)
        valid = sig.dropna(how="any")
        rank = valid.rank(axis=1, ascending=not prefer_high, method="first")
        selected = rank <= k
        weights = selected.astype("float64")
        weights = weights.div(weights.sum(axis=1), axis=0)
        aligned_returns = asset_returns.loc[valid.index]
        return (weights * aligned_returns).sum(axis=1)

    def _named(self, series: pd.Series) -> pd.Series:
        """Attach the strategy name and coerce dtype for a clean output."""
        return series.astype(np.float64).rename(self.name)
