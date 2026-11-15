"""Spread construction, z-scoring, and the mean-reversion half-life."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_spread(
    y: pd.Series, x: pd.Series, beta: float, alpha: float = 0.0
) -> pd.Series:
    """The hedged spread ``y - beta * x - alpha``."""
    return (y - beta * x - alpha).rename("spread")


def zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """Rolling z-score of a spread, using only trailing data.

    The mean and standard deviation are shifted by one period so the z-score at
    time ``t`` is knowable at ``t`` and carries no look-ahead.
    """
    mean = spread.rolling(window).mean().shift(1)
    sd = spread.rolling(window).std().shift(1)
    return ((spread - mean) / sd).rename("zscore")


def half_life(spread: pd.Series) -> float:
    """Ornstein-Uhlenbeck half-life of mean reversion, in periods.

    Fits the discrete analogue ``Δs_t = λ · s_{t-1} + ε`` by OLS; the half-life
    is ``-ln(2) / ln(1 + λ)``. A small positive half-life means fast reversion
    (good for trading); a non-mean-reverting spread (``λ ≥ 0``) returns ``inf``.

    Args:
        spread: The spread series.

    Returns:
        Half-life in the spread's native periods (``inf`` if not reverting).
    """
    s = spread.dropna()
    if len(s) < 3:
        return float("inf")
    lagged = s.shift(1).dropna()
    delta = (s - s.shift(1)).dropna()
    lagged, delta = lagged.align(delta, join="inner")
    lam, _ = np.polyfit(lagged.to_numpy(), delta.to_numpy(), 1)
    if lam >= 0:
        return float("inf")
    return float(-np.log(2) / np.log(1 + lam))
