"""Shared performance and statistical metrics.

Reused by both alpha-research layers and the allocator backtest so that Sharpe,
drawdown, t-stats, and the information coefficient are defined in exactly one
place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Annualized Sharpe ratio (zero risk-free rate)."""
    r = returns.dropna()
    sd = r.std(ddof=1)
    if len(r) < 2 or sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def annualized_return(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Geometric annualized return."""
    r = returns.dropna()
    if r.empty:
        return 0.0
    growth = float((1.0 + r).prod())
    years = len(r) / periods_per_year
    if years <= 0 or growth <= 0:
        return float("nan")
    return growth ** (1.0 / years) - 1.0


def annualized_vol(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Annualized volatility."""
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def t_statistic(returns: pd.Series) -> float:
    """One-sample t-stat that the mean per-period return differs from zero."""
    r = returns.dropna()
    sd = r.std(ddof=1)
    if len(r) < 2 or sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(r.mean() / (sd / np.sqrt(len(r))))


def drawdown_curve(returns: pd.Series) -> pd.Series:
    """Drawdown path of cumulative wealth (0 at highs, negative in drawdowns)."""
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough drawdown (a negative number)."""
    dd = drawdown_curve(returns)
    return float(dd.min()) if not dd.empty else 0.0


def hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with a strictly positive return."""
    r = returns.dropna()
    return float((r > 0).mean()) if not r.empty else 0.0


def information_coefficient(signal: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Per-period cross-sectional Spearman rank correlation of signal vs. return.

    For each period, the rank correlation between the signal and the realized
    forward return across all names with both values present. The time series of
    ICs summarizes how well the signal orders the cross-section; its mean, its
    information ratio (mean/std), and its t-stat are the headline numbers.

    Args:
        signal: Signal values (period index, one column per name).
        fwd: Forward returns aligned to ``signal``.

    Returns:
        Series of ICs indexed by period (periods with < 3 pairs are dropped).
    """
    fwd = fwd.reindex_like(signal)
    ics: dict[pd.Timestamp, float] = {}
    for period in signal.index:
        s = signal.loc[period]
        f = fwd.loc[period]
        mask = s.notna() & f.notna()
        if int(mask.sum()) < 3:
            continue
        rho, _ = stats.spearmanr(s[mask], f[mask])
        if np.isfinite(rho):
            ics[period] = float(rho)
    return pd.Series(ics, name="IC").sort_index()
