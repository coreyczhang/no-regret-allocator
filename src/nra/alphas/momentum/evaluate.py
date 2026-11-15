"""End-to-end momentum backtest and its evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nra.alphas.momentum.costs import apply_costs, turnover_series
from nra.alphas.momentum.portfolio import decile_returns, long_short_weights
from nra.alphas.momentum.risk import volatility_scale
from nra.alphas.momentum.signal import forward_returns, momentum_signal
from nra.metrics import (
    annualized_return,
    annualized_vol,
    hit_rate,
    information_coefficient,
    max_drawdown,
    sharpe_ratio,
    t_statistic,
)

# Months per year, used to annualize the monthly-frequency momentum results.
MONTHS_PER_YEAR = 12


@dataclass
class MomentumResult:
    """Series and summary metrics from one momentum backtest."""

    ls_gross: pd.Series  # monthly top-minus-bottom long/short return, gross
    ls_net: pd.Series  # same, net of transaction costs
    ls_risk_managed: pd.Series  # volatility-targeted (crash-managed) net series
    decile_returns: pd.DataFrame  # monthly return of each decile bucket
    decile_means: pd.Series  # annualized mean return per decile (1..n)
    ic: pd.Series  # monthly information coefficient
    weights: pd.DataFrame  # monthly long/short weights
    turnover: pd.Series  # monthly L1 turnover
    regime: pd.DataFrame = field(default_factory=pd.DataFrame)  # per-regime stats
    metrics: dict[str, float] = field(default_factory=dict)

    def summary(self) -> pd.Series:
        """Return the metrics as a labelled Series for printing."""
        return pd.Series(self.metrics)


def run_momentum(
    prices: pd.DataFrame,
    lookback: int = 12,
    skip: int = 1,
    n_quantiles: int = 10,
    cost_bps: float = 10.0,
    target_vol: float = 0.12,
    vol_window: int = 6,
) -> MomentumResult:
    """Run the cross-sectional momentum strategy and evaluate it.

    Args:
        prices: Daily adjusted close prices (date index, one column per name).
        lookback: Formation window in months.
        skip: Most-recent months skipped in the signal.
        n_quantiles: Number of ranking buckets (10 = deciles).
        cost_bps: Transaction cost per unit turnover, in basis points.
        target_vol: Annualized volatility target for the risk-managed overlay.
        vol_window: Trailing window (months) for the realized-vol estimate.

    Returns:
        A populated :class:`MomentumResult`.
    """
    signal = momentum_signal(prices, lookback=lookback, skip=skip)
    fwd = forward_returns(prices)

    # Keep months where the signal is defined for enough names to rank.
    enough = signal.notna().sum(axis=1) >= n_quantiles
    signal = signal.loc[enough]

    weights = long_short_weights(signal, n_quantiles)
    fwd_aligned = fwd.reindex_like(weights)
    ls_gross = (weights * fwd_aligned).sum(axis=1)
    # The final formation month has no realized forward return yet.
    ls_gross = ls_gross.loc[fwd_aligned.notna().any(axis=1)].astype(np.float64)

    weights = weights.loc[ls_gross.index]
    ls_net = apply_costs(ls_gross, weights, cost_bps=cost_bps)
    ls_rm = volatility_scale(
        ls_net,
        target_annual_vol=target_vol,
        window=vol_window,
        periods_per_year=MONTHS_PER_YEAR,
    )
    turnover = turnover_series(weights)

    deciles = decile_returns(signal, fwd, n_quantiles).loc[ls_gross.index]
    decile_means = (deciles.mean(axis=0) * MONTHS_PER_YEAR).rename("ann_mean")
    ic = information_coefficient(signal, fwd).loc[
        lambda s: s.index.isin(ls_gross.index)
    ]

    metrics = {
        "ann_return_gross": annualized_return(ls_gross, MONTHS_PER_YEAR),
        "ann_return_net": annualized_return(ls_net, MONTHS_PER_YEAR),
        "ann_vol": annualized_vol(ls_gross, MONTHS_PER_YEAR),
        "sharpe_gross": sharpe_ratio(ls_gross, MONTHS_PER_YEAR),
        "sharpe_net": sharpe_ratio(ls_net, MONTHS_PER_YEAR),
        "sharpe_risk_managed": sharpe_ratio(ls_rm, MONTHS_PER_YEAR),
        "t_stat_ls": t_statistic(ls_gross),
        "ic_mean": float(ic.mean()) if not ic.empty else 0.0,
        "ic_ir": float(ic.mean() / ic.std(ddof=1)) if ic.std(ddof=1) else 0.0,
        "ic_t_stat": t_statistic(ic),
        "max_drawdown_net": max_drawdown(ls_net),
        "max_drawdown_risk_managed": max_drawdown(ls_rm),
        "avg_turnover": float(turnover.mean()),
        "hit_rate": hit_rate(ls_gross),
        "decile_monotonicity": _spearman_monotonicity(decile_means),
    }

    return MomentumResult(
        ls_gross=ls_gross,
        ls_net=ls_net,
        ls_risk_managed=ls_rm,
        decile_returns=deciles,
        decile_means=decile_means,
        ic=ic,
        weights=weights,
        turnover=turnover,
        regime=_regime_table(ls_net, ls_rm),
        metrics=metrics,
    )


# Named regimes that frame the momentum-crash story: momentum earns strongly in
# a trending bear market (shorting losers), then suffers its signature crash in
# the violent rebound, and has been roughly flat in large caps since.
_REGIMES: dict[str, tuple[str, str]] = {
    "Bear market (2008)": ("2008-01-01", "2008-12-31"),
    "Rebound crash (2009)": ("2009-01-01", "2009-12-31"),
    "Post-crisis (2010+)": ("2010-01-01", "2100-01-01"),
}


def _regime_table(ls_net: pd.Series, ls_rm: pd.Series) -> pd.DataFrame:
    """Sharpe of the naive vs. risk-managed series across named regimes."""
    rows = {}
    for label, (start, end) in _REGIMES.items():
        seg_net = ls_net.loc[start:end]
        seg_rm = ls_rm.loc[start:end]
        rows[label] = {
            "months": int(seg_net.notna().sum()),
            "sharpe_net": sharpe_ratio(seg_net, MONTHS_PER_YEAR),
            "sharpe_risk_managed": sharpe_ratio(seg_rm, MONTHS_PER_YEAR),
        }
    return pd.DataFrame(rows).T


def _spearman_monotonicity(decile_means: pd.Series) -> float:
    """Rank correlation between decile index and decile mean return in [-1, 1].

    A value near +1 means higher-momentum deciles earn higher returns — the
    monotonic pattern the strategy predicts.
    """
    from scipy import stats

    order = np.arange(1, len(decile_means) + 1)
    rho, _ = stats.spearmanr(order, decile_means.to_numpy())
    return float(rho) if np.isfinite(rho) else 0.0
