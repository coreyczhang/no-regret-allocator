"""End-to-end pairs backtest with a strict in-sample / out-of-sample split."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nra.alphas.statarb.hedge import kalman_hedge_ratio, ols_hedge_ratio
from nra.alphas.statarb.selection import (
    CandidatePair,
    SelectionReport,
    find_cointegrated_pairs,
)
from nra.alphas.statarb.signals import generate_positions, pair_returns
from nra.alphas.statarb.spread import compute_spread, half_life, zscore
from nra.metrics import (
    annualized_return,
    max_drawdown,
    sharpe_ratio,
)

TRADING_DAYS = 252


@dataclass
class StatArbResult:
    """Series, per-pair diagnostics, and metrics from a pairs backtest."""

    portfolio_oos: pd.Series  # equal-weight out-of-sample portfolio return
    portfolio_is: pd.Series  # in-sample portfolio return (for decay comparison)
    pair_stats: pd.DataFrame  # per-pair selection + performance diagnostics
    selected: list[CandidatePair]
    report: SelectionReport
    example: dict[str, pd.Series] = field(default_factory=dict)
    example_pair: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


def run_statarb(
    prices: pd.DataFrame,
    sector_map: dict[str, str],
    in_sample_end: str = "2015-12-31",
    top_n: int = 20,
    method: str = "ols",
    entry: float = 2.0,
    exit_: float = 0.5,
    stop: float = 4.0,
    z_window: int = 60,
    cost_bps: float = 5.0,
    corr_threshold: float = 0.80,
    pvalue_threshold: float = 0.05,
) -> StatArbResult:
    """Select cointegrated pairs in-sample, trade them out-of-sample, evaluate.

    Args:
        prices: Daily prices (date index, one column per name).
        sector_map: Ticker -> sector; only within-sector pairs are considered.
        in_sample_end: Last date used for selection and hedge estimation.
        top_n: Keep the ``top_n`` most significant pairs.
        method: ``"ols"`` (static hedge) or ``"kalman"`` (dynamic hedge).
        entry, exit_, stop: Z-score trading thresholds.
        z_window: Rolling window for the OLS-spread z-score.
        cost_bps: Round-trip transaction cost per position change.
        corr_threshold, pvalue_threshold: Selection screens.

    Returns:
        A populated :class:`StatArbResult`.
    """
    if method not in {"ols", "kalman"}:
        raise ValueError("method must be 'ols' or 'kalman'.")

    in_sample = prices.loc[:in_sample_end]
    report = find_cointegrated_pairs(
        in_sample,
        sector_map,
        corr_threshold=corr_threshold,
        pvalue_threshold=pvalue_threshold,
    )
    selected = report.pairs[:top_n]

    returns = prices.pct_change(fill_method=None)
    split = pd.Timestamp(in_sample_end)

    per_pair_oos: dict[str, pd.Series] = {}
    per_pair_is: dict[str, pd.Series] = {}
    rows: list[dict[str, object]] = []
    example: dict[str, pd.Series] = {}
    example_name = ""

    for pair in selected:
        y, x = prices[pair.a], prices[pair.b]
        name = f"{pair.a}/{pair.b}"

        if method == "ols":
            hedge = ols_hedge_ratio(in_sample[pair.a], in_sample[pair.b])
            spread = compute_spread(y, x, hedge.beta, hedge.alpha)
            beta_used: float | pd.Series = hedge.beta
        else:
            # Use the Kalman *dynamic hedge ratio* to build the spread, then
            # z-score it on a rolling window. Two deliberate choices: (1) trade
            # the level spread, not the filter innovation (the innovation is
            # ~white noise with no mean reversion); (2) exclude the Kalman
            # intercept from the spread and use a slow drift, because a fast
            # intercept absorbs the very mean reversion we want to trade.
            kh = kalman_hedge_ratio(y, x, delta=1e-7)
            spread = (y - kh.beta * x).dropna().rename("spread")
            beta_used = kh.beta
        z = zscore(spread, z_window)
        is_spread = spread.loc[:in_sample_end]

        positions = generate_positions(z, entry, exit_, stop)
        pnl = pair_returns(
            returns[pair.a], returns[pair.b], positions, beta_used, cost_bps
        )

        pnl_oos = pnl.loc[pnl.index > split]
        pnl_is = pnl.loc[pnl.index <= split]
        per_pair_oos[name] = pnl_oos
        per_pair_is[name] = pnl_is

        hl = half_life(is_spread)
        pos_oos = positions.loc[positions.index > split]
        n_trades = int((pos_oos.diff().abs() > 0).sum())
        rows.append(
            {
                "pair": name,
                "sector": pair.sector,
                "coint_pvalue": pair.coint_pvalue,
                "half_life_days": hl,
                "sharpe_is": sharpe_ratio(pnl_is, TRADING_DAYS),
                "sharpe_oos": sharpe_ratio(pnl_oos, TRADING_DAYS),
                "oos_trades": n_trades,
            }
        )
        if not example:  # keep the most significant pair for plotting
            example = {"spread": spread, "zscore": z, "position": positions}
            example_name = name

    portfolio_oos = _equal_weight(per_pair_oos)
    portfolio_is = _equal_weight(per_pair_is)
    columns = [
        "pair",
        "sector",
        "coint_pvalue",
        "half_life_days",
        "sharpe_is",
        "sharpe_oos",
        "oos_trades",
    ]
    pair_stats = pd.DataFrame(rows, columns=columns)

    finite_hl = pair_stats["half_life_days"].replace([np.inf, -np.inf], np.nan).dropna()
    metrics = {
        "n_tested": float(report.n_tested),
        "n_passed": float(report.n_passed),
        "expected_false_positives": report.expected_false_positives,
        "n_traded": float(len(selected)),
        "sharpe_is": sharpe_ratio(portfolio_is, TRADING_DAYS),
        "sharpe_oos": sharpe_ratio(portfolio_oos, TRADING_DAYS),
        "ann_return_oos": annualized_return(portfolio_oos, TRADING_DAYS),
        "max_drawdown_oos": max_drawdown(portfolio_oos),
        "avg_half_life_days": (
            float(finite_hl.mean()) if not finite_hl.empty else float("nan")
        ),
        "pct_pairs_oos_positive": (
            float((pair_stats["sharpe_oos"] > 0).mean())
            if not pair_stats.empty
            else 0.0
        ),
    }

    return StatArbResult(
        portfolio_oos=portfolio_oos,
        portfolio_is=portfolio_is,
        pair_stats=pair_stats,
        selected=selected,
        report=report,
        example=example,
        example_pair=example_name,
        metrics=metrics,
    )


def _equal_weight(per_pair: dict[str, pd.Series]) -> pd.Series:
    """Equal-weight average of pair return series, aligned on the union index."""
    if not per_pair:
        return pd.Series(dtype=np.float64)
    frame = pd.DataFrame(per_pair)
    return frame.mean(axis=1).dropna().astype(np.float64)
