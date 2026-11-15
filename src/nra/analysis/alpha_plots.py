"""Figures for the momentum and stat-arb alphas and the allocator capstone."""

from __future__ import annotations

from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nra.alphas.momentum.evaluate import MomentumResult
from nra.alphas.statarb.evaluate import StatArbResult
from nra.backtest.engine import BacktestResult


def plot_momentum(result: MomentumResult) -> plt.Figure:
    """Four-panel momentum summary: deciles, equity curves, regimes, IC."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Decile monotonicity.
    ax = axes[0, 0]
    ax.bar(result.decile_means.index, result.decile_means.values, color="#4C78A8")
    ax.set_title(
        f"Decile annualized return "
        f"(monotonicity ρ = {result.metrics['decile_monotonicity']:+.2f})"
    )
    ax.set_xlabel("Momentum decile (1 = losers, 10 = winners)")
    ax.set_ylabel("Annualized return")
    ax.axhline(0, color="black", lw=0.6)

    # Equity curves: naive net vs risk-managed.
    ax = axes[0, 1]
    ax.plot((1 + result.ls_net).cumprod(), label="Long/short (net)", lw=1.5)
    ax.plot(
        (1 + result.ls_risk_managed).cumprod(),
        label="Risk-managed (vol-targeted)",
        lw=1.5,
    )
    ax.set_yscale("log")
    ax.set_title("Cumulative long/short wealth")
    ax.set_ylabel("Growth of $1 (log)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)

    # Regime Sharpe: the momentum crash.
    ax = axes[1, 0]
    reg = result.regime
    xpos = np.arange(len(reg))
    ax.bar(xpos - 0.2, reg["sharpe_net"], 0.4, label="Naive", color="#E45756")
    ax.bar(
        xpos + 0.2,
        reg["sharpe_risk_managed"],
        0.4,
        label="Risk-managed",
        color="#54A24B",
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels(reg.index, fontsize=8, rotation=10)
    ax.set_title("Sharpe by regime (the momentum crash)")
    ax.set_ylabel("Annualized Sharpe")
    ax.axhline(0, color="black", lw=0.6)
    ax.legend(fontsize=8)

    # Rolling information coefficient.
    ax = axes[1, 1]
    ax.plot(result.ic, color="#B279A2", lw=0.8, alpha=0.6, label="Monthly IC")
    ax.plot(
        result.ic.rolling(12).mean(),
        color="#B279A2",
        lw=2.0,
        label="12-month mean IC",
    )
    ax.axhline(
        result.ic.mean(),
        color="black",
        lw=0.8,
        ls="--",
        label=f"Mean IC = {result.ic.mean():+.3f}",
    )
    ax.set_title("Information coefficient over time")
    ax.set_ylabel("Rank IC (signal vs. forward return)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    fig.suptitle("Cross-sectional momentum: a rigorous, bias-aware test", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_statarb(
    result: StatArbResult, in_sample_end: str = "2015-12-31"
) -> plt.Figure:
    """Three-panel pairs summary: example spread, decay scatter, OOS wealth."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # Example pair z-score with trading bands.
    ax = axes[0]
    z = result.example.get("zscore")
    if z is not None:
        z = z.loc[z.index > pd.Timestamp(in_sample_end)]
        ax.plot(z, color="#4C78A8", lw=0.8)
        for lvl, style in [(2, "--"), (-2, "--"), (0.5, ":"), (-0.5, ":")]:
            ax.axhline(lvl, color="gray", lw=0.7, ls=style)
        ax.axhline(0, color="black", lw=0.6)
    ax.set_title(f"Spread z-score (OOS): {result.example_pair}")
    ax.set_ylabel("z-score")

    # In-sample vs out-of-sample per-pair Sharpe (edge decay).
    ax = axes[1]
    ps = result.pair_stats
    ax.scatter(ps["sharpe_is"], ps["sharpe_oos"], c="#72B7B2", edgecolor="k", s=35)
    lim = [
        min(ps["sharpe_is"].min(), ps["sharpe_oos"].min()) - 0.3,
        max(ps["sharpe_is"].max(), ps["sharpe_oos"].max()) + 0.3,
    ]
    ax.plot(lim, lim, color="gray", lw=0.8, ls="--")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_title("Per-pair Sharpe: in-sample vs. out-of-sample")
    ax.set_xlabel("In-sample Sharpe")
    ax.set_ylabel("Out-of-sample Sharpe")

    # Out-of-sample portfolio wealth.
    ax = axes[2]
    ax.plot((1 + result.portfolio_oos).cumprod(), color="#54A24B", lw=1.6)
    ax.set_title(f"OOS portfolio wealth (Sharpe {result.metrics['sharpe_oos']:+.2f})")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Statistical-arbitrage pairs: in-sample selection, out-of-sample trading",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_capstone(
    results: Sequence[BacktestResult],
    panel: pd.DataFrame,
    weights_for: str,
) -> plt.Figure:
    """Three-panel capstone: wealth vs. alphas, weight evolution, regret."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # Wealth: learners vs. the individual alpha streams.
    ax = axes[0]
    for col in panel.columns:
        ax.plot((1 + panel[col]).cumprod(), lw=1.0, ls="--", alpha=0.7, label=col)
    for res in results:
        ax.plot(res.dates, res.wealth, lw=1.8, label=res.name)
    ax.set_yscale("log")
    ax.set_title("Allocator vs. individual alphas")
    ax.set_ylabel("Growth of $1 (log)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.25)

    # Weight evolution of one learner.
    ax = axes[1]
    chosen = next(r for r in results if r.name == weights_for)
    ax.stackplot(chosen.dates, chosen.weights.T, labels=list(panel.columns), alpha=0.85)
    ax.set_title(f"{weights_for}: allocation over time")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper left")

    # Average regret over time.
    ax = axes[2]
    for res in results:
        horizon = np.arange(1, len(res.dates) + 1)
        ax.plot(res.dates, res.regret / horizon, lw=1.4, label=res.name)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_title("Average regret over time")
    ax.set_ylabel("$R_t / t$")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Capstone: no-regret allocation across the alpha strategies", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
