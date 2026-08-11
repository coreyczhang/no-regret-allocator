"""Matplotlib figures backing the README's empirical results.

These functions take already-computed backtest results (or a return panel) and
produce figures; they do not run the backtest themselves. The module forces the
non-interactive ``Agg`` backend so it renders identically in CI and headless
runs.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nra.backtest.engine import (
    BacktestResult,
    best_constant_rebalanced,
    run_backtest,
)
from nra.learners.exp_gradient import ExponentiatedGradient
from nra.learners.ogd import OnlineGradientDescent

_LEARNER_STYLE: dict[str, Any] = {"linewidth": 1.6}
_BASELINE_STYLE: dict[str, Any] = {"linewidth": 1.3, "linestyle": "--", "alpha": 0.9}


def plot_wealth(
    results: Sequence[BacktestResult],
    returns: pd.DataFrame,
    baseline_names: Sequence[str] = (),
    show_best_crp: bool = True,
) -> plt.Figure:
    """Cumulative wealth of each learner, with hindsight reference lines.

    Args:
        results: Backtest results to draw (learners and baselines).
        returns: The strategy return panel (for the hindsight benchmarks).
        baseline_names: Names in ``results`` to draw in the dashed baseline style.
        show_best_crp: Whether to overlay the best constant-rebalanced portfolio.

    Returns:
        The Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5.2))
    baselines = set(baseline_names)
    for res in results:
        style = _BASELINE_STYLE if res.name in baselines else _LEARNER_STYLE
        ax.plot(res.dates, res.wealth, label=res.name, **style)

    if show_best_crp:
        _, crp_wealth = best_constant_rebalanced(returns.to_numpy(dtype=np.float64))
        ax.plot(
            returns.index,
            crp_wealth,
            color="black",
            linewidth=1.2,
            linestyle=":",
            label="Best CRP (hindsight)",
        )

    ax.set_yscale("log")
    ax.set_title("Cumulative wealth: no-regret learners vs. baselines")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_regret_over_time(results: Sequence[BacktestResult]) -> plt.Figure:
    """Average external regret ``R_t / t`` for each learner.

    The no-regret property is the visual fact that these curves decay toward
    zero: the per-period gap to the best fixed allocation in hindsight vanishes.

    Args:
        results: Backtest results to draw.

    Returns:
        The Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for res in results:
        horizon = np.arange(1, len(res.dates) + 1)
        ax.plot(res.dates, res.regret / horizon, label=res.name, **_LEARNER_STYLE)

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title("Average regret over time (should decay toward zero)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average regret per period,  $R_t / t$")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_learning_rate_sensitivity(
    returns: pd.DataFrame,
    etas: Sequence[float],
    periods_per_year: int = 252,
) -> plt.Figure:
    """Final average regret vs. learning rate for EG and OGD.

    Sweeps the learning-rate hyperparameter for the two rate-based learners and
    plots the resulting final average regret, illustrating the classic U-shaped
    tradeoff (too small = slow to adapt, too large = noisy).

    Args:
        returns: The strategy return panel.
        etas: Grid of learning rates to sweep.
        periods_per_year: Annualization factor (unused for regret, kept for API).

    Returns:
        The Matplotlib figure.
    """
    n = returns.shape[1]
    families: Mapping[str, type] = {
        "ExpGradient": ExponentiatedGradient,
        "OGD": OnlineGradientDescent,
    }
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for label, cls in families.items():
        final_avg_regret = []
        for eta in etas:
            res = run_backtest(cls(n, eta=eta), returns, periods_per_year)
            final_avg_regret.append(res.metrics["final_avg_regret"])
        ax.plot(etas, final_avg_regret, marker="o", label=label, **_LEARNER_STYLE)

    ax.set_xscale("log")
    ax.set_title("Learning-rate sensitivity: final average regret")
    ax.set_xlabel("Learning rate  $\\eta$  (log scale)")
    ax.set_ylabel("Final average regret,  $R_T / T$")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    return fig
