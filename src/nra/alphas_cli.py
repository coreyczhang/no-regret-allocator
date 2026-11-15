"""Run the alphas and the allocator capstone, printing metrics and saving figures.

Usage (after ``pip install -e .``)::

    nra-alphas                 # uses the cached equity panel, writes figures/

Produces ``figures/momentum.png``, ``figures/statarb.png``, and
``figures/capstone.png`` plus a metrics summary for each layer.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

from nra.adapters import monthly_alpha_panel
from nra.alphas.momentum import run_momentum
from nra.alphas.statarb import run_statarb
from nra.analysis.alpha_plots import plot_capstone, plot_momentum, plot_statarb
from nra.backtest.engine import run_backtest
from nra.data import SECTOR_MAP, load_equity_prices
from nra.learners import (
    ExponentiatedGradient,
    FollowTheWinner,
    OnlineGradientDescent,
    RegretMatching,
    Uniform,
)

IN_SAMPLE_END = "2015-12-31"
MONTHS_PER_YEAR = 12


def _print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n{title}\n" + "=" * len(title))
    for key, value in metrics.items():
        print(f"  {key:28s} {value:+.4f}")


def run(prices: pd.DataFrame, figdir: Path) -> None:
    figdir.mkdir(parents=True, exist_ok=True)

    # --- Momentum -----------------------------------------------------------
    mom = run_momentum(prices)
    _print_metrics("Momentum (cross-sectional long/short)", mom.metrics)
    print("\n  Regime Sharpe (net vs. risk-managed):")
    print(mom.regime.round(2).to_string())
    plot_momentum(mom).savefig(figdir / "momentum.png", dpi=130)

    # --- Statistical arbitrage ---------------------------------------------
    sa = run_statarb(prices, SECTOR_MAP, in_sample_end=IN_SAMPLE_END, method="ols")
    _print_metrics("Stat-arb pairs (OLS hedge, OOS)", sa.metrics)
    print("\n  Most significant pairs (in-sample selection):")
    print(
        sa.pair_stats[
            [
                "pair",
                "sector",
                "coint_pvalue",
                "half_life_days",
                "sharpe_is",
                "sharpe_oos",
            ]
        ]
        .head(8)
        .round(3)
        .to_string(index=False)
    )
    plot_statarb(sa, in_sample_end=IN_SAMPLE_END).savefig(
        figdir / "statarb.png", dpi=130
    )

    # --- Capstone: no-regret allocation across the alphas -------------------
    panel = monthly_alpha_panel(prices, SECTOR_MAP)
    n = panel.shape[1]
    eg = ExponentiatedGradient(n, eta=1.5)  # gentle rate -> readable weight path
    learners = [
        eg,
        OnlineGradientDescent(n, eta=2.0),
        RegretMatching(n),
        Uniform(n),
        FollowTheWinner(n),
    ]
    results = [
        run_backtest(x, panel, periods_per_year=MONTHS_PER_YEAR) for x in learners
    ]
    print("\nCapstone: allocator over [" + ", ".join(panel.columns) + "]")
    print("=" * 52)
    summary = pd.DataFrame({r.name: r.metrics for r in results}).T
    print(
        summary[["total_return", "sharpe", "avg_turnover", "final_avg_regret"]]
        .round(4)
        .to_string()
    )
    plot_capstone(results, panel, weights_for=eg.name).savefig(
        figdir / "capstone.png", dpi=130
    )

    print(f"\nFigures written to: {figdir.resolve()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figdir", type=Path, default=Path("figures"))
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    run(load_equity_prices(), args.figdir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
