"""Command-line entry point: run the full backtest and write figures.

Usage (after ``pip install -e .``)::

    nra-backtest                 # uses the cached prices, writes figures/
    nra-backtest --prices path/to/prices.csv --figdir figures

The same logic is exposed through ``scripts/run_backtest.py`` for running from a
checkout without installing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nra.analysis.plots import (
    plot_learning_rate_sensitivity,
    plot_regret_over_time,
    plot_wealth,
)
from nra.backtest.data import DEFAULT_CACHE_PATH, load_prices, strategy_return_panel
from nra.backtest.engine import BacktestResult, run_backtest
from nra.learners import (
    ExponentiatedGradient,
    FollowTheWinner,
    Learner,
    OnlineGradientDescent,
    RegretMatching,
    Uniform,
)
from nra.strategies import BuyAndHold, MeanReversion, Momentum, VolatilityTarget


def build_strategies() -> list:
    """The default set of illustrative strategies used in the README results."""
    return [
        BuyAndHold(),
        Momentum(lookback=63),
        MeanReversion(lookback=5),
        VolatilityTarget(target_annual_vol=0.10, window=20),
    ]


def build_learners(n: int) -> tuple[list[Learner], list[str]]:
    """The default learners plus the names to treat as dashed baselines."""
    # Learning rates are calibrated to the arithmetic-return scale of the panel
    # (daily returns ~1e-2, cumulative spread ~1); values this size give visible,
    # gradual adaptation without collapsing instantly onto one strategy.
    learners: list[Learner] = [
        ExponentiatedGradient(n, eta=10.0),
        OnlineGradientDescent(n, eta=2.0),
        RegretMatching(n),
        Uniform(n),
        FollowTheWinner(n),
    ]
    baseline_names = [Uniform(n).name, FollowTheWinner(n).name]
    return learners, baseline_names


def _metrics_frame(results: list[BacktestResult]) -> pd.DataFrame:
    rows = {res.name: res.metrics for res in results}
    frame = pd.DataFrame(rows).T
    return frame[
        ["total_return", "sharpe", "avg_turnover", "final_regret", "final_avg_regret"]
    ]


def run(prices_path: Path, figdir: Path) -> pd.DataFrame:
    """Run the whole pipeline: panel -> backtests -> metrics table + figures."""
    prices = load_prices(prices_path)
    strategies = build_strategies()
    panel = strategy_return_panel(strategies, prices)
    if panel.empty:
        raise RuntimeError("Strategy return panel is empty; check the price data.")

    learners, baseline_names = build_learners(panel.shape[1])
    results = [run_backtest(learner, panel) for learner in learners]

    figdir.mkdir(parents=True, exist_ok=True)
    plot_wealth(results, panel, baseline_names).savefig(figdir / "wealth.png", dpi=140)
    no_regret = [r for r in results if r.name not in set(baseline_names)]
    plot_regret_over_time(
        no_regret + [r for r in results if r.name in baseline_names]
    ).savefig(figdir / "regret.png", dpi=140)
    etas = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    plot_learning_rate_sensitivity(panel, etas).savefig(
        figdir / "lr_sensitivity.png", dpi=140
    )

    metrics = _metrics_frame(results)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prices",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Path to the cached prices CSV.",
    )
    parser.add_argument(
        "--figdir",
        type=Path,
        default=Path("figures"),
        help="Directory to write output figures into.",
    )
    args = parser.parse_args(argv)

    metrics = run(args.prices, args.figdir)
    pd.set_option("display.float_format", lambda v: f"{v:0.4f}")
    print("\nBacktest summary metrics\n" + "=" * 26)
    print(metrics.to_string())
    print(f"\nFigures written to: {args.figdir.resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
