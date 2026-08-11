"""Backtest harness: data loading, the learner/strategy run loop, and metrics."""

from nra.backtest.data import (
    DEFAULT_TICKERS,
    load_prices,
    fetch_prices,
    strategy_return_panel,
)
from nra.backtest.engine import (
    BacktestResult,
    best_constant_rebalanced,
    best_fixed_weight,
    run_backtest,
)

__all__ = [
    "DEFAULT_TICKERS",
    "load_prices",
    "fetch_prices",
    "strategy_return_panel",
    "BacktestResult",
    "best_constant_rebalanced",
    "best_fixed_weight",
    "run_backtest",
]
