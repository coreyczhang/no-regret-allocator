"""Capstone: the adapters build a valid panel the allocator can run over."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nra.adapters import MarketStrategy, monthly_alpha_panel
from nra.backtest.engine import run_backtest
from nra.learners import ExponentiatedGradient, Uniform

_SECTORS = {"Alpha": 6, "Beta": 6}


@pytest.fixture
def toy_prices() -> pd.DataFrame:
    """Price panel with genuine within-sector cointegration.

    Each sector shares a common stochastic trend plus name-specific stationary
    noise, so same-sector names are cointegrated and the pairs screener has real
    pairs to find. Spans 2012+ so the default 2015 in-sample split is non-empty.
    """
    rng = np.random.default_rng(99)
    n_days = 2800
    idx = pd.date_range("2012-01-01", periods=n_days, freq="B")

    def ou(phi: float, scale: float) -> np.ndarray:
        e = np.zeros(n_days)
        noise = rng.normal(0.0, scale, n_days)
        for t in range(1, n_days):
            e[t] = phi * e[t - 1] + noise[t]
        return e

    cols: dict[str, np.ndarray] = {}
    for sector, k in _SECTORS.items():
        factor = np.cumsum(rng.normal(0.0002, 0.015, n_days))  # sector trend
        for j in range(k):
            load = 0.9 + 0.2 * rng.random()
            level = rng.uniform(3.8, 4.2)
            log_price = level + load * factor + ou(0.95, 0.012)
            cols[f"{sector[0]}{j:02d}"] = np.exp(log_price)
    return pd.DataFrame(cols, index=idx)


@pytest.fixture
def toy_sectors(toy_prices) -> dict[str, str]:
    return {c: ("Alpha" if c.startswith("A") else "Beta") for c in toy_prices.columns}


def test_market_stream_matches_equal_weight(toy_prices) -> None:
    mkt = MarketStrategy().compute_returns(toy_prices)
    expected = toy_prices.pct_change(fill_method=None).mean(axis=1)
    assert np.allclose(mkt.dropna().to_numpy(), expected.dropna().to_numpy())


def test_alpha_panel_is_clean_and_monthly(toy_prices, toy_sectors) -> None:
    panel = monthly_alpha_panel(toy_prices, toy_sectors)
    assert list(panel.columns) == ["Momentum", "Pairs", "Market"]
    assert not panel.isna().any().any()
    assert panel.index.is_monotonic_increasing
    # Month-end frequency.
    assert (panel.index.day > 20).all()


def test_allocator_runs_over_alpha_panel(toy_prices, toy_sectors) -> None:
    panel = monthly_alpha_panel(toy_prices, toy_sectors)
    res = run_backtest(ExponentiatedGradient(panel.shape[1], eta=2.0), panel, 12)
    # Weights stay on the simplex and regret is finite the whole way.
    assert np.allclose(res.weights.sum(axis=1), 1.0)
    assert np.isfinite(res.regret).all()


def test_no_regret_beats_or_matches_uniform_average(toy_prices, toy_sectors) -> None:
    # Over a long run, the learner's average regret should not exceed the fixed
    # uniform allocation's by more than noise (it is designed to be competitive).
    panel = monthly_alpha_panel(toy_prices, toy_sectors)
    eg = run_backtest(ExponentiatedGradient(panel.shape[1], eta=2.0), panel, 12)
    uni = run_backtest(Uniform(panel.shape[1]), panel, 12)
    assert eg.metrics["final_avg_regret"] <= uni.metrics["final_avg_regret"] + 1e-6
