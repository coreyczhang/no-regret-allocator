"""Momentum alpha: portfolio invariants, no look-ahead, and signal recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nra.alphas.momentum import (
    long_short_weights,
    momentum_signal,
    quantile_buckets,
    run_momentum,
    volatility_scale,
)
from nra.alphas.momentum.costs import apply_costs


@pytest.fixture
def trending_prices() -> pd.DataFrame:
    """Daily prices with a *persistent* cross-sectional drift ranking.

    Each name has its own constant drift, so past winners stay winners. A
    correct momentum implementation must detect this: positive IC, monotone
    deciles, positive long/short spread.
    """
    rng = np.random.default_rng(20260815)
    n_days, n_names = 2600, 20
    drifts = np.linspace(-0.0002, 0.0006, n_names)
    shocks = rng.normal(0.0, 0.01, size=(n_days, n_names)) + drifts
    prices = 100.0 * np.exp(np.cumsum(shocks, axis=0))
    idx = pd.date_range("2015-01-01", periods=n_days, freq="B")
    return pd.DataFrame(
        prices, index=idx, columns=[f"N{i:02d}" for i in range(n_names)]
    )


def test_long_short_is_dollar_neutral(trending_prices) -> None:
    sig = momentum_signal(trending_prices)
    w = long_short_weights(sig, n_quantiles=10)
    live = w.loc[w.abs().sum(axis=1) > 0]
    assert np.allclose(live.sum(axis=1), 0.0, atol=1e-9)  # net zero
    assert np.allclose(live.abs().sum(axis=1), 2.0, atol=1e-9)  # gross = 2
    assert np.allclose(live[live > 0].sum(axis=1), 1.0, atol=1e-9)  # long leg +1


def test_quantile_buckets_are_valid(trending_prices) -> None:
    sig = momentum_signal(trending_prices)
    buckets = quantile_buckets(sig, n_quantiles=10)
    vals = buckets.to_numpy()
    finite = vals[np.isfinite(vals)]
    assert finite.min() >= 1 and finite.max() <= 10
    # Every quantile is populated in a typical month.
    row = buckets.dropna(how="all").iloc[-1].dropna()
    assert row.nunique() == 10


def test_momentum_recovered_on_persistent_signal(trending_prices) -> None:
    res = run_momentum(trending_prices, lookback=12, skip=1, n_quantiles=10)
    # Deciles increase with past-return rank, and the signal orders returns.
    assert res.metrics["decile_monotonicity"] > 0.6
    assert res.metrics["ic_mean"] > 0.05
    assert res.metrics["sharpe_gross"] > 0.5
    assert res.decile_means.iloc[-1] > res.decile_means.iloc[0]  # winners > losers


def test_momentum_signal_has_no_lookahead(trending_prices) -> None:
    base = momentum_signal(trending_prices)
    bumped = trending_prices.copy()
    bumped.iloc[-1] *= 1.25  # shock only the very last day
    after = momentum_signal(bumped)
    common = base.dropna(how="all").index.intersection(after.index)[:-1]
    assert np.allclose(
        base.loc[common].fillna(0).to_numpy(),
        after.loc[common].fillna(0).to_numpy(),
    )


def test_costs_reduce_returns_monotonically(trending_prices) -> None:
    res = run_momentum(trending_prices, cost_bps=0.0)
    cheap = apply_costs(res.ls_gross, res.weights, cost_bps=5.0)
    dear = apply_costs(res.ls_gross, res.weights, cost_bps=25.0)
    assert dear.sum() < cheap.sum() <= res.ls_gross.sum() + 1e-12


def test_volatility_scale_targets_and_caps() -> None:
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0, 0.08, size=300))  # vol far above target
    scaled = volatility_scale(r, target_annual_vol=0.10, window=6, cap=2.0)
    # Exposure is capped, so a high-vol stream is scaled down, not up.
    ratio = (scaled / r).replace([np.inf, -np.inf], np.nan).dropna()
    assert ratio.abs().max() <= 2.0 + 1e-9
    assert scaled.std() < r.std()
