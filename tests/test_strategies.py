"""Strategy sanity checks: valid finite returns, alignment, no look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nra.backtest.data import strategy_return_panel
from nra.strategies import BuyAndHold, MeanReversion, Momentum, VolatilityTarget


@pytest.fixture
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n_days, n_assets = 400, 5
    # Geometric random walk so prices stay positive.
    shocks = rng.normal(0.0004, 0.011, size=(n_days, n_assets))
    prices = 100.0 * np.exp(np.cumsum(shocks, axis=0))
    idx = pd.date_range("2018-01-01", periods=n_days, freq="B")
    return pd.DataFrame(prices, index=idx, columns=[f"A{i}" for i in range(n_assets)])


ALL_STRATEGIES = [
    BuyAndHold(),
    Momentum(lookback=63),
    MeanReversion(lookback=5),
    VolatilityTarget(target_annual_vol=0.10, window=20),
]


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_returns_are_finite_and_named(strategy, prices) -> None:
    out = strategy.compute_returns(prices)
    assert isinstance(out, pd.Series)
    assert out.name == strategy.name
    assert len(out) > 0
    assert np.isfinite(out.to_numpy()).all()


def test_buy_and_hold_matches_manual_basket(prices) -> None:
    out = BuyAndHold().compute_returns(prices)
    growth = prices / prices.iloc[0]
    expected = growth.mean(axis=1).pct_change().iloc[1:]
    assert np.allclose(out.to_numpy(), expected.to_numpy())


def test_panel_alignment_and_validity(prices) -> None:
    panel = strategy_return_panel(ALL_STRATEGIES, prices)
    assert list(panel.columns) == [s.name for s in ALL_STRATEGIES]
    assert not panel.isna().any().any()
    assert panel.index.is_monotonic_increasing


def test_momentum_has_no_lookahead(prices) -> None:
    # Altering the final price must not change any earlier momentum return,
    # because positions are formed only from strictly past information.
    base = Momentum(lookback=20).compute_returns(prices)
    perturbed_prices = prices.copy()
    perturbed_prices.iloc[-1] *= 1.10
    perturbed = Momentum(lookback=20).compute_returns(perturbed_prices)
    common = base.index.intersection(perturbed.index)[:-1]
    assert np.allclose(base.loc[common].to_numpy(), perturbed.loc[common].to_numpy())
