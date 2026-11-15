"""Stat-arb pairs: cointegration, hedge/half-life recovery, signals, OOS split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nra.alphas.statarb import (
    find_cointegrated_pairs,
    generate_positions,
    half_life,
    kalman_hedge_ratio,
    ols_hedge_ratio,
    run_statarb,
    zscore,
)


def _ou(phi: float, n: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """A stationary AR(1)/OU process with autocorrelation ``phi``."""
    e = np.zeros(n)
    noise = rng.normal(0.0, scale, n)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + noise[t]
    return e


@pytest.fixture
def cointegrated_panel() -> pd.DataFrame:
    """Two cointegrated series (y = 1 + 2x + OU) and an unrelated random walk."""
    rng = np.random.default_rng(20260815)
    n = 1500
    x = 50.0 + np.cumsum(rng.normal(0.0, 0.5, n))
    y = 1.0 + 2.0 * x + _ou(0.9, n, rng, scale=1.0)
    z_indep = 50.0 + np.cumsum(rng.normal(0.0, 0.5, n))
    idx = pd.date_range("2016-01-01", periods=n, freq="B")
    return pd.DataFrame({"Y": y, "X": x, "Z": z_indep}, index=idx)


def test_cointegration_flags_true_pair_not_spurious(cointegrated_panel) -> None:
    sector_map = {"Y": "S", "X": "S", "Z": "S"}
    report = find_cointegrated_pairs(
        cointegrated_panel, sector_map, corr_threshold=0.0, pvalue_threshold=0.05
    )
    found = {tuple(sorted((p.a, p.b))) for p in report.pairs}
    assert ("X", "Y") in found  # the genuinely cointegrated pair is detected
    assert ("Y", "Z") not in found and ("X", "Z") not in found  # no spurious ones


def test_ols_recovers_known_hedge_ratio(cointegrated_panel) -> None:
    hedge = ols_hedge_ratio(cointegrated_panel["Y"], cointegrated_panel["X"])
    # The slope (hedge ratio) is what the strategy uses and is well identified.
    # The intercept is not reliably recovered when the regressor is a random
    # walk, so it is intentionally not asserted.
    assert hedge.beta == pytest.approx(2.0, abs=0.1)


def test_half_life_recovers_known_mean_reversion() -> None:
    rng = np.random.default_rng(3)
    phi = 0.9
    spread = pd.Series(_ou(phi, 4000, rng, scale=1.0))
    expected = -np.log(2) / np.log(phi)  # ~6.58 periods
    assert half_life(spread) == pytest.approx(expected, rel=0.25)


def test_half_life_infinite_for_random_walk() -> None:
    rng = np.random.default_rng(4)
    rw = pd.Series(np.cumsum(rng.normal(0, 1, 2000)))
    # A random walk does not mean-revert: half-life is infinite in theory and
    # very large (or inf) in finite samples — never a tradeable few-day figure.
    hl = half_life(rw)
    assert (not np.isfinite(hl)) or hl > 50


def test_kalman_beta_converges_to_truth() -> None:
    rng = np.random.default_rng(5)
    n = 2000
    x = 50.0 + np.cumsum(rng.normal(0.0, 0.5, n))
    y = 2.0 * x + rng.normal(0.0, 1.0, n)
    idx = pd.date_range("2016-01-01", periods=n, freq="B")
    kh = kalman_hedge_ratio(pd.Series(y, index=idx), pd.Series(x, index=idx))
    assert kh.beta.iloc[-500:].mean() == pytest.approx(2.0, abs=0.25)


def test_positions_respect_entry_exit_stop() -> None:
    z = pd.Series([np.nan, 0.0, -2.5, -1.0, -0.2, 0.0, 2.5, 1.0, 0.3, 5.5])
    pos = generate_positions(z, entry=2.0, exit_=0.5, stop=4.0)
    assert pos.iloc[2] == 1  # z=-2.5 -> enter long spread
    assert pos.iloc[3] == 1  # z=-1.0 -> still beyond exit, hold
    assert pos.iloc[4] == 0  # z=-0.2 -> reverted past exit, close
    assert pos.iloc[6] == -1  # z=+2.5 -> enter short spread
    assert pos.iloc[8] == 0  # z=+0.3 -> reverted, close
    assert pos.iloc[9] == 0  # z=+5.5 -> beyond stop, stay flat


def test_zscore_has_no_lookahead() -> None:
    rng = np.random.default_rng(6)
    s = pd.Series(_ou(0.9, 500, rng), index=pd.date_range("2016-01-01", periods=500))
    base = zscore(s, window=60)
    bumped = s.copy()
    bumped.iloc[-1] += 100.0  # shock only the last point
    after = zscore(bumped, window=60)
    assert np.allclose(
        base.iloc[:-1].fillna(0).to_numpy(), after.iloc[:-1].fillna(0).to_numpy()
    )


def test_selection_uses_only_in_sample(cointegrated_panel) -> None:
    sector_map = {"Y": "S", "X": "S", "Z": "S"}
    base = run_statarb(
        cointegrated_panel,
        sector_map,
        in_sample_end="2019-01-01",
        top_n=5,
        corr_threshold=0.0,
        z_window=30,
    )
    scrambled = cointegrated_panel.copy()
    scrambled.loc[scrambled.index > pd.Timestamp("2019-01-01")] *= 1.5
    after = run_statarb(
        scrambled,
        sector_map,
        in_sample_end="2019-01-01",
        top_n=5,
        corr_threshold=0.0,
        z_window=30,
    )
    # Changing only out-of-sample data must not change which pairs were selected.
    assert [(p.a, p.b) for p in base.selected] == [(p.a, p.b) for p in after.selected]
