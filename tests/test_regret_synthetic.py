"""Empirical no-regret behavior on synthetic data with a known best strategy.

These tests encode the theory's qualitative predictions: on a stationary panel
with one dominant strategy, a no-regret learner should (a) concentrate weight on
that strategy, (b) drive its average regret toward zero, and (c) do so with
sublinearly growing cumulative regret.
"""

from __future__ import annotations

import numpy as np
import pytest

from nra.backtest.engine import run_backtest
from nra.learners import (
    ExponentiatedGradient,
    OnlineGradientDescent,
    RegretMatching,
)

# Learning rates scaled to the fixture's return magnitude (~1e-3 per period):
# eta must be large enough that eta * (cumulative-return spread) drives real
# concentration. EG's smooth softmax tolerates a large rate; OGD's single-step
# gradient is noisier, so it uses a smaller, stability-appropriate rate.
NO_REGRET_FACTORIES = [
    lambda n: ExponentiatedGradient(n, eta=20.0),
    lambda n: OnlineGradientDescent(n, eta=5.0),
    lambda n: RegretMatching(n),
]


@pytest.mark.parametrize("factory", NO_REGRET_FACTORIES)
def test_concentrates_on_dominant_strategy(dominant_returns, factory) -> None:
    n = dominant_returns.shape[1]
    res = run_backtest(factory(n), dominant_returns)
    # Judge concentration on the time-averaged allocation: a single period's
    # weight is noisy for gradient-style learners, but the average is what the
    # no-regret guarantee governs. Strategy 0 is the dominant one in the fixture.
    avg_weights = res.weights.mean(axis=0)
    assert int(np.argmax(avg_weights)) == 0
    assert avg_weights[0] > 0.4


@pytest.mark.parametrize("factory", NO_REGRET_FACTORIES)
def test_average_regret_decays(dominant_returns, factory) -> None:
    n = dominant_returns.shape[1]
    res = run_backtest(factory(n), dominant_returns)
    horizon = np.arange(1, len(res.dates) + 1)
    avg_regret = res.regret / horizon
    # Average regret ends close to zero in absolute terms; the best strategy's
    # mean return is ~1.5e-3/period, so this is well under 1/2 of the edge.
    # (The shrinking of the regret curve is checked in the envelope test.)
    assert avg_regret[-1] < 6e-4


@pytest.mark.parametrize("factory", NO_REGRET_FACTORIES)
def test_regret_envelope_shrinks(dominant_returns, factory) -> None:
    n = dominant_returns.shape[1]
    res = run_backtest(factory(n), dominant_returns)
    t = len(res.dates)
    avg_regret = np.abs(res.regret / np.arange(1, t + 1))
    # Sublinear regret means the |R_t / t| envelope decays: its peak over the
    # second half of the run is strictly below its peak over the first half.
    # (A short warmup is skipped because R_t / t is erratic for tiny t.)
    warmup = t // 20
    first_half_peak = avg_regret[warmup : t // 2].max()
    second_half_peak = avg_regret[t // 2 :].max()
    assert second_half_peak < first_half_peak


def test_no_regret_beats_naive_on_dominant_panel(dominant_returns) -> None:
    """No-regret learners should end with less regret than chasing the winner."""
    from nra.learners import FollowTheWinner

    n = dominant_returns.shape[1]
    eg = run_backtest(ExponentiatedGradient(n, eta=0.5), dominant_returns)
    ftw = run_backtest(FollowTheWinner(n), dominant_returns)
    assert eg.metrics["final_regret"] < ftw.metrics["final_regret"]
