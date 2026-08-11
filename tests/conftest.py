"""Shared fixtures: synthetic return panels and the learner roster."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nra.learners import (
    ExponentiatedGradient,
    FollowTheWinner,
    Learner,
    OnlineGradientDescent,
    RegretMatching,
    Uniform,
)

RNG_SEED = 20260810


def all_learners(n: int) -> list[Learner]:
    """One instance of every learner, for parametrized invariance tests."""
    return [
        Uniform(n),
        FollowTheWinner(n),
        ExponentiatedGradient(n, eta=0.05),
        OnlineGradientDescent(n, eta=0.1),
        RegretMatching(n),
    ]


def rate_learners(n: int) -> list[Learner]:
    """Learners that expose a learning rate (for the zero-rate test)."""
    return [ExponentiatedGradient(n, eta=0.0), OnlineGradientDescent(n, eta=0.0)]


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(RNG_SEED)


@pytest.fixture
def random_returns(rng: np.random.Generator) -> pd.DataFrame:
    """A noisy, mean-zero-ish return panel with no dominant strategy."""
    n_periods, n_strategies = 500, 4
    data = rng.normal(loc=0.0003, scale=0.01, size=(n_periods, n_strategies))
    idx = pd.date_range("2020-01-01", periods=n_periods, freq="B")
    return pd.DataFrame(data, index=idx, columns=[f"s{i}" for i in range(n_strategies)])


@pytest.fixture
def dominant_returns(rng: np.random.Generator) -> pd.DataFrame:
    """A panel where strategy 0 has a clearly higher mean return.

    Used to check that the no-regret learners concentrate on the best strategy
    and that their average regret decays.
    """
    n_periods, n_strategies = 800, 4
    means = np.array([0.0015, 0.0002, 0.0000, -0.0003])
    data = rng.normal(loc=means, scale=0.008, size=(n_periods, n_strategies))
    idx = pd.date_range("2019-01-01", periods=n_periods, freq="B")
    return pd.DataFrame(data, index=idx, columns=[f"s{i}" for i in range(n_strategies)])
