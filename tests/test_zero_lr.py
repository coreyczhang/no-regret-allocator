"""A zero learning rate must freeze the weights at the uniform initialization."""

from __future__ import annotations

import numpy as np

from conftest import rate_learners
from nra.simplex import uniform_weights


def test_zero_learning_rate_stays_uniform(random_returns) -> None:
    n = random_returns.shape[1]
    r = random_returns.to_numpy()
    for learner in rate_learners(n):
        for row in r:
            w = learner.update(row)
            assert np.allclose(
                w, uniform_weights(n)
            ), f"{learner.name} drifted from uniform at eta=0"


def test_zero_learning_rate_respects_nonuniform_init(random_returns) -> None:
    # With eta = 0 the learner should hold whatever it started with.
    from nra.learners import ExponentiatedGradient, OnlineGradientDescent

    n = random_returns.shape[1]
    init = np.array([0.4, 0.3, 0.2, 0.1])
    for cls in (ExponentiatedGradient, OnlineGradientDescent):
        learner = cls(n, eta=0.0, initial_weights=init)
        for row in random_returns.to_numpy():
            w = learner.update(row)
            assert np.allclose(w, init)
