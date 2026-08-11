"""Simplex validity: projection correctness and per-learner weight invariants."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import all_learners
from nra.simplex import euclidean_projection_simplex, is_on_simplex, uniform_weights


def test_projection_of_simplex_point_is_identity() -> None:
    w = np.array([0.1, 0.2, 0.3, 0.4])
    projected = euclidean_projection_simplex(w)
    assert np.allclose(projected, w)


def test_projection_outputs_valid_simplex_point() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        v = rng.normal(size=6) * 5.0
        w = euclidean_projection_simplex(v)
        assert is_on_simplex(w)


def test_projection_matches_known_case() -> None:
    # Projecting [0.5, 0.5, -0.5]: sorted cumulative gives theta = 0, so the
    # negative coordinate is clipped to 0 and the other two are unchanged.
    w = euclidean_projection_simplex(np.array([0.5, 0.5, -0.5]))
    assert np.allclose(w, [0.5, 0.5, 0.0])


def test_projection_is_closest_point() -> None:
    # The projection should be no farther from v than any random simplex point.
    rng = np.random.default_rng(1)
    v = rng.normal(size=5) * 3.0
    proj = euclidean_projection_simplex(v)
    for _ in range(50):
        other = euclidean_projection_simplex(rng.normal(size=5))
        assert np.linalg.norm(proj - v) <= np.linalg.norm(other - v) + 1e-9


def test_uniform_weights_valid() -> None:
    w = uniform_weights(7)
    assert is_on_simplex(w)
    assert np.allclose(w, 1.0 / 7)


@pytest.mark.parametrize("learner_idx", range(5))
def test_learner_weights_stay_on_simplex(random_returns, learner_idx) -> None:
    n = random_returns.shape[1]
    learner = all_learners(n)[learner_idx]
    r = random_returns.to_numpy()
    assert is_on_simplex(learner.weights)
    for row in r:
        w = learner.update(row)
        assert is_on_simplex(w), f"{learner.name} left the simplex"
        assert np.all(w >= 0.0)
