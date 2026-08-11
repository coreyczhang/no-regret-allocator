"""Common interface for all allocation learners.

Conventions
-----------
Time is discrete. At the start of period ``t`` the learner holds a weight
vector ``w_t`` on the probability simplex over ``N`` strategies. The market then
reveals the realized simple returns ``r_t in R^N`` of each strategy over that
period, and the learner's realized portfolio return is the inner product
``w_t . r_t``. The learner calls :meth:`update` with ``r_t`` to produce the
weights ``w_{t+1}`` it will use next period.

"No regret" is measured against the best *fixed* weight vector chosen with
hindsight knowledge of the whole return path (see :mod:`nra.backtest.engine`).
Each learner guarantees its cumulative regret grows sublinearly in the horizon
``T``, so the average per-period gap to the best fixed allocation vanishes.
"""

from __future__ import annotations

import abc

import numpy as np
from numpy.typing import NDArray

from nra.simplex import is_on_simplex, uniform_weights

FloatArray = NDArray[np.float64]


class Learner(abc.ABC):
    """Abstract base class for an online allocation learner.

    Subclasses implement :meth:`_step`, which maps the realized per-strategy
    returns of the period just observed to the next weight vector. The base
    class owns the current weights, validates that every produced weight vector
    is a valid simplex point, and provides :meth:`reset`.

    Args:
        n: Number of candidate strategies.
        initial_weights: Optional starting allocation; defaults to uniform.
    """

    def __init__(self, n: int, initial_weights: FloatArray | None = None) -> None:
        if n <= 0:
            raise ValueError("n must be a positive integer.")
        self.n = int(n)
        self._initial_weights = self._validate(
            uniform_weights(n)
            if initial_weights is None
            else np.asarray(initial_weights)
        )
        self._weights: FloatArray = self._initial_weights.copy()

    @property
    def name(self) -> str:
        """Human-readable learner name (used in plots and metrics tables)."""
        return type(self).__name__

    @property
    def weights(self) -> FloatArray:
        """The current allocation weights (a copy, to keep internal state safe)."""
        return self._weights.copy()

    def reset(self) -> None:
        """Restore the learner to its initial weights and clear any state."""
        self._weights = self._initial_weights.copy()
        self._reset_state()

    def update(self, realized_returns: FloatArray) -> FloatArray:
        """Observe one period of returns and return the next weight vector.

        Args:
            realized_returns: Length-``N`` vector of realized simple returns for
                each strategy over the period just ended.

        Returns:
            The new weight vector to use for the next period.
        """
        r = np.asarray(realized_returns, dtype=np.float64).ravel()
        if r.shape != (self.n,):
            raise ValueError(
                f"Expected realized_returns of shape ({self.n},), got {r.shape}."
            )
        self._weights = self._validate(self._step(r))
        return self.weights

    # --- Hooks for subclasses -------------------------------------------------

    @abc.abstractmethod
    def _step(self, r: FloatArray) -> FloatArray:
        """Compute the next weight vector from realized returns ``r``."""

    def _reset_state(self) -> None:
        """Clear any auxiliary accumulators. Overridden by stateful learners."""

    # --- Internal helpers -----------------------------------------------------

    @staticmethod
    def _validate(w: FloatArray) -> FloatArray:
        w = np.asarray(w, dtype=np.float64).ravel()
        if not is_on_simplex(w):
            raise ValueError(
                f"Learner produced weights off the simplex "
                f"(min={w.min():.3e}, sum={w.sum():.6f})."
            )
        return w
