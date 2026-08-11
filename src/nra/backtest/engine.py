"""Run loop, hindsight benchmarks, and performance metrics.

Timing convention
-----------------
At period ``t`` the learner already holds weights ``w_t`` (decided from returns
observed through ``t-1``). The strategies then realize returns ``r_t``, the
learner earns ``w_t . r_t``, and only afterwards does it call ``update(r_t)`` to
form ``w_{t+1}``. This ordering is what keeps the backtest free of look-ahead:
no weight is ever informed by the very return it is about to earn.

Regret
------
The learners optimize the *linear* per-period reward ``w . r`` (this is exactly
the "prediction with expert advice" / Hedge setting that EG, OGD, and Regret
Matching all bound). Regret is therefore measured against the best fixed weight
vector in hindsight under that same linear objective,

    Regret_T = max_{u in simplex} sum_t u . r_t  -  sum_t w_t . r_t .

Because the objective is linear, the maximizing ``u`` is a vertex of the
simplex, i.e. the single best strategy by summed return. The no-regret property
is the empirical fact that ``Regret_T / T -> 0``. For the multiplicative
wealth picture we additionally report the best *constant-rebalanced portfolio*
(the interior comparator behind Cover's 1991 universal portfolios), computed by
:func:`best_constant_rebalanced`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from nra.learners.base import Learner
from nra.simplex import euclidean_projection_simplex, uniform_weights

FloatArray = NDArray[np.float64]


@dataclass
class BacktestResult:
    """Per-period series and summary metrics from one learner's backtest."""

    name: str
    dates: pd.DatetimeIndex
    weights: FloatArray  # shape (T, N): weight held during each period
    portfolio_returns: FloatArray  # shape (T,): realized w_t . r_t
    wealth: FloatArray  # shape (T,): cumulative product of (1 + return)
    turnover: FloatArray  # shape (T,): L1 change in weights entering the period
    regret: FloatArray  # shape (T,): running external regret vs best fixed weight
    metrics: dict[str, float] = field(default_factory=dict)

    def as_frame(self) -> pd.DataFrame:
        """Return the per-period series as a tidy DataFrame indexed by date."""
        return pd.DataFrame(
            {
                "portfolio_return": self.portfolio_returns,
                "wealth": self.wealth,
                "turnover": self.turnover,
                "regret": self.regret,
                "avg_regret": self.regret / np.arange(1, len(self.dates) + 1),
            },
            index=self.dates,
        )


def _sharpe(returns: FloatArray, periods_per_year: int) -> float:
    """Annualized Sharpe ratio (zero risk-free rate), 0 if volatility is 0."""
    sd = returns.std(ddof=1) if returns.size > 1 else 0.0
    if sd == 0.0:
        return 0.0
    return float(returns.mean() / sd * np.sqrt(periods_per_year))


def best_fixed_weight(returns: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Best fixed weight under the linear objective ``sum_t u . r_t``.

    The maximizer is the vertex on the single strategy with the largest summed
    return, so this returns that one-hot weight vector and the running cumulative
    reward path of the best-in-hindsight strategy.

    Args:
        returns: ``(T, N)`` array of per-period strategy returns.

    Returns:
        ``(weight, cumulative_reward_path)`` where ``weight`` is the one-hot
        best fixed allocation and the path has length ``T``.
    """
    cumulative = np.cumsum(returns, axis=0)  # (T, N)
    best_idx = int(np.argmax(cumulative[-1]))
    weight = np.zeros(returns.shape[1], dtype=np.float64)
    weight[best_idx] = 1.0
    # Running best-in-hindsight-so-far cumulative reward, for the regret curve.
    running_best = cumulative.max(axis=1).astype(np.float64)
    return weight, running_best


def best_constant_rebalanced(
    returns: FloatArray,
    iterations: int = 5000,
    step: float = 0.5,
    tol: float = 1e-10,
) -> tuple[FloatArray, FloatArray]:
    """Best constant-rebalanced portfolio by log-wealth (projected gradient).

    Maximizes the concave objective ``sum_t log(1 + u . r_t)`` over the simplex,
    the hindsight comparator behind Cover's universal portfolios. Solved with
    projected gradient ascent using the Euclidean simplex projection.

    Args:
        returns: ``(T, N)`` array of per-period strategy returns.
        iterations: Maximum ascent iterations.
        step: Ascent step size.
        tol: Early-stop tolerance on the objective improvement.

    Returns:
        ``(weight, wealth_path)`` where ``weight`` is the optimal CRP and
        ``wealth_path`` is its cumulative multiplicative wealth of length ``T``.
    """
    n = returns.shape[1]
    w = uniform_weights(n)
    prev_obj = -np.inf
    for _ in range(iterations):
        gross = 1.0 + returns @ w  # (T,)
        # Guard against a weight that would bankrupt the portfolio on some date.
        gross = np.maximum(gross, 1e-12)
        obj = float(np.log(gross).sum())
        grad = (returns / gross[:, None]).sum(axis=0)  # d/dw sum log(1 + r.w)
        w = euclidean_projection_simplex(w + step * grad / len(returns))
        if abs(obj - prev_obj) < tol:
            break
        prev_obj = obj
    wealth_path = np.cumprod(1.0 + returns @ w).astype(np.float64)
    return w, wealth_path


def run_backtest(
    learner: Learner,
    returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> BacktestResult:
    """Run one learner over the return panel and collect series and metrics.

    Args:
        learner: The allocation learner (its state is reset before running).
        returns: ``(T, N)`` DataFrame of per-period strategy returns.
        periods_per_year: Annualization factor for the Sharpe ratio.

    Returns:
        A populated :class:`BacktestResult`.
    """
    learner.reset()
    r = returns.to_numpy(dtype=np.float64)
    n_periods, n_strategies = r.shape
    if learner.n != n_strategies:
        raise ValueError(
            f"Learner expects {learner.n} strategies but the panel has "
            f"{n_strategies} columns."
        )

    weights = np.empty((n_periods, n_strategies), dtype=np.float64)
    port_returns = np.empty(n_periods, dtype=np.float64)
    turnover = np.zeros(n_periods, dtype=np.float64)

    w = learner.weights  # w_0
    prev_w = w
    for t in range(n_periods):
        weights[t] = w
        port_returns[t] = float(w @ r[t])
        turnover[t] = float(np.abs(w - prev_w).sum()) if t > 0 else 0.0
        prev_w = w
        w = learner.update(r[t])  # w_{t+1}

    wealth = np.cumprod(1.0 + port_returns).astype(np.float64)
    cumulative_reward = np.cumsum(port_returns)
    _, running_best = best_fixed_weight(r)
    regret = running_best - cumulative_reward

    horizon = np.arange(1, n_periods + 1)
    metrics = {
        "total_return": float(wealth[-1] - 1.0),
        "sharpe": _sharpe(port_returns, periods_per_year),
        "avg_turnover": float(turnover[1:].mean()) if n_periods > 1 else 0.0,
        "final_regret": float(regret[-1]),
        "avg_regret": float(regret[-1] / n_periods),
        "final_avg_regret": float((regret / horizon)[-1]),
    }

    return BacktestResult(
        name=learner.name,
        dates=returns.index,
        weights=weights,
        portfolio_returns=port_returns,
        wealth=wealth,
        turnover=turnover,
        regret=regret,
        metrics=metrics,
    )
