"""Hedge-ratio estimation: static OLS and a dynamic Kalman filter.

The hedge ratio ``beta`` says how many units of ``x`` to short against one unit of
``y`` so the combination ``y - beta * x`` is (hopefully) stationary. OLS gives one
fixed number from the in-sample window; a Kalman filter lets ``beta`` drift as the
relationship evolves, which is the standard way to keep a pair tradeable when the
hedge ratio is not truly constant (Chan, *Algorithmic Trading*, 2013).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OLSHedge:
    """Static hedge: ``y ≈ alpha + beta * x``."""

    beta: float
    alpha: float


def ols_hedge_ratio(y: pd.Series, x: pd.Series) -> OLSHedge:
    """Estimate a constant hedge ratio by ordinary least squares.

    Args:
        y: Dependent price series.
        x: Independent (hedge) price series.

    Returns:
        The fitted :class:`OLSHedge` (slope ``beta`` and intercept ``alpha``).
    """
    aligned = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
    slope, intercept = np.polyfit(aligned["x"].to_numpy(), aligned["y"].to_numpy(), 1)
    return OLSHedge(beta=float(slope), alpha=float(intercept))


@dataclass
class KalmanHedge:
    """Dynamic hedge path from a Kalman filter."""

    beta: pd.Series  # time-varying hedge ratio
    alpha: pd.Series  # time-varying intercept
    forecast_error: pd.Series  # measurement residual e_t (the tradeable spread)
    forecast_std: pd.Series  # sqrt predictive variance of e_t (for z-scoring)


def kalman_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    delta: float = 1e-4,
    obs_var: float = 1e-3,
) -> KalmanHedge:
    """Estimate a time-varying hedge ratio with a Kalman filter.

    The hidden state is ``[beta_t, alpha_t]`` following a random walk; the
    observation is ``y_t = beta_t * x_t + alpha_t + noise``. The one-step
    forecast error and its standard deviation are returned directly — their
    ratio is a self-normalizing spread z-score that needs no rolling window.

    Args:
        y: Dependent price series.
        x: Independent (hedge) price series.
        delta: Controls state drift; larger lets ``beta`` move faster.
        obs_var: Observation-noise variance.

    Returns:
        A :class:`KalmanHedge` with the ``beta``/``alpha`` paths and the
        forecast-error spread and its predictive standard deviation.
    """
    aligned = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
    yv = aligned["y"].to_numpy()
    xv = aligned["x"].to_numpy()
    n = len(yv)

    trans_cov = delta / (1.0 - delta) * np.eye(2)  # state random-walk covariance
    state: np.ndarray = np.zeros(2)  # [beta, alpha]
    cov: np.ndarray = np.ones((2, 2))  # diffuse-ish prior

    betas = np.empty(n)
    alphas = np.empty(n)
    errs = np.empty(n)
    stds = np.empty(n)

    for t in range(n):
        obs = np.array([xv[t], 1.0])  # observation matrix row H_t
        # Predict (random-walk transition => state unchanged, cov grows).
        cov_pred = cov + trans_cov
        # One-step forecast of y_t and its variance.
        y_hat = float(obs @ state)
        e = float(yv[t] - y_hat)
        s = float(obs @ cov_pred @ obs + obs_var)
        # Kalman gain and update.
        gain = cov_pred @ obs / s
        state = state + gain * e
        cov = cov_pred - np.outer(gain, obs) @ cov_pred

        betas[t] = state[0]
        alphas[t] = state[1]
        errs[t] = e
        stds[t] = np.sqrt(s)

    idx = aligned.index
    return KalmanHedge(
        beta=pd.Series(betas, index=idx, name="beta"),
        alpha=pd.Series(alphas, index=idx, name="alpha"),
        forecast_error=pd.Series(errs, index=idx, name="spread"),
        forecast_std=pd.Series(stds, index=idx, name="spread_std"),
    )
