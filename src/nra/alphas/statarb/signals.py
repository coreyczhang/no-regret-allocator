"""Z-score trading rules and pair PnL.

Trade the spread's mean reversion with hysteresis: enter when the spread is
stretched beyond ``entry`` standard deviations, exit as it reverts toward the
mean (``exit_``), and stop out if it keeps diverging past ``stop`` — the signature
that the relationship may have broken, the central risk of pairs trading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_positions(
    z: pd.Series,
    entry: float = 2.0,
    exit_: float = 0.5,
    stop: float = 4.0,
) -> pd.Series:
    """Convert a spread z-score into a {-1, 0, +1} position on the spread.

    ``+1`` is long the spread (spread unusually low, betting it rises); ``-1`` is
    short the spread. Positions are stateful: once in a trade we hold until the
    spread reverts past ``exit_`` or diverges past ``stop``.

    Args:
        z: Spread z-score series (NaN warmup treated as flat).
        entry: Entry threshold in standard deviations.
        exit_: Reversion threshold at which a winning trade is closed.
        stop: Divergence threshold at which a losing trade is stopped out.

    Returns:
        The position to hold *into the next period* (use ``.shift(1)`` for PnL).
    """
    pos = np.zeros(len(z))
    current = 0
    values = z.to_numpy()
    for t, zt in enumerate(values):
        if not np.isfinite(zt):
            current = 0
        elif current == 0:
            # Enter only when stretched past `entry` but not already beyond the
            # `stop` band — never open a position that is already blown out.
            if -stop < zt <= -entry:
                current = 1
            elif entry <= zt < stop:
                current = -1
        elif current == 1:  # long spread
            if zt >= -exit_ or zt <= -stop:
                current = 0
        elif current == -1:  # short spread
            if zt <= exit_ or zt >= stop:
                current = 0
        pos[t] = current
    return pd.Series(pos, index=z.index, name="position")


def pair_returns(
    ret_y: pd.Series,
    ret_x: pd.Series,
    positions: pd.Series,
    beta: float | pd.Series,
    cost_bps: float = 5.0,
) -> pd.Series:
    """Per-period return of one pair trade, net of transaction costs.

    Long one unit of the spread means long ``y`` and short ``beta`` units of
    ``x``; the return is normalized by the gross notional ``1 + |beta|`` so pairs
    are comparable and combine sensibly in a portfolio. Positions are lagged one
    period so the signal never trades on contemporaneous information.

    Args:
        ret_y: Simple returns of ``y``.
        ret_x: Simple returns of ``x``.
        positions: Spread positions from :func:`generate_positions`.
        beta: Hedge ratio (scalar, or a time-varying series for Kalman).
        cost_bps: Round-trip cost per unit position change, in basis points.

    Returns:
        The pair's net per-period return series.
    """
    df = pd.concat([ret_y, ret_x], axis=1, keys=["y", "x"])
    if isinstance(beta, pd.Series):
        b = beta.reindex(df.index).ffill()
    else:
        b = pd.Series(float(beta), index=df.index)
    gross = 1.0 + b.abs()
    held = positions.reindex(df.index).shift(1).fillna(0.0)
    raw = held * (df["y"] - b * df["x"]) / gross
    cost = (cost_bps / 1e4) * positions.reindex(df.index).diff().abs().fillna(0.0)
    return (raw - cost).rename("pair_return").astype(np.float64)
