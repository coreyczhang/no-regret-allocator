"""Turn the momentum signal into decile buckets and a long/short portfolio."""

from __future__ import annotations

import numpy as np
import pandas as pd


def quantile_buckets(signal: pd.DataFrame, n_quantiles: int = 10) -> pd.DataFrame:
    """Assign each name to a cross-sectional quantile bucket each month.

    Bucket ``1`` holds the lowest-signal names (losers), bucket ``n_quantiles``
    the highest (winners). Names with a missing signal get ``NaN``.

    Args:
        signal: Momentum signal (month index, one column per name).
        n_quantiles: Number of buckets.

    Returns:
        DataFrame of bucket labels in ``1..n_quantiles`` (float, ``NaN`` where
        the signal is undefined).
    """
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be >= 2.")
    rank = signal.rank(axis=1, method="first")
    count = signal.notna().sum(axis=1)
    bucket = np.ceil(rank.div(count, axis=0) * n_quantiles)
    return bucket.clip(lower=1, upper=n_quantiles)


def decile_returns(
    signal: pd.DataFrame, fwd: pd.DataFrame, n_quantiles: int = 10
) -> pd.DataFrame:
    """Equal-weight forward return of each quantile bucket, per month.

    Args:
        signal: Momentum signal.
        fwd: Forward (next-month) returns aligned to the signal's months.
        n_quantiles: Number of buckets.

    Returns:
        DataFrame indexed by month with columns ``1..n_quantiles`` giving each
        bucket's equal-weight forward return.
    """
    buckets = quantile_buckets(signal, n_quantiles)
    fwd = fwd.reindex_like(buckets)
    out = {}
    for b in range(1, n_quantiles + 1):
        masked = fwd.where(buckets == b)
        out[b] = masked.mean(axis=1)
    return pd.DataFrame(out).dropna(how="all")


def long_short_weights(signal: pd.DataFrame, n_quantiles: int = 10) -> pd.DataFrame:
    """Dollar-neutral long/short weights: long the top bucket, short the bottom.

    The long leg sums to ``+1`` and the short leg to ``-1`` each month (gross
    exposure 2, net 0), so the portfolio's return equals the top-minus-bottom
    decile spread.

    Args:
        signal: Momentum signal.
        n_quantiles: Number of buckets.

    Returns:
        Weight DataFrame aligned to ``signal`` (``0`` for names held flat).
    """
    buckets = quantile_buckets(signal, n_quantiles)
    longs = buckets == n_quantiles
    shorts = buckets == 1
    long_w = longs.div(longs.sum(axis=1), axis=0).fillna(0.0)
    short_w = shorts.div(shorts.sum(axis=1), axis=0).fillna(0.0)
    return (long_w - short_w).fillna(0.0)
