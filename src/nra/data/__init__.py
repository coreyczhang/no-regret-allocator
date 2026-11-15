"""Data layer: equity universe, sector map, and price loading.

This package backs the two alpha-research layers (momentum and statistical
arbitrage). It is deliberately kept separate from :mod:`nra.backtest.data`,
which loads the small ETF panel used by the allocator demo — the alphas need a
much wider, sector-labelled equity cross-section.

A cached CSV of adjusted close prices ships with the repository so everything
runs offline and reproducibly; :func:`fetch_equity_prices` (optional
``yfinance``) only exists to regenerate that cache.
"""

from nra.data.loader import (
    EQUITIES_CACHE_PATH,
    fetch_equity_prices,
    load_equity_prices,
)
from nra.data.universe import (
    EQUITY_UNIVERSE,
    SECTOR_MAP,
    sectors,
    tickers_in_sector,
)

__all__ = [
    "EQUITIES_CACHE_PATH",
    "fetch_equity_prices",
    "load_equity_prices",
    "EQUITY_UNIVERSE",
    "SECTOR_MAP",
    "sectors",
    "tickers_in_sector",
]
