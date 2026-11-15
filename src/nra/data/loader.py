"""Load the cached equity price panel, or refresh it from yfinance.

The cached CSV (`data/equities.csv`) is the source of truth for the test suite
and the committed figures. :func:`fetch_equity_prices` is only needed to
regenerate it and requires the optional ``yfinance`` dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from nra.data.universe import EQUITY_UNIVERSE

# The repo-root data/ directory (…/no-regret-allocator/data).
_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
EQUITIES_CACHE_PATH = _DATA_DIR / "equities.csv"


def load_equity_prices(
    path: str | Path = EQUITIES_CACHE_PATH,
    tickers: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load the cached equity price panel from CSV.

    Args:
        path: CSV with a date index in the first column, one column per ticker.
        tickers: Optional subset to return (defaults to every cached column).

    Returns:
        DataFrame of adjusted close prices indexed by date, sorted ascending,
        with all-NaN rows dropped.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached equity prices at {path}. Regenerate them with "
            f"scripts/fetch_data.py --equities (requires the 'data' extra)."
        )
    prices = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if tickers is not None:
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            raise KeyError(f"Tickers not in cache: {missing}")
        prices = prices[list(tickers)]
    return prices.dropna(how="all")


def fetch_equity_prices(
    tickers: Sequence[str] = tuple(EQUITY_UNIVERSE),
    start: str = "2007-01-01",
    end: str | None = "2024-01-01",
) -> pd.DataFrame:
    """Download adjusted close prices with yfinance (optional dependency).

    Args:
        tickers: Ticker symbols to download.
        start: Inclusive start date (YYYY-MM-DD).
        end: Optional exclusive end date.

    Returns:
        DataFrame of adjusted close prices indexed by date, columns = tickers.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "fetch_equity_prices requires 'yfinance'. Install it with "
            "`pip install no-regret-allocator[data]`."
        ) from exc

    raw = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    # Preserve universe order where possible; drop tickers that failed entirely.
    cols = [t for t in tickers if t in close.columns]
    return close[cols].dropna(how="all").sort_index().round(4)
