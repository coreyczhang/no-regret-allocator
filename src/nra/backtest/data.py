"""Historical price data: cached CSV loading and optional yfinance refresh.

The repository ships a cached CSV of adjusted close prices so the backtest is
fully reproducible offline and continuous integration never touches the network.
:func:`fetch_prices` (which requires the optional ``yfinance`` dependency) is
provided only to regenerate that cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from nra.strategies.base import Strategy

# A handful of liquid, diversified ETFs: US large cap, small cap, tech, long
# Treasuries, and gold. Chosen only to give the toy strategies varied return
# streams to allocate across -- not a recommended universe.
DEFAULT_TICKERS: tuple[str, ...] = ("SPY", "QQQ", "IWM", "TLT", "GLD")

# Cached prices live alongside the repo root's data/ directory.
_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
DEFAULT_CACHE_PATH = _DATA_DIR / "prices.csv"


def load_prices(path: str | Path = DEFAULT_CACHE_PATH) -> pd.DataFrame:
    """Load the cached price panel from CSV.

    Args:
        path: CSV path with a date index in the first column.

    Returns:
        DataFrame of prices indexed by date, one column per ticker.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached prices at {path}. Regenerate them with "
            f"scripts/fetch_data.py (requires the optional 'yfinance' extra)."
        )
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    return prices.sort_index().dropna(how="all")


def fetch_prices(
    tickers: Sequence[str] = DEFAULT_TICKERS,
    start: str = "2010-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices with yfinance (optional dependency).

    Args:
        tickers: Ticker symbols to download.
        start: Inclusive start date (YYYY-MM-DD).
        end: Optional exclusive end date; defaults to the latest available.

    Returns:
        DataFrame of adjusted close prices indexed by date, columns = tickers.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "fetch_prices requires 'yfinance'. Install it with "
            "`pip install no-regret-allocator[data]`."
        ) from exc

    raw = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    # With auto_adjust=True the adjusted price is under "Close".
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close[list(tickers)] if set(tickers).issubset(close.columns) else close
    return close.dropna(how="all").sort_index()


def strategy_return_panel(
    strategies: Iterable[Strategy], prices: pd.DataFrame
) -> pd.DataFrame:
    """Run each strategy and align results onto a shared return panel.

    Args:
        strategies: Strategies to evaluate on the shared price panel.
        prices: Price panel passed to every strategy.

    Returns:
        DataFrame of per-period strategy returns, one column per strategy,
        restricted to the dates where *every* strategy has a defined return.
    """
    series = [s.compute_returns(prices) for s in strategies]
    panel = pd.concat(series, axis=1)
    return panel.dropna(how="any")
