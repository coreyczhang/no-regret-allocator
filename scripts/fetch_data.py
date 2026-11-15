#!/usr/bin/env python3
"""Refresh a cached price panel via yfinance (optional dependency).

    # ETF panel used by the allocator demo:
    python scripts/fetch_data.py --start 2010-01-01 --out data/prices.csv

    # Wide equity universe used by the momentum & stat-arb alphas:
    python scripts/fetch_data.py --equities

Only needed to regenerate the committed caches; the shipped CSVs are enough to
run the backtests and the test suite offline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nra.backtest.data import (  # noqa: E402
    DEFAULT_CACHE_PATH,
    DEFAULT_TICKERS,
    fetch_prices,
)
from nra.data.loader import EQUITIES_CACHE_PATH, fetch_equity_prices  # noqa: E402
from nra.data.universe import EQUITY_UNIVERSE  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--equities",
        action="store_true",
        help="Fetch the wide equity universe (alphas) instead of the ETF panel.",
    )
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.equities:
        tickers = args.tickers or list(EQUITY_UNIVERSE)
        prices = fetch_equity_prices(
            tickers,
            start=args.start or "2007-01-01",
            end=args.end or "2024-01-01",
        )
        out = args.out or EQUITIES_CACHE_PATH
    else:
        tickers = args.tickers or list(DEFAULT_TICKERS)
        prices = fetch_prices(tickers, start=args.start or "2010-01-01", end=args.end)
        out = args.out or DEFAULT_CACHE_PATH

    out.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(out)
    print(f"Wrote {prices.shape[0]} rows x {prices.shape[1]} tickers to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
