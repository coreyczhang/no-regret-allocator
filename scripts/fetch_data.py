#!/usr/bin/env python3
"""Refresh the cached price panel via yfinance (optional dependency).

    python scripts/fetch_data.py --start 2010-01-01 --out data/prices.csv

Only needed to regenerate ``data/prices.csv``; the committed cache is enough to
run the backtest and the test suite offline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nra.backtest.data import (
    DEFAULT_CACHE_PATH,
    DEFAULT_TICKERS,
    fetch_prices,
)  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_CACHE_PATH)
    args = parser.parse_args(argv)

    prices = fetch_prices(args.tickers, start=args.start, end=args.end)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(args.out)
    print(f"Wrote {prices.shape[0]} rows x {prices.shape[1]} tickers to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
