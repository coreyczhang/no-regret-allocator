"""A liquid, sector-labelled equity universe.

A curated set of ~120 large-cap US names spanning all sectors, chosen so that
(a) the momentum cross-section is wide enough for a meaningful decile study
(~12 names per decile), and (b) the pairs screener has many plausible
within-sector cointegration candidates (e.g. KO/PEP, XOM/CVX, HD/LOW, V/MA,
GS/MS, DUK/SO).

Names that listed after the 2007 start simply enter the cross-section once they
have enough history; the loader carries leading NaNs for them.

Survivorship bias, stated plainly: this is a *current*, hand-picked list of
survivors, so backtests on it are survivorship-biased and flatter any long-only
tilt. The momentum alpha is long/short and dollar-neutral, which blunts but does
not remove that bias; a production study would use point-in-time index
constituents. This is called out as a known limitation in the README.
"""

from __future__ import annotations

# Ticker -> GICS-style sector label. Sectors with several members double as the
# candidate pools for within-sector pair selection.
SECTOR_MAP: dict[str, str] = {
    # Information Technology
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "NVDA": "Information Technology",
    "AVGO": "Information Technology",
    "ORCL": "Information Technology",
    "CRM": "Information Technology",
    "ADBE": "Information Technology",
    "CSCO": "Information Technology",
    "INTC": "Information Technology",
    "AMD": "Information Technology",
    "QCOM": "Information Technology",
    "TXN": "Information Technology",
    "IBM": "Information Technology",
    "MU": "Information Technology",
    "AMAT": "Information Technology",
    "ADI": "Information Technology",
    "INTU": "Information Technology",
    "NOW": "Information Technology",
    # Communication Services
    "GOOGL": "Communication Services",
    "META": "Communication Services",
    "NFLX": "Communication Services",
    "DIS": "Communication Services",
    "CMCSA": "Communication Services",
    "T": "Communication Services",
    "VZ": "Communication Services",
    "TMUS": "Communication Services",
    "EA": "Communication Services",
    # Financials
    "JPM": "Financials",
    "BAC": "Financials",
    "WFC": "Financials",
    "C": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "AXP": "Financials",
    "USB": "Financials",
    "PNC": "Financials",
    "BK": "Financials",
    "SCHW": "Financials",
    "BLK": "Financials",
    "SPGI": "Financials",
    "CME": "Financials",
    "ICE": "Financials",
    "V": "Financials",
    "MA": "Financials",
    "AIG": "Financials",
    "MET": "Financials",
    "PRU": "Financials",
    "COF": "Financials",
    # Energy
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "SLB": "Energy",
    "EOG": "Energy",
    "OXY": "Energy",
    "HAL": "Energy",
    "VLO": "Energy",
    "WMB": "Energy",
    "OKE": "Energy",
    "DVN": "Energy",
    # Health Care
    "JNJ": "Health Care",
    "LLY": "Health Care",
    "UNH": "Health Care",
    "PFE": "Health Care",
    "MRK": "Health Care",
    "ABT": "Health Care",
    "TMO": "Health Care",
    "DHR": "Health Care",
    "MDT": "Health Care",
    "BMY": "Health Care",
    "AMGN": "Health Care",
    "GILD": "Health Care",
    "CVS": "Health Care",
    "ISRG": "Health Care",
    "SYK": "Health Care",
    "BSX": "Health Care",
    "BDX": "Health Care",
    # Consumer Staples
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "PG": "Consumer Staples",
    "WMT": "Consumer Staples",
    "COST": "Consumer Staples",
    "CL": "Consumer Staples",
    "MO": "Consumer Staples",
    "WBA": "Consumer Staples",
    "KMB": "Consumer Staples",
    "GIS": "Consumer Staples",
    "KR": "Consumer Staples",
    "SYY": "Consumer Staples",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "LOW": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",
    "TGT": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary",
    "TJX": "Consumer Discretionary",
    "ROST": "Consumer Discretionary",
    "YUM": "Consumer Discretionary",
    "ORLY": "Consumer Discretionary",
    "AZO": "Consumer Discretionary",
    "F": "Consumer Discretionary",
    # Industrials
    "GE": "Industrials",
    "CAT": "Industrials",
    "BA": "Industrials",
    "HON": "Industrials",
    "UNP": "Industrials",
    "RTX": "Industrials",
    "GD": "Industrials",
    "NOC": "Industrials",
    "LMT": "Industrials",
    "DE": "Industrials",
    "MMM": "Industrials",
    "UPS": "Industrials",
    "FDX": "Industrials",
    "CSX": "Industrials",
    "NSC": "Industrials",
    "EMR": "Industrials",
    "ETN": "Industrials",
    "ITW": "Industrials",
    # Utilities
    "NEE": "Utilities",
    "DUK": "Utilities",
    "SO": "Utilities",
    "D": "Utilities",
    "AEP": "Utilities",
    "EXC": "Utilities",
    "XEL": "Utilities",
    "SRE": "Utilities",
    "PEG": "Utilities",
    # Materials
    "APD": "Materials",
    "SHW": "Materials",
    "ECL": "Materials",
    "NUE": "Materials",
    "FCX": "Materials",
    "NEM": "Materials",
    "DD": "Materials",
    "PPG": "Materials",
    # Real Estate
    "AMT": "Real Estate",
    "PLD": "Real Estate",
    "EQIX": "Real Estate",
    "PSA": "Real Estate",
    "O": "Real Estate",
    "SPG": "Real Estate",
    "CCI": "Real Estate",
}

# Deterministic, sorted ticker list.
EQUITY_UNIVERSE: list[str] = sorted(SECTOR_MAP)


def sectors() -> list[str]:
    """Return the sorted list of distinct sector labels."""
    return sorted(set(SECTOR_MAP.values()))


def tickers_in_sector(sector: str) -> list[str]:
    """Return the sorted tickers belonging to ``sector``."""
    return sorted(t for t, s in SECTOR_MAP.items() if s == sector)
