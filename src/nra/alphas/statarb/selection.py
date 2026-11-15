"""In-sample pair selection: within-sector correlation screen + cointegration.

Two series are *cointegrated* if a linear combination of them is stationary even
though each is individually non-stationary — exactly the property a mean-reverting
spread needs. We screen candidate pairs by price correlation (cheap), then apply
the Engle-Granger two-step test (``statsmodels.tsa.stattools.coint``) to the log
prices and keep pairs whose residual is stationary at the chosen level.

Only within-sector pairs are considered: an economic reason for the relationship
(same business drivers) guards against the spurious cointegration that blind
all-pairs mining produces. The multiple-testing exposure is reported so the
reader can judge how many "significant" pairs to expect by chance alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


@dataclass(frozen=True)
class CandidatePair:
    """A screened, cointegration-tested pair (all stats are in-sample)."""

    a: str
    b: str
    sector: str
    correlation: float
    coint_pvalue: float


@dataclass(frozen=True)
class SelectionReport:
    """Selected pairs plus the multiple-testing context."""

    pairs: list[CandidatePair]
    n_tested: int
    n_passed: int
    pvalue_threshold: float

    @property
    def expected_false_positives(self) -> float:
        """Pairs expected to pass by chance alone at this threshold."""
        return self.n_tested * self.pvalue_threshold


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    sector_map: dict[str, str],
    corr_threshold: float = 0.80,
    pvalue_threshold: float = 0.05,
    min_obs: int = 252,
) -> SelectionReport:
    """Select within-sector cointegrated pairs from an in-sample price panel.

    Args:
        prices: In-sample daily prices only (date index, one column per name).
            Pass the pre-split slice — this function does not withhold data.
        sector_map: Ticker -> sector label; only same-sector pairs are tested.
        corr_threshold: Minimum log-price correlation to bother testing.
        pvalue_threshold: Engle-Granger p-value to accept cointegration.
        min_obs: Minimum overlapping observations required to test a pair.

    Returns:
        A :class:`SelectionReport` with the accepted pairs (sorted by p-value)
        and the count of pairs tested, for multiple-testing context.
    """
    log_px = np.log(prices)
    by_sector: dict[str, list[str]] = {}
    for ticker in prices.columns:
        sector = sector_map.get(ticker)
        if sector is not None:
            by_sector.setdefault(sector, []).append(ticker)

    accepted: list[CandidatePair] = []
    n_tested = 0
    for sector, members in by_sector.items():
        for a, b in combinations(sorted(members), 2):
            pair = log_px[[a, b]].dropna()
            if len(pair) < min_obs:
                continue
            corr = float(pair[a].corr(pair[b]))
            if not np.isfinite(corr) or corr < corr_threshold:
                continue
            n_tested += 1
            # Engle-Granger: regress a on b, ADF-test the residual.
            _, pvalue, _ = coint(pair[a], pair[b])
            if pvalue < pvalue_threshold:
                accepted.append(CandidatePair(a, b, sector, corr, float(pvalue)))

    accepted.sort(key=lambda p: p.coint_pvalue)
    return SelectionReport(
        pairs=accepted,
        n_tested=n_tested,
        n_passed=len(accepted),
        pvalue_threshold=pvalue_threshold,
    )
