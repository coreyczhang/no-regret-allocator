"""A small library of illustrative toy trading strategies.

Each strategy consumes a shared panel of historical prices (a ``pandas``
DataFrame indexed by date, one column per asset) and emits its own realized
per-period return series. Those series are the *inputs* the allocation learners
compete over — the point of this project is the meta-algorithm that allocates
across strategies, not the strategies themselves.

The strategies here are deliberately simple and are **illustrative only**. None
is presented as a good trading idea on its own.
"""

from nra.strategies.base import Strategy
from nra.strategies.buy_and_hold import BuyAndHold
from nra.strategies.momentum import Momentum
from nra.strategies.mean_reversion import MeanReversion
from nra.strategies.vol_target import VolatilityTarget

__all__ = [
    "Strategy",
    "BuyAndHold",
    "Momentum",
    "MeanReversion",
    "VolatilityTarget",
]
