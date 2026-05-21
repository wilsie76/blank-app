"""Layer 1 — Data intake.

The base.DataSource interface is the contract every adapter must satisfy.
SyntheticSource is fully working. SportsBet/TAB/Betfair are stubs you must
fill in with real scraping/API code (and respect each operator's ToS).
"""
from .base import DataSource
from .synthetic import SyntheticSource
from .sportsbet import SportsBetSource
from .tab import TabSource
from .betfair import BetfairSource

__all__ = [
    "DataSource",
    "SyntheticSource",
    "SportsBetSource",
    "TabSource",
    "BetfairSource",
    "get_source",
]


def get_source(name: str) -> DataSource:
    name = name.lower()
    return {
        "synthetic": SyntheticSource(),
        "sportsbet": SportsBetSource(),
        "tab": TabSource(),
        "betfair": BetfairSource(),
    }[name]
