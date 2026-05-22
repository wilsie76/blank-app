"""Layer 1 — Data intake.

The base.DataSource interface is the contract every adapter must satisfy.
SyntheticSource is fully working. ManualSource lets you feed real form data
in by hand (programmatically or via the paste parser). SportsBet/TAB/Betfair
are stubs you must fill in with real scraping/API code (and respect each
operator's ToS).
"""
from .base import DataSource
from .synthetic import SyntheticSource
from .sportsbet import SportsBetSource
from .tab import TabSource
from .betfair import BetfairSource
from .manual import ManualSource, build_greyhound_race, build_thoroughbred_race
from .paste_parser import parse_pasted_form

__all__ = [
    "DataSource",
    "SyntheticSource",
    "SportsBetSource",
    "TabSource",
    "BetfairSource",
    "ManualSource",
    "build_greyhound_race",
    "build_thoroughbred_race",
    "parse_pasted_form",
    "get_source",
]


def get_source(name: str) -> DataSource:
    name = name.lower()
    return {
        "synthetic": SyntheticSource(),
        "sportsbet": SportsBetSource(),
        "tab": TabSource(),
        "betfair": BetfairSource(),
        "manual": ManualSource(),
    }[name]
