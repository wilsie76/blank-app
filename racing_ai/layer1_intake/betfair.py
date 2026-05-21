"""Stub Betfair adapter.

Betfair is the cleanest source — real exchange API with documented endpoints.
Use betfairlightweight (Python client). You'll need an app key + cert login.
"""
from __future__ import annotations
from datetime import date
from ..schema import Race
from .base import DataSource


class BetfairSource(DataSource):
    name = "betfair"
    is_live = True

    def fetch_races(self, target_date: date | None = None) -> list[Race]:
        raise NotImplementedError(
            "Betfair adapter not implemented. Recommended: pip install betfairlightweight, "
            "then call list_market_catalogue + list_market_book for HORSE_RACING."
        )
