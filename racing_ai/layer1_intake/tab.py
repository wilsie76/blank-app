"""Stub TAB adapter — fill in with real client."""
from __future__ import annotations
from datetime import date
from ..schema import Race
from .base import DataSource


class TabSource(DataSource):
    name = "tab"
    is_live = True

    def fetch_races(self, target_date: date | None = None) -> list[Race]:
        raise NotImplementedError(
            "TAB adapter not implemented. Their public meeting JSON endpoints "
            "are usable, e.g. https://api.beta.tab.com.au/v1/tab-info-service/racing/dates/..."
        )
