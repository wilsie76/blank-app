"""Stub SportsBet adapter.

Replace with real implementation. Two viable paths:
  1. Authenticated session + their internal /api/racing endpoints (fragile).
  2. Headless browser scrape (Playwright) — slower but more resilient.
Always check the operator's ToS before automating anything.
"""
from __future__ import annotations
from datetime import date
from ..schema import Race
from .base import DataSource


class SportsBetSource(DataSource):
    name = "sportsbet"
    is_live = True

    def fetch_races(self, target_date: date | None = None) -> list[Race]:
        raise NotImplementedError(
            "SportsBet adapter not implemented. "
            "Plug in your scraper or API client here, returning a list[Race]."
        )
