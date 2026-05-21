"""DataSource contract — implement once per upstream provider."""
from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import date
from ..schema import Race


class DataSource(ABC):
    name: str = "abstract"
    is_live: bool = False

    @abstractmethod
    def fetch_races(self, target_date: date | None = None) -> list[Race]:
        """Return all races for the given date (defaults to today)."""

    def fetch_closing_odds(self, race_id: str) -> dict[str, float]:
        """Return {runner_id: closing_decimal_odds}. Optional — used for CLV."""
        return {}
