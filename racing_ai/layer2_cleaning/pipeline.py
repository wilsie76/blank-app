"""Layer 2 — Cleaning.

- drop scratched / vacant boxes
- normalize odds to sane decimal range
- detect corrupted parser rows (NaN, missing fields)
- validate runner counts (>=4 to be useful)
"""
from __future__ import annotations
import logging
from ..schema import Race, Runner

log = logging.getLogger(__name__)


def _valid_runner(r: Runner) -> bool:
    if not r.runner_id or not r.name:
        return False
    if r.box <= 0:
        return False
    if r.win_odds is None or r.win_odds <= 1.0:
        return False
    return True


def _normalize_odds(r: Runner) -> None:
    # Coerce to safe decimal range
    if r.win_odds is None or r.win_odds <= 1.0:
        r.win_odds = 1.01
    if r.win_odds_open is None or r.win_odds_open <= 1.0:
        r.win_odds_open = r.win_odds
    if r.place_odds is None or r.place_odds <= 1.0:
        r.place_odds = max(1.05, 1.0 + (r.win_odds - 1.0) * 0.35)


def clean_race(race: Race) -> Race | None:
    """Returns the cleaned Race, or None if it should be discarded."""
    cleaned: list[Runner] = []
    for r in race.runners:
        if r.scratched:
            continue
        if not _valid_runner(r):
            log.warning("Dropping corrupted row: %s", r.runner_id)
            continue
        _normalize_odds(r)
        cleaned.append(r)

    if len(cleaned) < 4:
        log.warning("Race %s discarded: only %d valid runners", race.race_id, len(cleaned))
        return None

    race.runners = cleaned
    return race


def clean_races(races: list[Race]) -> list[Race]:
    out = []
    for race in races:
        cleaned = clean_race(race)
        if cleaned is not None:
            out.append(cleaned)
    return out
