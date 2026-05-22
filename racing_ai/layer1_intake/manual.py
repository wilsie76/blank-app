"""Manual race intake.

The point of this adapter is to let you feed real form data into the pipeline
without a scraper. Two ways:

  1. Programmatic — call `build_greyhound_race(...)` or
     `build_thoroughbred_race(...)` with explicit dicts.
  2. Paste — call `parse_pasted_form(text, race_type=...)` to convert the
     raw text dump from a tote/operator's race card into a Race object.

Then wrap the resulting list[Race] in a ManualSource and run the Pipeline
against it like any other source.
"""
from __future__ import annotations
from datetime import date, datetime
from typing import Optional

from ..schema import Race, Runner, RACE_TYPE_GREYHOUND, RACE_TYPE_THOROUGHBRED
from .base import DataSource


class ManualSource(DataSource):
    """A trivial DataSource backed by an in-memory list of Race objects."""
    name = "manual"
    is_live = True

    def __init__(self, races: Optional[list[Race]] = None):
        self._races: list[Race] = list(races or [])

    def add(self, race: Race) -> None:
        self._races.append(race)

    def replace(self, races: list[Race]) -> None:
        self._races = list(races)

    def fetch_races(self, target_date: date | None = None) -> list[Race]:
        return list(self._races)


# ============================================================================
# Builders
# ============================================================================

def _slug(s: str) -> str:
    return "".join(c for c in s.upper() if c.isalnum())[:6] or "RAC"


def build_greyhound_race(
    track: str,
    race_number: int,
    distance_m: int,
    track_condition: str,
    runners: list[dict],
    weather: str = "clear",
    surface: str = "track",
    start_time: datetime | None = None,
    race_id: str | None = None,
) -> Race:
    """Build a greyhound Race from a list of runner dicts.

    Required runner keys:
        box, name, win_odds
    Optional runner keys:
        last_6_form, trainer, win_odds_open, place_odds, weight_kg,
        days_since_run, best_time_s, win_pct, place_pct, career_starts,
        career_wins, career_places, td_starts, td_wins, td_places,
        sb_rating, sire, dam, scratched
    """
    start_time = start_time or datetime.now()
    rid = race_id or f"{start_time.date().isoformat()}_{_slug(track)}_R{race_number}_GH"

    built: list[Runner] = []
    for r in runners:
        box = int(r["box"])
        name = str(r["name"])
        win_odds = float(r["win_odds"])
        win_odds_open = float(r.get("win_odds_open", win_odds))
        place_odds = float(r.get("place_odds", max(1.05, 1.0 + (win_odds - 1.0) * 0.35)))
        last_6 = str(r.get("last_6_form", ""))
        last_5 = str(r.get("last_5_form", last_6[-5:] if last_6 else ""))

        built.append(Runner(
            runner_id=f"{rid}_BOX{box}",
            name=name,
            box=box,
            weight_kg=float(r.get("weight_kg", 30.0)),
            jockey="",
            trainer=str(r.get("trainer", "")),
            last_5_form=last_5,
            days_since_run=int(r.get("days_since_run", 14)),
            win_odds_open=win_odds_open,
            win_odds=win_odds,
            place_odds=place_odds,
            scratched=bool(r.get("scratched", False)),
            last_6_form=last_6,
            best_time_s=_optf(r.get("best_time_s")),
            win_pct=_optf(r.get("win_pct")),
            place_pct=_optf(r.get("place_pct")),
            career_starts=_opti(r.get("career_starts")),
            career_wins=_opti(r.get("career_wins")),
            career_places=_opti(r.get("career_places")),
            td_starts=_opti(r.get("td_starts")),
            td_wins=_opti(r.get("td_wins")),
            td_places=_opti(r.get("td_places")),
            sb_rating=_optf(r.get("sb_rating")),
            sire=str(r.get("sire", "")),
            dam=str(r.get("dam", "")),
        ))

    return Race(
        race_id=rid,
        track=track,
        race_number=race_number,
        distance_m=distance_m,
        surface=surface,
        track_condition=track_condition,
        weather=weather,
        start_time=start_time,
        runners=built,
        race_type=RACE_TYPE_GREYHOUND,
    )


def build_thoroughbred_race(
    track: str,
    race_number: int,
    distance_m: int,
    track_condition: str,
    runners: list[dict],
    weather: str = "clear",
    surface: str = "turf",
    start_time: datetime | None = None,
    race_id: str | None = None,
) -> Race:
    """Build a thoroughbred Race from a list of runner dicts.

    Required runner keys:
        box (barrier), name, win_odds, jockey, trainer, weight_kg
    Optional runner keys:
        last_5_form, last_6_form, days_since_run, win_odds_open, place_odds,
        win_pct, place_pct, career_starts, career_wins, career_places,
        td_starts, td_wins, td_places, sb_rating, sire, dam, scratched
    """
    start_time = start_time or datetime.now()
    rid = race_id or f"{start_time.date().isoformat()}_{_slug(track)}_R{race_number}_TB"

    built: list[Runner] = []
    for r in runners:
        box = int(r["box"])
        name = str(r["name"])
        win_odds = float(r["win_odds"])
        win_odds_open = float(r.get("win_odds_open", win_odds))
        place_odds = float(r.get("place_odds", max(1.05, 1.0 + (win_odds - 1.0) * 0.35)))
        last_5 = str(r.get("last_5_form", ""))

        built.append(Runner(
            runner_id=f"{rid}_BARR{box}",
            name=name,
            box=box,
            weight_kg=float(r.get("weight_kg", 58.0)),
            jockey=str(r.get("jockey", "")),
            trainer=str(r.get("trainer", "")),
            last_5_form=last_5,
            days_since_run=int(r.get("days_since_run", 21)),
            win_odds_open=win_odds_open,
            win_odds=win_odds,
            place_odds=place_odds,
            scratched=bool(r.get("scratched", False)),
            last_6_form=str(r.get("last_6_form", last_5)),
            best_time_s=_optf(r.get("best_time_s")),
            win_pct=_optf(r.get("win_pct")),
            place_pct=_optf(r.get("place_pct")),
            career_starts=_opti(r.get("career_starts")),
            career_wins=_opti(r.get("career_wins")),
            career_places=_opti(r.get("career_places")),
            td_starts=_opti(r.get("td_starts")),
            td_wins=_opti(r.get("td_wins")),
            td_places=_opti(r.get("td_places")),
            sb_rating=_optf(r.get("sb_rating")),
            sire=str(r.get("sire", "")),
            dam=str(r.get("dam", "")),
        ))

    return Race(
        race_id=rid,
        track=track,
        race_number=race_number,
        distance_m=distance_m,
        surface=surface,
        track_condition=track_condition,
        weather=weather,
        start_time=start_time,
        runners=built,
        race_type=RACE_TYPE_THOROUGHBRED,
    )


def _optf(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _opti(v) -> int | None:
    try:
        return int(float(v)) if v is not None else None
    except (TypeError, ValueError):
        return None
