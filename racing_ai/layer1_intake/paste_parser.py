"""Best-effort paste parser for race-card text dumps.

Targets the layout commonly seen on Australian/NZ tote/operator pages —
runner blocks like:

    1. Aston Verona (1)
    F: 542355
    T: Marcello Calicchio
    Early Speed:
    4.40
    3.30
    3.40

    3.30          <- current win price (decimal)

    1.75          <- current place price (decimal)
    Recent: ...
    ...
    Career  90: 8-13-21
    Win %  8.89%
    Place %  46.67%
    Trk/Dist  19: 0-2-5
    Best Time  23.03
    SB Rating  -
    Weight  27.2kg
    Recent Starts
    4 days since last race

This is a heuristic parser — it tries to be tolerant but will skip runners
it can't pin down. Always verify the parsed table against the source page.
"""
from __future__ import annotations
import re
from datetime import datetime
from typing import Optional

from ..schema import Race, Runner, RACE_TYPE_GREYHOUND, RACE_TYPE_THOROUGHBRED
from .manual import build_greyhound_race, build_thoroughbred_race


# Header line at the start of a runner block: "1. Aston Verona (1)"
_RUNNER_HEAD = re.compile(r"^\s*(\d{1,2})\.\s+([^()\n]+?)\s*\((\d{1,2})\)\s*$")
# "F: 542355" or "F: 542X5"
_FORM_LINE = re.compile(r"^\s*F:\s*([0-9Xx\-]+)\s*$")
# "T: Trainer Name"
_TRAINER_LINE = re.compile(r"^\s*T:\s*(.+?)\s*$")
# Plain decimal odds line
_PRICE_LINE = re.compile(r"^\s*(\d{1,4}(?:\.\d{1,2})?)\s*$")
# "Career   90: 8-13-21"
_CAREER_LINE = re.compile(r"^\s*Career\s+(\d+):\s*(\d+)-(\d+)-(\d+)\s*$")
# "Win %  8.89%"
_WIN_PCT = re.compile(r"^\s*Win\s*%\s+(-|[\d.]+)%?\s*$")
_PLACE_PCT = re.compile(r"^\s*Place\s*%\s+(-|[\d.]+)%?\s*$")
# "Trk/Dist  19: 0-2-5"
_TD_LINE = re.compile(r"^\s*Trk/Dist\s+(\d+):\s*(\d+)-(\d+)-(\d+)\s*$")
_BEST_TIME = re.compile(r"^\s*Best Time\s+(-|[\d.]+)\s*$")
_SB_RATING = re.compile(r"^\s*SB Rating\s+(-|\d+(?:\.\d+)?)\s*$")
_WEIGHT_LINE = re.compile(r"^\s*Weight\s+([\d.]+)\s*kg\s*$", re.IGNORECASE)
_DAYS_SINCE = re.compile(r"^\s*(\d+)\s+days?\s+since\s+last\s+race\s*$", re.IGNORECASE)
_SCRATCHED = re.compile(r"\bScratched\b", re.IGNORECASE)
_VACANT = re.compile(r"\bVacant\s+Box\b", re.IGNORECASE)


def _f(s: str) -> Optional[float]:
    if s is None or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _i(s: str) -> Optional[int]:
    if s is None or s == "-":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def parse_pasted_form(
    text: str,
    track: str,
    race_number: int,
    distance_m: int,
    track_condition: str = "good",
    race_type: str = RACE_TYPE_GREYHOUND,
    weather: str = "clear",
    start_time: datetime | None = None,
) -> Race:
    """Parse pasted race-card text into a Race object.

    The parser splits on runner-header lines ("N. Name (box)") and looks
    inside each block for the recognisable fields. Anything it can't pin
    down is left at a sensible default so the pipeline still runs.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    blocks: list[list[str]] = []
    current: list[str] = []

    for ln in lines:
        if _RUNNER_HEAD.match(ln):
            if current:
                blocks.append(current)
            current = [ln]
        elif current:
            current.append(ln)
    if current:
        blocks.append(current)

    runners: list[dict] = []
    for block in blocks:
        head = _RUNNER_HEAD.match(block[0])
        if not head:
            continue
        program_no = int(head.group(1))
        name = head.group(2).strip()
        box = int(head.group(3))

        # Skip vacant / scratched runners
        joined = "\n".join(block[:5])
        if _VACANT.search(joined) or _SCRATCHED.search(joined):
            continue

        d = {"box": box, "name": name}

        # Walk the block looking for fields
        prices: list[float] = []   # collect all standalone numerics
        for ln in block[1:]:
            stripped = ln.strip()
            if not stripped:
                continue

            if (m := _FORM_LINE.match(ln)):
                form = m.group(1).replace("-", "")
                d["last_6_form"] = form
                d["last_5_form"] = form[-5:]
                continue
            if (m := _TRAINER_LINE.match(ln)):
                d["trainer"] = m.group(1)
                continue
            if (m := _CAREER_LINE.match(ln)):
                d["career_starts"] = int(m.group(1))
                d["career_wins"] = int(m.group(2))
                # 2nds + 3rds = "places" in our schema
                d["career_places"] = int(m.group(3)) + int(m.group(4))
                continue
            if (m := _WIN_PCT.match(ln)):
                d["win_pct"] = _f(m.group(1))
                continue
            if (m := _PLACE_PCT.match(ln)):
                d["place_pct"] = _f(m.group(1))
                continue
            if (m := _TD_LINE.match(ln)):
                d["td_starts"] = int(m.group(1))
                d["td_wins"] = int(m.group(2))
                d["td_places"] = int(m.group(3)) + int(m.group(4))
                continue
            if (m := _BEST_TIME.match(ln)):
                d["best_time_s"] = _f(m.group(1))
                continue
            if (m := _SB_RATING.match(ln)):
                d["sb_rating"] = _f(m.group(1))
                continue
            if (m := _WEIGHT_LINE.match(ln)):
                d["weight_kg"] = float(m.group(1))
                continue
            if (m := _DAYS_SINCE.match(ln)):
                d["days_since_run"] = int(m.group(1))
                continue
            if (m := _PRICE_LINE.match(ln)):
                v = float(m.group(1))
                # Sensible decimal-odds range
                if 1.01 <= v <= 1000.0:
                    prices.append(v)

        # Heuristic: in the typical card, after the flucs block we see:
        #   <current win odds>  (often 1.50 - 200)
        #   <current place odds> (1.05 - 50)
        # Take the last two distinct values; if there's only one, treat it as win.
        if prices:
            # Take the most-recent block of prices (last 6) — earlier ones are flucs
            tail = prices[-6:]
            # Win odds = the largest of the trailing trio (often the current price)
            # Place odds = the smaller value < win
            if len(tail) >= 2:
                # The card's structural pattern: ..., open, fluc1, fluc2, win_curr, place_curr
                # Take the second-last as win and last as place — but ONLY if last < second-last.
                last, prev = tail[-1], tail[-2]
                if last < prev and last <= 50:
                    d["win_odds"] = prev
                    d["place_odds"] = last
                    # The third-last is often the open
                    if len(tail) >= 3:
                        d["win_odds_open"] = tail[-3]
                else:
                    d["win_odds"] = last
            else:
                d["win_odds"] = tail[-1]

        if "win_odds" not in d:
            # Couldn't find a price — we have to skip this runner
            continue

        runners.append(d)

    if race_type == RACE_TYPE_GREYHOUND:
        return build_greyhound_race(
            track=track, race_number=race_number, distance_m=distance_m,
            track_condition=track_condition, runners=runners,
            weather=weather, start_time=start_time,
        )
    return build_thoroughbred_race(
        track=track, race_number=race_number, distance_m=distance_m,
        track_condition=track_condition, runners=runners,
        weather=weather, start_time=start_time,
    )
