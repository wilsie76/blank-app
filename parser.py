"""Parser for raw pasted fixture listings.

Expected format (one match block):

    Date line:        e.g. "Tuesday, 19 May"  or  "Tuesday, 19 May 2026"
    Competition:      e.g. "Colombian Primera B"
    Home team
    home_odds (float)
    "Draw"
    draw_odds (float)
    Away team
    away_odds (float)

Blank lines and trailer lines (e.g. "Must be", responsible-gambling notices)
are ignored. Lines that don't fit any pattern are treated as a new competition
header.
"""

from __future__ import annotations

import re
from typing import Any

DATE_RE = re.compile(
    r"^[A-Z][a-z]+,\s+\d{1,2}\s+[A-Z][a-z]+(\s+\d{4})?$"
)
NUM_RE = re.compile(r"^\d+(?:\.\d+)?$")


def parse_fixtures_text(text: str) -> list[dict[str, Any]]:
    """Parse pasted text into a list of match-day dicts.

    Returns a list shaped like the entries in fixtures.json's `match_days`.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    days: list[dict[str, Any]] = []
    cur_day: dict[str, Any] | None = None
    cur_comp: dict[str, Any] | None = None
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]

        if DATE_RE.match(line):
            cur_day = {"date": line, "competitions": []}
            days.append(cur_day)
            cur_comp = None
            i += 1
            continue

        if cur_day is None:
            # Skip anything before the first date line.
            i += 1
            continue

        # Try to parse a match block of 6 lines starting at i.
        if (
            i + 5 < n
            and NUM_RE.match(lines[i + 1])
            and lines[i + 2] == "Draw"
            and NUM_RE.match(lines[i + 3])
            and NUM_RE.match(lines[i + 5])
        ):
            if cur_comp is None:
                # Match block with no preceding competition header; skip it.
                i += 6
                continue
            cur_comp["matches"].append(
                {
                    "home": line,
                    "away": lines[i + 4],
                    "home_odds": float(lines[i + 1]),
                    "draw_odds": float(lines[i + 3]),
                    "away_odds": float(lines[i + 5]),
                }
            )
            i += 6
            continue

        # Not a match block: treat as competition header.
        cur_comp = {"name": line, "matches": []}
        cur_day["competitions"].append(cur_comp)
        i += 1

    # Drop empty competitions and empty days.
    cleaned: list[dict[str, Any]] = []
    for d in days:
        comps = [c for c in d["competitions"] if c["matches"]]
        if comps:
            cleaned.append({"date": d["date"], "competitions": comps})
    return cleaned


def merge_match_days(
    existing: list[dict[str, Any]], new: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Merge `new` match days into `existing`.

    Rules:
    - Match days are matched by exact `date` string.
    - Competitions within a date are matched by exact `name`.
    - A match is considered duplicate if (home, away) match an existing entry
      in the same competition. Duplicates are skipped (existing odds win).

    Returns the merged list and a dict of counters.
    """
    counters = {"days_added": 0, "competitions_added": 0, "matches_added": 0, "matches_skipped": 0}
    by_date = {d["date"]: d for d in existing}

    for new_day in new:
        if new_day["date"] not in by_date:
            existing.append({"date": new_day["date"], "competitions": []})
            by_date[new_day["date"]] = existing[-1]
            counters["days_added"] += 1
        target_day = by_date[new_day["date"]]
        comps_by_name = {c["name"]: c for c in target_day["competitions"]}

        for new_comp in new_day["competitions"]:
            if new_comp["name"] not in comps_by_name:
                target_day["competitions"].append({"name": new_comp["name"], "matches": []})
                comps_by_name[new_comp["name"]] = target_day["competitions"][-1]
                counters["competitions_added"] += 1
            target_comp = comps_by_name[new_comp["name"]]
            existing_pairs = {(m["home"], m["away"]) for m in target_comp["matches"]}

            for m in new_comp["matches"]:
                if (m["home"], m["away"]) in existing_pairs:
                    counters["matches_skipped"] += 1
                    continue
                target_comp["matches"].append(m)
                counters["matches_added"] += 1

    return existing, counters
