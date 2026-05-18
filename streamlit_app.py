"""Fixtures viewer.

Loads `fixtures.json` and renders matches grouped by date and competition.
For each match, displays the bookmaker odds and the implied probability for
each outcome. Implied probabilities are normalized to remove the bookmaker
margin (overround), so the three probabilities sum to 100%.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

FIXTURES_PATH = Path(__file__).parent / "fixtures.json"


@st.cache_data
def load_fixtures(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def implied_probabilities(home_odds: float, draw_odds: float, away_odds: float):
    """Return (p_home, p_draw, p_away, overround) as percentages.

    Raw implied probability is 1/odds. The three raw values typically sum to
    more than 1 (the bookmaker margin / overround). We normalize by dividing
    each raw probability by their sum to get fair probabilities.
    """
    raw_home = 1.0 / home_odds
    raw_draw = 1.0 / draw_odds
    raw_away = 1.0 / away_odds
    total = raw_home + raw_draw + raw_away
    overround_pct = (total - 1.0) * 100.0
    return (
        raw_home / total * 100.0,
        raw_draw / total * 100.0,
        raw_away / total * 100.0,
        overround_pct,
    )


def matches_to_dataframe(matches: list[dict]) -> pd.DataFrame:
    rows = []
    for m in matches:
        p_home, p_draw, p_away, overround = implied_probabilities(
            m["home_odds"], m["draw_odds"], m["away_odds"]
        )
        rows.append(
            {
                "Match": f"{m['home']} vs {m['away']}",
                "Home odds": m["home_odds"],
                "Draw odds": m["draw_odds"],
                "Away odds": m["away_odds"],
                "Home %": round(p_home, 1),
                "Draw %": round(p_draw, 1),
                "Away %": round(p_away, 1),
                "Margin %": round(overround, 1),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Fixtures viewer", page_icon="📅", layout="wide")
    st.title("📅 Fixtures viewer")
    st.caption(
        "Bookmaker odds with normalized implied probabilities. "
        "Margin % is the bookmaker overround (how much the raw 1/odds exceed 100%)."
    )

    data = load_fixtures(FIXTURES_PATH)
    match_days = data.get("match_days", [])

    if not match_days:
        st.info("No fixtures found in fixtures.json.")
        return

    # Sidebar filters
    all_dates = [day["date"] for day in match_days]
    selected_dates = st.sidebar.multiselect(
        "Match days", options=all_dates, default=all_dates
    )

    all_comps = sorted(
        {
            comp["name"]
            for day in match_days
            for comp in day.get("competitions", [])
        }
    )
    selected_comps = st.sidebar.multiselect(
        "Competitions", options=all_comps, default=all_comps
    )

    total_matches = 0
    for day in match_days:
        if day["date"] not in selected_dates:
            continue

        comps = [
            c for c in day.get("competitions", []) if c["name"] in selected_comps
        ]
        if not comps:
            continue

        st.header(day["date"])
        for comp in comps:
            matches = comp.get("matches", [])
            if not matches:
                continue
            total_matches += len(matches)
            st.subheader(comp["name"])
            df = matches_to_dataframe(matches)
            st.dataframe(df, use_container_width=True, hide_index=True)

    st.sidebar.markdown("---")
    st.sidebar.metric("Matches shown", total_matches)


if __name__ == "__main__":
    main()
