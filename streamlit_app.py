"""Fixtures viewer.

Loads `fixtures.json` and renders matches grouped by date and competition.
For each match, displays the bookmaker odds, normalized implied probabilities
for each outcome, and a "Favored outcome" column that re-states which side
the bookmaker has shortest odds on. Nothing here is a prediction; it is a
restatement of the odds you provided.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from parser import merge_match_days, parse_fixtures_text

FIXTURES_PATH = Path(__file__).parent / "fixtures.json"


def load_fixtures(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_fixtures(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


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


def favored_outcome(home_pct: float, draw_pct: float, away_pct: float, home: str, away: str):
    """Return (label, side, top_pct) for the outcome with highest implied probability.

    label is a human-readable string like "Home win (Envigado)".
    side is one of "Home", "Draw", "Away".
    """
    options = [
        ("Home", home_pct, f"Home win ({home})"),
        ("Draw", draw_pct, "Draw"),
        ("Away", away_pct, f"Away win ({away})"),
    ]
    side, top_pct, label = max(options, key=lambda x: x[1])
    return label, side, top_pct


def matches_to_dataframe(matches: list[dict]) -> pd.DataFrame:
    rows = []
    for m in matches:
        p_home, p_draw, p_away, overround = implied_probabilities(
            m["home_odds"], m["draw_odds"], m["away_odds"]
        )
        label, side, top_pct = favored_outcome(p_home, p_draw, p_away, m["home"], m["away"])
        # Confidence "gap" = lead of favored outcome over the next-best one.
        sorted_pcts = sorted([p_home, p_draw, p_away], reverse=True)
        gap = sorted_pcts[0] - sorted_pcts[1]
        rows.append(
            {
                "Match": f"{m['home']} vs {m['away']}",
                "Home odds": m["home_odds"],
                "Draw odds": m["draw_odds"],
                "Away odds": m["away_odds"],
                "Home %": round(p_home, 1),
                "Draw %": round(p_draw, 1),
                "Away %": round(p_away, 1),
                "Favored outcome": label,
                "Favored %": round(top_pct, 1),
                "Lead over next %": round(gap, 1),
                "Margin %": round(overround, 1),
                "_side": side,  # internal, used for filtering
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Fixtures viewer", page_icon="📅", layout="wide")
    st.title("📅 Fixtures viewer")
    st.caption(
        "Bookmaker odds with normalized implied probabilities. "
        "The 'Favored outcome' column simply re-states which side has the shortest odds. "
        "These are not predictions."
    )

    data = load_fixtures(FIXTURES_PATH)
    match_days = data.get("match_days", [])

    # --- Sidebar: add new matches by pasting raw text ---
    with st.sidebar.expander("➕ Add matches", expanded=False):
        st.caption(
            "Paste a listing in the same format as your original message: "
            "a date line, then competition headers, then each match as "
            "`home / odds / Draw / odds / away / odds`."
        )
        pasted = st.text_area("Paste fixtures text", height=200, key="paste_text")
        col_a, col_b = st.columns(2)
        preview = col_a.button("Preview")
        save = col_b.button("Save to fixtures.json", type="primary")

        if preview or save:
            try:
                parsed = parse_fixtures_text(pasted or "")
            except Exception as exc:  # pragma: no cover - defensive
                st.error(f"Could not parse text: {exc}")
                parsed = []

            n_matches = sum(
                len(c["matches"]) for d in parsed for c in d["competitions"]
            )
            if n_matches == 0:
                st.warning("No matches detected. Check the format.")
            else:
                st.success(
                    f"Parsed {n_matches} match(es) across "
                    f"{sum(len(d['competitions']) for d in parsed)} competition(s) "
                    f"on {len(parsed)} date(s)."
                )
                with st.expander("Parsed preview (JSON)", expanded=False):
                    st.json(parsed)

                if save:
                    merged, counters = merge_match_days(match_days, parsed)
                    data["match_days"] = merged
                    save_fixtures(FIXTURES_PATH, data)
                    st.success(
                        f"Saved. Added {counters['matches_added']} match(es); "
                        f"skipped {counters['matches_skipped']} duplicate(s); "
                        f"new dates: {counters['days_added']}, "
                        f"new competitions: {counters['competitions_added']}."
                    )
                    st.rerun()

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

    favored_filter = st.sidebar.multiselect(
        "Favored outcome",
        options=["Home", "Draw", "Away"],
        default=["Home", "Draw", "Away"],
        help="Show only matches where the bookmaker's shortest-odds outcome is one of these.",
    )

    min_confidence = st.sidebar.slider(
        "Min favored % (implied)",
        min_value=34,
        max_value=95,
        value=34,
        help="Minimum implied probability of the favored outcome. 34% is roughly a 3-way coin flip.",
    )

    sort_mode = st.sidebar.radio(
        "Sort matches by",
        options=[
            "Original order",
            "Highest favored %",
            "Largest lead over next outcome",
            "Lowest bookmaker margin",
        ],
        index=0,
    )

    view_mode = st.sidebar.radio(
        "View",
        options=["Grouped by date and competition", "Flat ranked list"],
        index=0,
    )

    # Build a flat dataframe of every match (for flat view / global sort).
    all_rows = []
    for day in match_days:
        if day["date"] not in selected_dates:
            continue
        for comp in day.get("competitions", []):
            if comp["name"] not in selected_comps:
                continue
            df = matches_to_dataframe(comp.get("matches", []))
            if df.empty:
                continue
            df.insert(0, "Date", day["date"])
            df.insert(1, "Competition", comp["name"])
            all_rows.append(df)

    if not all_rows:
        st.warning("No matches match the current filters.")
        return

    flat = pd.concat(all_rows, ignore_index=True)
    flat = flat[flat["_side"].isin(favored_filter)]
    flat = flat[flat["Favored %"] >= min_confidence]

    if flat.empty:
        st.warning("No matches match the current filters.")
        return

    sort_keys = {
        "Highest favored %": ("Favored %", False),
        "Largest lead over next outcome": ("Lead over next %", False),
        "Lowest bookmaker margin": ("Margin %", True),
    }

    display_cols = [
        "Match",
        "Home odds",
        "Draw odds",
        "Away odds",
        "Home %",
        "Draw %",
        "Away %",
        "Favored outcome",
        "Favored %",
        "Lead over next %",
        "Margin %",
    ]

    if view_mode == "Flat ranked list":
        if sort_mode in sort_keys:
            col, ascending = sort_keys[sort_mode]
            flat = flat.sort_values(col, ascending=ascending)
        st.subheader(f"All matches ({len(flat)})")
        st.dataframe(
            flat[["Date", "Competition", *display_cols]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        # Grouped view: sort within each (date, competition) block.
        for date_value in [d for d in all_dates if d in selected_dates]:
            day_df = flat[flat["Date"] == date_value]
            if day_df.empty:
                continue
            st.header(date_value)
            for comp_name in sorted(day_df["Competition"].unique()):
                comp_df = day_df[day_df["Competition"] == comp_name].copy()
                if sort_mode in sort_keys:
                    col, ascending = sort_keys[sort_mode]
                    comp_df = comp_df.sort_values(col, ascending=ascending)
                st.subheader(comp_name)
                st.dataframe(
                    comp_df[display_cols],
                    use_container_width=True,
                    hide_index=True,
                )

    st.sidebar.markdown("---")
    st.sidebar.metric("Matches shown", len(flat))


if __name__ == "__main__":
    main()
