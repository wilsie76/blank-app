"""Streamlit UI for the AFL SGM builder.

Exposes a single function `render()` that draws the entire tab. Call it
from streamlit_app.py inside `with tab_sgm:`.

Design notes
------------
* The Smart Paste pattern is intentional, mirroring race_predictor.html and
  the Smart Paste section of streamlit_app.py.
* Mode toggle is the most important control. Banker maximises P(win) -- the
  "ensure best winning outcome" objective.
* The leg table is sortable and supports manual selection. The "Auto-build"
  button runs the optimiser on the current pool and toggles its choices on.
* The live SGM panel re-runs combine_legs() on every selection change so the
  user sees the price/prob impact immediately.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
import streamlit as st

from .parser import parse_paste
from .model import score_legs
from .schema import Leg, SGMMode
from .optimizer import optimise_sgm, ticket_ladder
from .combiner import combine_legs
from .correlation import avg_pair_correlation


_FIXTURE = Path(__file__).parent / "fixtures" / "geelong_v_sydney.txt"


# --- session state helpers ------------------------------------------------

def _ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def _legs_to_df(legs: list[Leg], selected_ids: set[str]) -> pd.DataFrame:
    rows = []
    for leg in legs:
        rows.append({
            "Pick": leg.leg_id in selected_ids,
            "Player / Team": leg.player or leg.team or "",
            "Market": leg.description.replace((leg.player or leg.team or "") + " ", "", 1),
            "Price": leg.price,
            "Hits": f"{leg.hits}/{leg.n_games}" if leg.n_games else "-",
            "Model %": round(leg.model_prob * 100, 1),
            "Market %": round((1.0 / leg.price) * 100, 1) if leg.price > 0 else 0,
            "EV %": round(leg.ev_pct, 1),
            "Conf": leg.confidence,
            "Flags": ", ".join(leg.flags) if leg.flags else "",
            "_id": leg.leg_id,
        })
    return pd.DataFrame(rows)


# --- main render ----------------------------------------------------------

def render():
    """Draw the AFL SGM tab. Call from inside `with tab_sgm:`."""
    st.markdown("### AFL Same Game Multi Builder")
    st.caption(
        "Paste any AFL match's market dump (Sportsbet/TAB/etc.). "
        "The parser pulls out every player prop with last-5 form, "
        "models the probability, and the optimiser picks legs to "
        "**maximise P(SGM wins)** -- not raw EV. Switch to Value mode for "
        "edge-hunting instead."
    )

    legs: list[Leg] = _ss("sgm_legs", [])
    selected_ids: set[str] = _ss("sgm_selected", set())

    # ------------------------------------------------------------------
    # 1. Paste & parse
    # ------------------------------------------------------------------
    with st.expander("1. Paste market dump", expanded=not legs):
        c1, c2 = st.columns([3, 1])
        with c1:
            paste = st.text_area(
                "Paste the entire 'Hot Legs' section (or full match page)",
                value=_ss("sgm_paste", ""),
                key="sgm_paste",
                height=180,
                placeholder=(
                    "9\nMax Holmes\n25+ Disposals\n28\n25\n29\n33\n26\n\n1.65\n..."
                ),
            )
        with c2:
            st.markdown("&nbsp;")
            if st.button("Parse paste", use_container_width=True, type="primary"):
                parsed = parse_paste(paste)
                score_legs(parsed)
                st.session_state["sgm_legs"] = parsed
                st.session_state["sgm_selected"] = set()
                legs = parsed
                selected_ids = set()
                st.success(f"Parsed {len(parsed)} legs.")

            if st.button("Load demo (Cats v Swans)", use_container_width=True):
                if _FIXTURE.exists():
                    parsed = parse_paste(_FIXTURE.read_text())
                    score_legs(parsed)
                    st.session_state["sgm_legs"] = parsed
                    st.session_state["sgm_selected"] = set()
                    st.session_state["sgm_paste"] = _FIXTURE.read_text()
                    legs = parsed
                    selected_ids = set()
                    st.success(f"Demo loaded: {len(parsed)} legs.")
                else:
                    st.error("Demo fixture missing.")

            if st.button("Clear all", use_container_width=True):
                st.session_state["sgm_legs"] = []
                st.session_state["sgm_selected"] = set()
                st.session_state["sgm_paste"] = ""
                legs = []
                selected_ids = set()

    if not legs:
        st.info("Paste a market dump above (or click **Load demo**) to start.")
        return

    # ------------------------------------------------------------------
    # 2. Optimiser controls + auto-build
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 2. Build")

    cols = st.columns([1.4, 1, 1, 1, 1.2])
    with cols[0]:
        mode = st.radio(
            "Mode",
            ["Banker (max P win)", "Value (max EV)"],
            horizontal=True,
            index=0,
            help=(
                "Banker keeps the SGM as safe as possible, biasing toward "
                "legs that visibly cleared the threshold in last-5. Value "
                "hunts +EV legs even if win probability is moderate."
            ),
        )
        mode_enum = SGMMode.BANKER if mode.startswith("Banker") else SGMMode.VALUE
    with cols[1]:
        target_legs = st.slider("Legs", 3, 15, 15)
    with cols[2]:
        min_book_price = st.number_input(
            "Min book $", min_value=1.5, max_value=50.0, value=3.0, step=0.5,
            help="Banker mode floor: refuses to drop below this estimated payout."
        )
    with cols[3]:
        min_per_leg_ev = st.number_input(
            "Min leg EV %", min_value=-30.0, max_value=20.0, value=-10.0, step=1.0,
            help="Drop legs whose model EV is below this. Set to 0 for strict +EV only."
        )
    with cols[4]:
        if st.button("Auto-build SGM", use_container_width=True, type="primary"):
            ticket = optimise_sgm(
                legs, mode=mode_enum, target_legs=target_legs,
                min_book_price=min_book_price, min_per_leg_ev_pct=min_per_leg_ev,
            )
            st.session_state["sgm_selected"] = {l.leg_id for l in ticket.legs}
            selected_ids = st.session_state["sgm_selected"]
            st.success(f"Auto-built {ticket.n_legs}-leg ticket.")

    # ------------------------------------------------------------------
    # 3. Leg table with checkboxes
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 3. Legs")
    df = _legs_to_df(legs, selected_ids)
    df_sorted = df.sort_values(by=["Pick", "Model %"], ascending=[False, False])
    edited = st.data_editor(
        df_sorted,
        use_container_width=True,
        height=420,
        hide_index=True,
        column_config={
            "Pick": st.column_config.CheckboxColumn(),
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Model %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            "Market %": st.column_config.NumberColumn(format="%.1f%%"),
            "EV %": st.column_config.NumberColumn(format="%+.1f%%"),
            "Conf": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
            "_id": None,
        },
        disabled=["Player / Team", "Market", "Price", "Hits", "Model %",
                  "Market %", "EV %", "Conf", "Flags"],
        key="sgm_editor",
    )

    # Update selection from edit
    new_selected = {row["_id"] for _, row in edited.iterrows() if row["Pick"]}
    if new_selected != selected_ids:
        st.session_state["sgm_selected"] = new_selected
        selected_ids = new_selected

    # ------------------------------------------------------------------
    # 4. Live combined SGM panel
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 4. Combined SGM")

    selected_legs = [l for l in legs if l.leg_id in selected_ids]
    if not selected_legs:
        st.info("Tick legs in the table above (or use **Auto-build**) to see the combined ticket.")
        return

    ticket = combine_legs(selected_legs, mode=mode_enum.value)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Legs", ticket.n_legs)
    m2.metric("P(win)", f"{ticket.combined_prob*100:.1f}%")
    m3.metric("Est book price", f"${ticket.estimated_book_price:.2f}",
              help=("Approximation of Sportsbet's quote after their "
                    "correlation discount. Their exact matrix is private; "
                    "compare to the actual bookie quote when placing."))
    m4.metric("Model fair price", f"${ticket.fair_price:.2f}",
              help="Zero-edge price implied by the model. If Sportsbet "
                   "offers above this, the bet is +EV.")
    m5.metric("Model EV", f"{ticket.combined_ev_pct:+.1f}%")

    avg_rho = avg_pair_correlation(selected_legs)
    st.caption(
        f"Avg pairwise correlation: **{avg_rho:+.2f}** "
        f"(higher = legs share signal -- safer but smaller payout). "
        f"Naive product price: ${ticket.naive_price:.2f} (book won't offer this)."
    )

    if ticket.warnings:
        for w in ticket.warnings:
            st.warning(w)

    # Show selected legs as a clean list
    rows = []
    for leg in ticket.legs:
        rows.append({
            "Leg": leg.description,
            "Price": leg.price,
            "Hits": f"{leg.hits}/{leg.n_games}" if leg.n_games else "-",
            "Model %": round(leg.model_prob * 100, 1),
            "EV %": round(leg.ev_pct, 1),
        })
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Model %": st.column_config.NumberColumn(format="%.1f%%"),
            "EV %": st.column_config.NumberColumn(format="%+.1f%%"),
        },
    )

    # ------------------------------------------------------------------
    # 5. Ticket ladder (compare leg counts side by side)
    # ------------------------------------------------------------------
    with st.expander("5. Ticket ladder -- compare leg counts"):
        st.caption("Re-runs the optimiser at multiple leg counts so you can "
                   "see the prob/price tradeoff and pick the sweet spot.")
        ladder = ticket_ladder(
            legs, mode=mode_enum,
            min_book_price=min_book_price,
            min_per_leg_ev_pct=min_per_leg_ev,
        )
        ladder_df = pd.DataFrame([{
            "Legs": t.n_legs,
            "P(win) %": round(t.combined_prob * 100, 1),
            "Est book $": round(t.estimated_book_price, 2),
            "Fair $": round(t.fair_price, 2),
            "EV %": round(t.combined_ev_pct, 1),
        } for t in ladder])
        st.dataframe(ladder_df, use_container_width=True, hide_index=True,
                     column_config={
                         "P(win) %": st.column_config.ProgressColumn(
                             min_value=0, max_value=100, format="%.1f%%"),
                         "Est book $": st.column_config.NumberColumn(format="$%.2f"),
                         "Fair $": st.column_config.NumberColumn(format="$%.2f"),
                         "EV %": st.column_config.NumberColumn(format="%+.1f%%"),
                     })
