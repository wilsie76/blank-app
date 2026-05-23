"""Racing AI — Streamlit dashboard.

Tabs:
  1. Today's Card  — pick a race, see model probs + EV + sim combos + no-bet flags
  2. Backtest      — retrain on synthetic data and inspect per-model accuracy
  3. Performance   — ROI / CLV / hit-rate / miss-type breakdown
  4. Settings      — current config snapshot
"""
from __future__ import annotations
import pandas as pd
import streamlit as st
import plotly.express as px

from racing_ai.config import (
    MIN_EV_PCT, MAX_VOLATILITY, MIN_CONFIDENCE, KELLY_FRACTION,
    DEFAULT_SIMULATIONS, MIN_BANKROLL,
)
from racing_ai.layer4_models import Ensemble, MODELS_AVAILABLE, train_and_save
from racing_ai.layer7_learning import LearningTracker
from racing_ai.pipeline import Pipeline
from racing_ai.storage import get_db


st.set_page_config(page_title="Racing AI", layout="wide", page_icon="🏇")


# --- session state --------------------------------------------------------
if "data_source" not in st.session_state:
    st.session_state.data_source = "synthetic"
if "n_sims" not in st.session_state:
    st.session_state.n_sims = DEFAULT_SIMULATIONS
if "races" not in st.session_state:
    st.session_state.races = None


@st.cache_resource(show_spinner="Booting pipeline (training ensemble on first run)...")
def get_pipeline(source: str) -> Pipeline:
    return Pipeline(source=source, auto_train=True)


# --- sidebar --------------------------------------------------------------
with st.sidebar:
    st.title("🏇 Racing AI")
    st.caption("7-layer prediction stack")

    st.session_state.data_source = st.selectbox(
        "Data source", ["synthetic", "sportsbet", "tab", "betfair"],
        index=["synthetic", "sportsbet", "tab", "betfair"].index(st.session_state.data_source),
        help="SportsBet/TAB/Betfair are stubs — implement the adapter to use them.",
    )
    st.session_state.n_sims = st.slider("Monte Carlo simulations", 1000, 50000, st.session_state.n_sims, 1000)

    if st.button("⟳ Fetch races", use_container_width=True, type="primary"):
        try:
            pipe = get_pipeline(st.session_state.data_source)
            st.session_state.races = pipe.run(n_simulations=st.session_state.n_sims)
            st.success(f"Loaded {len(st.session_state.races)} races")
        except NotImplementedError as e:
            st.error(str(e))

    if st.button("🧠 Retrain ensemble", use_container_width=True):
        get_pipeline.clear()
        ens, scores, n_rows = train_and_save(n_meetings=60)
        st.success(f"Trained on {n_rows} runner-rows. Train acc: " +
                   ", ".join(f"{k}={v:.3f}" for k, v in scores.items()))

    st.divider()
    st.caption("Models loaded:")
    for k, v in MODELS_AVAILABLE.items():
        st.write(("✅" if v else "⚪") + f" {k}")


# --- tabs -----------------------------------------------------------------
tab_card, tab_sgm, tab_backtest, tab_perf, tab_settings = st.tabs(
    ["Today's Card", "AFL SGM", "Backtest", "Performance", "Settings"]
)


# =========================================================================
with tab_card:
    races = st.session_state.races
    if not races:
        st.info("Hit **Fetch races** in the sidebar to load today's card.")
    else:
        labels = [f"R{r.race_number} · {r.track} · {r.distance_m}m" for r, _ in races]
        idx = st.selectbox("Race", range(len(labels)), format_func=lambda i: labels[i])
        race, sim = races[idx]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Distance", f"{race.distance_m} m")
        c2.metric("Condition", race.track_condition)
        c3.metric("Pace pressure", f"{race.pace_pressure:.2f}")
        c4.metric("Track bias", f"{race.track_bias:.2f}")

        # Runner-level table
        rows = []
        for r in sorted(race.active_runners, key=lambda x: -x.win_prob):
            rows.append({
                "Box": r.box,
                "Runner": r.name,
                "Odds": r.win_odds,
                "Model P(win)": round(r.win_prob, 4),
                "P(place)": round(r.place_prob, 4),
                "EV %": round(r.ev_pct, 2),
                "Confidence": round(r.confidence, 3),
                "Volatility": round(r.features.get("volatility_score", 0), 3),
                "Steam": round(r.features.get("steam_index", 0), 3),
                "Kelly %": round(r.kelly_stake * 100, 2),
                "No-bet": "🚫" if r.no_bet else "✅",
                "Reason": r.no_bet_reason,
            })
        df = pd.DataFrame(rows)
        st.subheader("Runner-by-runner predictions")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Bet recording
        bet_candidates = df[df["No-bet"] == "✅"]
        if not bet_candidates.empty:
            st.success(f"{len(bet_candidates)} value bet(s) detected")
            with st.expander("Record a bet (optional)"):
                runner_choice = st.selectbox("Runner", bet_candidates["Runner"].tolist())
                stake = st.number_input("Stake ($)", min_value=1.0, value=20.0, step=5.0)
                if st.button("Place bet"):
                    runner = next(r for r in race.active_runners if r.name == runner_choice)
                    bid = get_db().record_bet(
                        race.race_id, runner.runner_id, "win", stake, runner.win_odds
                    )
                    st.success(f"Bet #{bid} recorded — {stake:.2f} @ {runner.win_odds:.2f}")
        else:
            st.warning("No bets meet the EV/confidence/volatility filters in this race.")

        st.subheader("Top combos (Monte Carlo)")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.caption(f"Exacta — top {len(sim.exacta)}")
            st.dataframe(pd.DataFrame(
                [(" → ".join(c), round(p, 4)) for c, p in sim.exacta],
                columns=["Combo", "Prob"]), hide_index=True, use_container_width=True)
        with cc2:
            st.caption(f"Trifecta — top {len(sim.trifecta)}")
            st.dataframe(pd.DataFrame(
                [(" → ".join(c), round(p, 4)) for c, p in sim.trifecta],
                columns=["Combo", "Prob"]), hide_index=True, use_container_width=True)
        with cc3:
            st.caption(f"First-4 — top {len(sim.first4)}")
            st.dataframe(pd.DataFrame(
                [(" → ".join(c), round(p, 4)) for c, p in sim.first4],
                columns=["Combo", "Prob"]), hide_index=True, use_container_width=True)


# =========================================================================
with tab_sgm:
    from afl_sgm.streamlit_tab import render as render_sgm_tab
    render_sgm_tab()


# =========================================================================
with tab_backtest:
    st.subheader("Train ensemble on synthetic data")
    st.caption("In production this would query your historical DB instead of synthesising.")
    n_meetings = st.slider("Synthetic meetings", 10, 200, 60, step=10)
    if st.button("Train now"):
        get_pipeline.clear()
        ens, scores, n_rows = train_and_save(n_meetings=n_meetings)
        st.success(f"Trained on {n_rows} runner-rows.")
        st.json(scores)


# =========================================================================
with tab_perf:
    tracker = LearningTracker()
    snap = tracker.snapshot()
    if snap.n_bets == 0:
        st.info("No bets recorded yet. Place some on the Today's Card tab to populate metrics.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bets", snap.n_bets)
        c2.metric("Hit rate", f"{snap.hit_rate*100:.1f}%")
        c3.metric("ROI", f"{snap.roi_pct:.1f}%")
        c4.metric("CLV", f"{snap.avg_clv_pct:.1f}%")
        st.metric("Profit", f"${snap.profit:.2f}")

        eq = tracker.equity_curve()
        if not eq.empty:
            fig = px.line(eq, x="placed_at", y="cumulative_pnl", title="Equity curve")
            st.plotly_chart(fig, use_container_width=True)

        if snap.by_miss_type:
            st.subheader("Miss-type breakdown")
            mt = pd.DataFrame(list(snap.by_miss_type.items()), columns=["Type", "Count"])
            st.bar_chart(mt.set_index("Type"))


# =========================================================================
with tab_settings:
    st.subheader("Active configuration")
    st.json({
        "MIN_EV_PCT": MIN_EV_PCT,
        "MAX_VOLATILITY": MAX_VOLATILITY,
        "MIN_CONFIDENCE": MIN_CONFIDENCE,
        "KELLY_FRACTION": KELLY_FRACTION,
        "DEFAULT_SIMULATIONS": DEFAULT_SIMULATIONS,
        "MIN_BANKROLL": MIN_BANKROLL,
    })
    st.caption("Edit `racing_ai/config.py` to change thresholds. App restart required.")

    with st.expander("Architecture (7 layers)"):
        st.markdown("""
        | Layer | Module | Purpose |
        |---|---|---|
        | 1 | `layer1_intake` | DataSource adapters (synthetic working; SB/TAB/Betfair stubs) |
        | 2 | `layer2_cleaning` | Drop scratched, normalize odds, validate runners |
        | 3 | `layer3_features` | 13 features: pace, bias, steam, fatigue, etc. |
        | 4 | `layer4_models` | XGBoost + LightGBM + CatBoost + MLP ensemble |
        | 5 | `layer5_simulation` | Plackett-Luce Monte Carlo for exacta/trifecta/first4 |
        | 6 | `layer6_value` | EV%, Kelly stake, no-bet filters |
        | 7 | `layer7_learning` | ROI / CLV / miss-type tracking |
        """)
