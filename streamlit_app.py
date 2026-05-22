"""Racing AI — Streamlit dashboard.

Tabs:
  1. Today's Card       Pick a race, see model probs + EV + sim combos + no-bet flags.
  2. Manual Race        Paste form text or fill a structured form, get predictions
                        without needing a live data adapter.
  3. Bot Insights       What the RaceLearningBot has learned (calibration drift,
                        per-context multipliers, weight updates, plain-English findings).
  4. Backtest           Retrain the thoroughbred ensemble on synthetic data.
  5. Performance        ROI / CLV / hit rate / miss-type breakdown.
  6. Settings           Active config snapshot.
"""
from __future__ import annotations
import json
import pandas as pd
import streamlit as st
import plotly.express as px

from racing_ai.config import (
    MIN_EV_PCT, MAX_VOLATILITY, MIN_CONFIDENCE, KELLY_FRACTION,
    DEFAULT_SIMULATIONS, MIN_BANKROLL,
)
from racing_ai.layer1_intake import (
    ManualSource, build_greyhound_race, build_thoroughbred_race, parse_pasted_form,
)
from racing_ai.layer4_models import Ensemble, MODELS_AVAILABLE, train_and_save
from racing_ai.layer7_learning import LearningTracker, RaceLearningBot
from racing_ai.pipeline import Pipeline
from racing_ai.schema import RACE_TYPE_GREYHOUND, RACE_TYPE_THOROUGHBRED
from racing_ai.storage import get_db


st.set_page_config(page_title="Racing AI", layout="wide", page_icon="🏇")


# --- session state --------------------------------------------------------
if "data_source" not in st.session_state:
    st.session_state.data_source = "synthetic"
if "n_sims" not in st.session_state:
    st.session_state.n_sims = DEFAULT_SIMULATIONS
if "races" not in st.session_state:
    st.session_state.races = None
if "manual_races" not in st.session_state:
    st.session_state.manual_races = []   # list[(Race, SimulationResult)]


@st.cache_resource(show_spinner="Booting pipeline (training ensemble on first run)...")
def get_pipeline(source: str) -> Pipeline:
    return Pipeline(source=source, auto_train=True)


@st.cache_resource
def get_manual_pipeline() -> Pipeline:
    """Pipeline whose source is overwritten on demand for manual races."""
    return Pipeline(source=ManualSource(), auto_train=True)


# --- sidebar --------------------------------------------------------------
with st.sidebar:
    st.title("🏇 Racing AI")
    st.caption("7-layer prediction stack + learning bot")

    st.session_state.data_source = st.selectbox(
        "Data source", ["synthetic", "manual", "sportsbet", "tab", "betfair"],
        index=["synthetic", "manual", "sportsbet", "tab", "betfair"].index(st.session_state.data_source),
        help="Use 'manual' with the Manual Race tab. SB/TAB/Betfair are stubs.",
    )
    st.session_state.n_sims = st.slider(
        "Monte Carlo simulations", 1000, 50000, st.session_state.n_sims, 1000,
    )

    if st.button("⟳ Fetch races", use_container_width=True, type="primary"):
        try:
            pipe = get_pipeline(st.session_state.data_source)
            st.session_state.races = pipe.run(n_simulations=st.session_state.n_sims)
            st.success(f"Loaded {len(st.session_state.races)} races")
        except NotImplementedError as e:
            st.error(str(e))

    if st.button("🧠 Retrain TB ensemble", use_container_width=True):
        get_pipeline.clear()
        ens, scores, n_rows = train_and_save(n_meetings=60)
        st.success(
            f"Trained on {n_rows} runner-rows. Train acc: "
            + ", ".join(f"{k}={v:.3f}" for k, v in scores.items())
        )

    st.divider()
    st.caption("Models loaded:")
    for k, v in MODELS_AVAILABLE.items():
        st.write(("✅" if v else "⚪") + f" {k}")

    bot = RaceLearningBot.get()
    summary = bot.summary()
    st.divider()
    st.caption(f"🤖 Bot: {summary['races_seen']} races learnt, "
               f"{summary['mature_buckets']} mature buckets")


# --- tabs -----------------------------------------------------------------
tab_card, tab_manual, tab_bot, tab_backtest, tab_perf, tab_settings = st.tabs([
    "Today's Card", "Manual Race", "Bot Insights",
    "Backtest", "Performance", "Settings",
])


# =========================================================================
# TAB 1 — Today's Card
# =========================================================================
def _runner_rows(race) -> list[dict]:
    rows = []
    for r in sorted(race.active_runners, key=lambda x: -x.win_prob):
        rows.append({
            "Box": r.box,
            "Runner": r.name,
            "Odds": r.win_odds,
            "Raw P(win)": round(r.raw_model_prob, 4),
            "Bot adj": round(r.bot_adjustment, 3),
            "Final P(win)": round(r.win_prob, 4),
            "P(place)": round(r.place_prob, 4),
            "EV %": round(r.ev_pct, 2),
            "Confidence": round(r.confidence, 3),
            "Volatility": round(r.features.get("volatility_score", 0), 3),
            "Kelly %": round(r.kelly_stake * 100, 2),
            "No-bet": "🚫" if r.no_bet else "✅",
            "Reason": r.no_bet_reason,
        })
    return rows


def _render_race(race, sim, key_prefix: str = ""):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Race type", race.race_type)
    c2.metric("Distance", f"{race.distance_m} m")
    c3.metric("Condition", race.track_condition)
    c4.metric("Pace pressure", f"{race.pace_pressure:.2f}")
    c5.metric("Track bias", f"{race.track_bias:.2f}")

    df = pd.DataFrame(_runner_rows(race))
    st.subheader("Runner-by-runner predictions")
    st.dataframe(df, use_container_width=True, hide_index=True)

    bet_candidates = df[df["No-bet"] == "✅"]
    if not bet_candidates.empty:
        st.success(f"{len(bet_candidates)} value bet(s) detected")
        with st.expander("Record a bet (optional)"):
            runner_choice = st.selectbox(
                "Runner", bet_candidates["Runner"].tolist(),
                key=f"{key_prefix}_runner",
            )
            stake = st.number_input(
                "Stake ($)", min_value=1.0, value=20.0, step=5.0,
                key=f"{key_prefix}_stake",
            )
            if st.button("Place bet", key=f"{key_prefix}_place"):
                runner = next(r for r in race.active_runners if r.name == runner_choice)
                bid = get_db().record_bet(
                    race.race_id, runner.runner_id, "win", stake, runner.win_odds,
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

    # Settle-result form for the bot
    with st.expander("Enter actual finishing order — feed the bot"):
        st.caption(
            "Comma-separated runner names in finishing order, e.g. `Neon Blue, Aston Verona, Phantom Carat`. "
            "Used by the RaceLearningBot to update calibration and feature weights."
        )
        order_str = st.text_input("Finishing order", key=f"{key_prefix}_order")
        if st.button("Submit result", key=f"{key_prefix}_settle"):
            names = [n.strip().lower() for n in order_str.split(",") if n.strip()]
            by_name = {r.name.lower(): r for r in race.active_runners}
            ids = [by_name[n].runner_id for n in names if n in by_name]
            if not ids:
                st.error("No matching runners — check the names.")
            else:
                pipe = get_manual_pipeline()
                insights = pipe.record_result(race, ids)
                st.success(f"Bot updated. {len(insights)} new insight(s) generated.")
                for line in insights:
                    st.write("• " + line)


with tab_card:
    races = st.session_state.races
    if not races:
        st.info("Hit **Fetch races** in the sidebar to load today's card.")
    else:
        labels = [f"R{r.race_number} · {r.track} · {r.distance_m}m · {r.race_type}"
                  for r, _ in races]
        idx = st.selectbox("Race", range(len(labels)), format_func=lambda i: labels[i])
        race, sim = races[idx]
        _render_race(race, sim, key_prefix=f"card_{idx}")


# =========================================================================
# TAB 2 — Manual Race
# =========================================================================
with tab_manual:
    st.subheader("Predict a race from real form data")
    mode = st.radio(
        "Input mode",
        ["Paste form text", "Structured table"],
        horizontal=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    track = c1.text_input("Track", value="Gawler")
    race_no = c2.number_input("Race #", min_value=1, max_value=14, value=5)
    distance = c3.number_input("Distance (m)", min_value=200, max_value=4000, value=400, step=20)
    race_type = c4.selectbox("Race type", [RACE_TYPE_GREYHOUND, RACE_TYPE_THOROUGHBRED])

    c5, c6 = st.columns(2)
    track_cond = c5.selectbox("Condition", ["fast", "good", "slow", "heavy", "firm", "soft"], index=1)
    weather = c6.selectbox("Weather", ["clear", "overcast", "rainy", "windy"])

    race = None

    if mode == "Paste form text":
        text = st.text_area(
            "Paste the race-card text (runner blocks)",
            height=320,
            help="Best-effort parser. Targets the 'N. Name (box) / F: form / T: trainer / "
                 "Career / Win % / Place % / Trk/Dist / Best Time / Weight / N days since' layout.",
        )
        if st.button("Parse + predict", type="primary"):
            try:
                race = parse_pasted_form(
                    text=text, track=track, race_number=int(race_no),
                    distance_m=int(distance), track_condition=track_cond,
                    race_type=race_type, weather=weather,
                )
                if not race.active_runners:
                    st.error("Parser found no runners. Check the formatting.")
                else:
                    st.success(f"Parsed {len(race.active_runners)} runners.")
            except Exception as e:
                st.exception(e)

    else:  # Structured table
        st.caption("Edit the table below — at minimum: box, name, win_odds. "
                   "More columns produce a sharper prediction.")
        default_runners = pd.DataFrame([
            {"box": 1, "name": "Runner 1", "win_odds": 3.50, "win_odds_open": 4.00,
             "place_odds": 1.90, "trainer": "", "last_6_form": "12345",
             "weight_kg": 30.0, "days_since_run": 7, "best_time_s": 23.10,
             "win_pct": 25.0, "place_pct": 60.0,
             "career_starts": 20, "career_wins": 5, "career_places": 7,
             "td_starts": 6, "td_wins": 1, "td_places": 3, "sb_rating": 95.0},
        ])
        edited = st.data_editor(default_runners, num_rows="dynamic", use_container_width=True)
        if st.button("Build + predict", type="primary"):
            runners = edited.to_dict(orient="records")
            try:
                if race_type == RACE_TYPE_GREYHOUND:
                    race = build_greyhound_race(
                        track=track, race_number=int(race_no),
                        distance_m=int(distance), track_condition=track_cond,
                        runners=runners, weather=weather,
                    )
                else:
                    race = build_thoroughbred_race(
                        track=track, race_number=int(race_no),
                        distance_m=int(distance), track_condition=track_cond,
                        runners=runners, weather=weather,
                    )
                st.success(f"Built race with {len(race.active_runners)} runners.")
            except Exception as e:
                st.exception(e)

    if race is not None and race.active_runners:
        pipe = get_manual_pipeline()
        race, sim = pipe.run_one(race, n_simulations=st.session_state.n_sims)
        st.session_state.manual_races = [(race, sim)]
        _render_race(race, sim, key_prefix=f"manual_{race.race_id}")
    elif st.session_state.manual_races:
        race, sim = st.session_state.manual_races[-1]
        st.info(f"Showing last manual race: {race.track} R{race.race_number}.")
        _render_race(race, sim, key_prefix=f"manual_{race.race_id}")


# =========================================================================
# TAB 3 — Bot Insights
# =========================================================================
with tab_bot:
    bot = RaceLearningBot.get()
    summary = bot.summary()

    c1, c2, c3 = st.columns(3)
    c1.metric("Races learned", summary["races_seen"])
    c2.metric("Calibration buckets", summary["context_buckets"])
    c3.metric("Mature buckets", summary["mature_buckets"])

    if summary["last_updated"]:
        st.caption(f"Last bot update (UTC): {summary['last_updated']}")
    else:
        st.caption("Bot has not learned any races yet — submit a finishing order to train it.")

    st.subheader("Per-race-type feature weights — current vs initial")
    weights_now = summary.get("weights_now", {})
    weights_drift = summary.get("weights_drift", {})
    if weights_now:
        for rt in weights_now:
            st.caption(f"**{rt.title()}**")
            drift_df = pd.DataFrame({
                "feature": list(weights_now[rt].keys()),
                "current": list(weights_now[rt].values()),
                "drift_from_initial": [weights_drift.get(rt, {}).get(f, 0.0)
                                       for f in weights_now[rt].keys()],
            })
            st.dataframe(drift_df, use_container_width=True, hide_index=True)
    else:
        st.info("No model state yet.")

    st.subheader("Top per-context calibrations")
    bucket_rows = bot.top_calibration_buckets(top_n=30)
    if bucket_rows:
        st.dataframe(pd.DataFrame(bucket_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No mature calibration buckets yet (each needs ~6 settled races).")

    st.subheader("Recent bot insights")
    insights = bot.recent_insights(limit=30)
    if insights:
        for ins in insights:
            sev_emoji = {"high": "🔥", "medium": "💡", "low": "📌"}.get(ins["severity"], "📌")
            st.write(f"{sev_emoji} **{ins['created_at'][:19]}** — {ins['text']}")
    else:
        st.caption("No insights yet — they appear after enough races settle in a bucket.")


# =========================================================================
# TAB 4 — Backtest
# =========================================================================
with tab_backtest:
    st.subheader("Train thoroughbred ensemble on synthetic data")
    st.caption("In production, replace generate_training_set() with a query against your historical DB.")
    n_meetings = st.slider("Synthetic meetings", 10, 200, 60, step=10)
    if st.button("Train now"):
        get_pipeline.clear()
        get_manual_pipeline.clear()
        ens, scores, n_rows = train_and_save(n_meetings=n_meetings)
        st.success(f"Trained on {n_rows} runner-rows.")
        st.json(scores)


# =========================================================================
# TAB 5 — Performance
# =========================================================================
with tab_perf:
    tracker = LearningTracker()
    snap = tracker.snapshot()
    if snap.n_bets == 0:
        st.info("No bets recorded yet. Place some on the Today's Card or Manual Race tab.")
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
# TAB 6 — Settings
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

    with st.expander("Architecture"):
        st.markdown("""
        | Layer | Module | Purpose |
        |---|---|---|
        | 1 | `layer1_intake` | DataSource adapters: synthetic, **manual** (paste/builder), SB/TAB/Betfair stubs |
        | 2 | `layer2_cleaning` | Drop scratched, normalise odds, validate runners |
        | 3 | `layer3_features` | 13 features, race-type aware (thoroughbred vs greyhound) |
        | 4 | `layer4_models` | Trained Ensemble (TB) **+ GreyhoundHeuristicModel (GH)** |
        | 5 | `layer5_simulation` | Plackett-Luce Monte Carlo (exacta/trifecta/first4) |
        | 6 | `layer6_value` | EV%, Kelly stake, no-bet filters |
        | 7 | `layer7_learning` | ROI/CLV tracker **+ RaceLearningBot (online SGD + per-context calibration)** |
        """)
