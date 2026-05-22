"""End-to-end demo on the three races the user supplied:

  1. Gawler R5  — 400m greyhound (good).  Won by #5 Neon Blue.
  2. Pakenham R1 — 1200m turf, Soft (6).  Won by #5 Le Notre.
  3. Hatrick R9 — 520m greyhound (good).  Won by #2 Boom Dynamite.

For each race we:
  - build the Race object from the form data the user pasted,
  - run it through the pipeline,
  - print the model's pre-result picks (with EV / Kelly / no-bet flags),
  - feed the actual finishing order to the bot,
  - print the insights generated.

This is run directly with `python demo_user_races.py`.
"""
from __future__ import annotations
import logging
from datetime import datetime
from textwrap import shorten

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from racing_ai.layer1_intake import build_greyhound_race, build_thoroughbred_race
from racing_ai.layer4_models import Ensemble, train_and_save
from racing_ai.layer7_learning import RaceLearningBot
from racing_ai.pipeline import Pipeline


# ============================================================================
# 1. Gawler R5 — greyhound, 400m, good
# ============================================================================
GAWLER_RUNNERS = [
    dict(box=1, name="Aston Verona", trainer="Marcello Calicchio",
         last_6_form="542355", win_odds_open=4.40, win_odds=3.30, place_odds=1.75,
         weight_kg=27.2, days_since_run=4,
         career_starts=90, career_wins=8, career_places=13 + 21,
         win_pct=8.89, place_pct=46.67,
         td_starts=19, td_wins=0, td_places=2 + 5,
         best_time_s=23.03),
    dict(box=2, name="Myah Miss", trainer="Anthony Nobes",
         last_6_form="14446X", win_odds_open=2.40, win_odds=3.30, place_odds=1.75,
         weight_kg=25.9, days_since_run=183,
         career_starts=21, career_wins=5, career_places=3 + 4,
         win_pct=23.81, place_pct=57.14,
         td_starts=6, td_wins=1, td_places=2 + 0,
         best_time_s=23.11, sb_rating=93),
    dict(box=4, name="Phantom Carat", trainer="Yvonne King",
         last_6_form="426332", win_odds_open=6.50, win_odds=7.00, place_odds=2.80,
         weight_kg=27.7, days_since_run=3,
         career_starts=75, career_wins=5, career_places=12 + 14,
         win_pct=6.67, place_pct=41.33,
         td_starts=19, td_wins=2, td_places=3 + 5,
         best_time_s=23.11, sb_rating=91),
    dict(box=5, name="Neon Blue", trainer="Anthony Nobes",
         last_6_form="523436", win_odds_open=3.90, win_odds=5.00, place_odds=2.25,
         weight_kg=25.2, days_since_run=7,
         career_starts=32, career_wins=5, career_places=5 + 8,
         win_pct=15.63, place_pct=56.25,
         td_starts=12, td_wins=1, td_places=2 + 4,
         best_time_s=23.09, sb_rating=100),
    dict(box=7, name="Wait And See", trainer="Donald Turner",
         last_6_form="255445", win_odds_open=10.00, win_odds=12.00, place_odds=3.90,
         weight_kg=29.4, days_since_run=3,
         career_starts=88, career_wins=6, career_places=10 + 19,
         win_pct=6.82, place_pct=39.77,
         td_starts=9, td_wins=3, td_places=1 + 1,
         best_time_s=23.06, sb_rating=94),
    dict(box=8, name="Aston Baker", trainer="Marcello Calicchio",
         last_6_form="354361", win_odds_open=6.00, win_odds=6.00, place_odds=2.50,
         weight_kg=31.6, days_since_run=3,
         career_starts=54, career_wins=7, career_places=8 + 9,
         win_pct=12.96, place_pct=44.44,
         td_starts=14, td_wins=0, td_places=4 + 2,
         best_time_s=23.17, sb_rating=100),
]
GAWLER_RESULT = ["Neon Blue", "Aston Verona", "Phantom Carat", "Myah Miss"]


# ============================================================================
# 2. Pakenham R1 — 1200m turf, soft (6)
#    Box = barrier number (the figure in parens on the card).
# ============================================================================
PAKENHAM_RUNNERS = [
    dict(box=9, name="Circus Circus", jockey="Billy Egan", trainer="G M Begg",
         weight_kg=58.0, last_5_form="44x5", win_odds_open=3.10, win_odds=2.35,
         place_odds=1.22, days_since_run=78,
         career_starts=3, career_wins=0, career_places=0,
         win_pct=0.0, place_pct=0.0, sb_rating=92),
    dict(box=1, name="Jenni Poppins", jockey="Beau Mertens", trainer="Lloyd Kennewell",
         weight_kg=58.0, last_5_form="4x2", win_odds_open=2.90, win_odds=2.90,
         place_odds=1.30, days_since_run=36,
         career_starts=2, career_wins=0, career_places=1,
         win_pct=0.0, place_pct=50.0, sb_rating=100),
    dict(box=8, name="Sparkling Award", jockey="Ben Allen", trainer="B Will & J Hayes",
         weight_kg=58.0, last_5_form="95x4", win_odds_open=8.00, win_odds=8.00,
         place_odds=1.95, days_since_run=18,
         career_starts=3, career_wins=0, career_places=0,
         win_pct=0.0, place_pct=0.0, sb_rating=85),
    dict(box=7, name="Le Notre", jockey="Jamie Mott", trainer="S B Laming",
         weight_kg=58.0, last_5_form="x3345", win_odds_open=7.50, win_odds=9.00,
         place_odds=2.10, days_since_run=17,
         career_starts=5, career_wins=0, career_places=2,
         win_pct=0.0, place_pct=40.0, sb_rating=92),
    dict(box=5, name="Forward Ho", jockey="Luke Nolen", trainer="J F Moloney",
         weight_kg=58.0, last_5_form="46x52", win_odds_open=8.50, win_odds=12.00,
         place_odds=2.45, days_since_run=18,
         career_starts=4, career_wins=0, career_places=1,
         win_pct=0.0, place_pct=25.0, sb_rating=83),
    dict(box=6, name="Waggish", jockey="Luke Currie", trainer="Clayton Douglas",
         weight_kg=58.0, last_5_form="6x", win_odds_open=6.00, win_odds=13.00,
         place_odds=2.60, days_since_run=141,
         career_starts=1, career_wins=0, career_places=0,
         win_pct=0.0, place_pct=0.0, sb_rating=86),
    dict(box=2, name="Karumba Girl", jockey="Jackie Beriman", trainer="Dale Short",
         weight_kg=58.0, last_5_form="850x6", win_odds_open=31.00, win_odds=41.00,
         place_odds=5.50, days_since_run=10,
         career_starts=4, career_wins=0, career_places=0,
         win_pct=0.0, place_pct=0.0, sb_rating=81),
    dict(box=3, name="Miss Gossip Girl", jockey="Samantha Noble", trainer="S G Noble",
         weight_kg=58.0, last_5_form="068x86", win_odds_open=81.00, win_odds=101.00,
         place_odds=12.00, days_since_run=15,
         career_starts=15, career_wins=0, career_places=3,
         win_pct=0.0, place_pct=20.0, sb_rating=80),
]
PAKENHAM_RESULT = ["Le Notre", "Jenni Poppins", "Circus Circus", "Waggish",
                   "Forward Ho", "Karumba Girl", "Miss Gossip Girl", "Sparkling Award"]


# ============================================================================
# 3. Hatrick R9 — 520m greyhound (good)
# ============================================================================
HATRICK_RUNNERS = [
    dict(box=1, name="Avatar Speed", trainer="Lisa J Cole",
         last_6_form="411163", win_odds_open=2.05, win_odds=2.00, place_odds=1.30,
         weight_kg=33.2, days_since_run=8,
         career_starts=20, career_wins=8, career_places=2 + 4,
         win_pct=40.0, place_pct=70.0,
         best_time_s=29.91, sb_rating=99),
    dict(box=2, name="Boom Dynamite", trainer="Lisa J Cole",
         last_6_form="451221", win_odds_open=3.90, win_odds=4.40, place_odds=1.75,
         weight_kg=33.0, days_since_run=10,
         career_starts=65, career_wins=23, career_places=16 + 11,
         win_pct=35.38, place_pct=76.92,
         best_time_s=29.56, sb_rating=99),
    dict(box=3, name="Need An Uber", trainer="Lisa J Cole",
         last_6_form="541144", win_odds_open=9.50, win_odds=9.00, place_odds=4.33,
         weight_kg=34.4, days_since_run=3,
         career_starts=41, career_wins=14, career_places=8 + 3,
         win_pct=34.15, place_pct=60.98,
         best_time_s=30.10, sb_rating=94),
    dict(box=5, name="Magic Jonty", trainer="Dylan J Voyce",
         last_6_form="235232", win_odds_open=8.50, win_odds=8.00, place_odds=3.00,
         weight_kg=32.9, days_since_run=8,
         career_starts=37, career_wins=8, career_places=11 + 8,
         win_pct=21.62, place_pct=72.97,
         best_time_s=30.03, sb_rating=85),
    dict(box=6, name="Super League", trainer="Lisa J Cole",
         last_6_form="412113", win_odds_open=4.80, win_odds=5.50, place_odds=1.91,
         weight_kg=36.2, days_since_run=7,
         career_starts=34, career_wins=14, career_places=6 + 5,
         win_pct=41.18, place_pct=73.53,
         best_time_s=29.96, sb_rating=100),
    dict(box=7, name="Pumbaa And Timon", trainer="Lisa J Cole",
         last_6_form="155532", win_odds_open=18.00, win_odds=23.00, place_odds=7.00,
         weight_kg=34.0, days_since_run=3,
         career_starts=42, career_wins=7, career_places=6 + 7,
         win_pct=16.67, place_pct=47.62,
         best_time_s=30.20, sb_rating=97),
    dict(box=8, name="More Airflow", trainer="Lisa J Cole",
         last_6_form="132213", win_odds_open=13.00, win_odds=14.00, place_odds=4.75,
         weight_kg=31.6, days_since_run=3,
         career_starts=22, career_wins=6, career_places=7 + 5,
         win_pct=27.27, place_pct=81.82,
         best_time_s=30.07, sb_rating=98),
]
HATRICK_RESULT = ["Boom Dynamite", "Avatar Speed", "Magic Jonty", "Need An Uber"]


# ============================================================================
# Pretty-printer
# ============================================================================
def print_predictions(race, sim, label: str, actual_winner: str | None = None):
    print(f"\n{'═' * 78}")
    print(f"  {label}")
    print(f"  Track: {race.track}  R{race.race_number}  "
          f"{race.distance_m}m  {race.race_type}  ({race.track_condition})")
    print(f"  Pace pressure: {race.pace_pressure:.2f}   Track bias: {race.track_bias:.2f}")
    print("═" * 78)
    print(f"{'Rank':<5}{'Box':<5}{'Runner':<22}{'Odds':>7}{'P(win)':>9}"
          f"{'P(plc)':>8}{'EV%':>8}{'Kelly%':>8}  Pick")
    print("─" * 78)
    sorted_runners = sorted(race.active_runners, key=lambda x: -x.win_prob)
    for i, r in enumerate(sorted_runners, 1):
        flag = "✅ BET" if not r.no_bet else "—"
        if actual_winner and r.name == actual_winner:
            flag = "🏆 " + ("WIN!" if not r.no_bet else "won (no-bet)")
        name = shorten(r.name, width=20, placeholder="…")
        print(f"{i:<5}{r.box:<5}{name:<22}{r.win_odds:>7.2f}"
              f"{r.win_prob:>9.4f}{r.place_prob:>8.4f}{r.ev_pct:>8.1f}"
              f"{r.kelly_stake*100:>7.2f}  {flag}")

    # Top combos
    print("\nTop exacta:   " + " | ".join(
        f"{'-'.join(c)} ({p:.3f})" for c, p in sim.exacta[:3]))
    print("Top trifecta: " + " | ".join(
        f"{'-'.join(c)} ({p:.3f})" for c, p in sim.trifecta[:3]))


def name_order_to_ids(race, names: list[str]) -> list[str]:
    by_name = {r.name.lower(): r.runner_id for r in race.active_runners}
    return [by_name[n.lower()] for n in names if n.lower() in by_name]


# ============================================================================
# Main
# ============================================================================
def main():
    # Make sure thoroughbred ensemble exists
    if not Ensemble.exists():
        print("Training thoroughbred ensemble (one-time, ~30s)…")
        train_and_save(n_meetings=60)

    pipe = Pipeline(source="manual", auto_train=False)
    bot = pipe.bot
    print(f"\nBot summary at start: {bot.summary()}")

    # ----- Build all three races -----
    gawler = build_greyhound_race(
        track="Gawler", race_number=5, distance_m=400, track_condition="good",
        runners=GAWLER_RUNNERS, weather="clear",
        start_time=datetime(2026, 5, 22, 17, 11),
    )
    pakenham = build_thoroughbred_race(
        track="Pakenham", race_number=1, distance_m=1200, track_condition="soft",
        runners=PAKENHAM_RUNNERS, weather="overcast",
        start_time=datetime(2026, 5, 22, 16, 45),
    )
    hatrick = build_greyhound_race(
        track="Hatrick", race_number=9, distance_m=520, track_condition="good",
        runners=HATRICK_RUNNERS, weather="clear",
        start_time=datetime(2026, 5, 22, 17, 12),
    )

    # ----- Run predictions, then teach the bot -----
    rounds: list[tuple] = []   # (label, race, sim, actual_winner_name, actual_order)

    for label, race, result_names in [
        ("RACE 1 — GAWLER R5 (greyhound 400m good)", gawler, GAWLER_RESULT),
        ("RACE 2 — PAKENHAM R1 (thoroughbred 1200m soft)", pakenham, PAKENHAM_RESULT),
        ("RACE 3 — HATRICK R9 (greyhound 520m good)", hatrick, HATRICK_RESULT),
    ]:
        race, sim = pipe.run_one(race, n_simulations=8000, persist=False)
        print_predictions(race, sim, label, actual_winner=result_names[0])
        rounds.append((label, race, sim, result_names[0], result_names))

    # ----- Feed actual results to the bot -----
    print("\n\n" + "═" * 78)
    print("  TEACHING THE BOT — feeding actual finishing orders")
    print("═" * 78)
    for label, race, _sim, _winner, names in rounds:
        ids = name_order_to_ids(race, names)
        insights = pipe.record_result(race, ids)
        print(f"\n• {race.track} R{race.race_number}: bot updated. "
              f"{len(insights)} new insight(s).")
        for line in insights:
            print(f"    💡 {line}")

    # ----- Show what the bot learned -----
    print("\n\n" + "═" * 78)
    print("  WHAT THE BOT LEARNED (greyhound feature-weight drift)")
    print("═" * 78)
    summary = bot.summary()
    print(f"  Races learned: {summary['races_seen']}")
    print(f"  Calibration buckets stored: {summary['context_buckets']}")
    print(f"  Mature buckets (≥6 obs):  {summary['mature_buckets']}")
    print(f"\n  Greyhound feature-weight drift after 2 greyhound races:")
    drift = summary["greyhound_weight_drift"]
    for k, v in drift.items():
        if abs(v) >= 0.001:
            arrow = "↑" if v > 0 else "↓"
            print(f"    {arrow} {k:<22}{v:+.4f}")

    # ----- Re-predict the same races to show the bot already nudged things -----
    print("\n\n" + "═" * 78)
    print("  RE-PREDICTING after one round of bot learning")
    print("═" * 78)
    for label, race, _sim, winner, _names in rounds:
        # Rebuild a fresh race so cached features aren't reused
        if "GAWLER" in label:
            race = build_greyhound_race(
                track="Gawler", race_number=5, distance_m=400, track_condition="good",
                runners=GAWLER_RUNNERS, weather="clear")
        elif "PAKENHAM" in label:
            race = build_thoroughbred_race(
                track="Pakenham", race_number=1, distance_m=1200, track_condition="soft",
                runners=PAKENHAM_RUNNERS, weather="overcast")
        else:
            race = build_greyhound_race(
                track="Hatrick", race_number=9, distance_m=520, track_condition="good",
                runners=HATRICK_RUNNERS, weather="clear")
        race, sim = pipe.run_one(race, n_simulations=8000, persist=False)
        print_predictions(race, sim, label + "  (after learning)", actual_winner=winner)


if __name__ == "__main__":
    main()
