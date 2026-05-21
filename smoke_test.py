"""Smoke test: train models, run full pipeline, print results."""
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from racing_ai.layer4_models import MODELS_AVAILABLE, train_and_save
from racing_ai.pipeline import Pipeline

print("Models detected:", MODELS_AVAILABLE)

print("\n=== Training ensemble ===")
ens, scores, n_rows = train_and_save(n_meetings=30)
print(f"Trained on {n_rows} rows. Train accuracy:", scores)

print("\n=== Running pipeline (synthetic source) ===")
pipe = Pipeline(source="synthetic", auto_train=False)
results = pipe.run(n_simulations=2000)
print(f"Pipeline produced {len(results)} races")

if results:
    race, sim = results[0]
    print(f"\n--- Race: {race.race_id} ({race.track} R{race.race_number}, {race.distance_m}m, {race.track_condition}) ---")
    print(f"Pace pressure: {race.pace_pressure:.2f} | Track bias: {race.track_bias:.2f}")
    print(f"\n{'Box':<4}{'Runner':<14}{'Odds':>7}{'P(win)':>9}{'EV%':>7}{'Conf':>6}  No-bet")
    for r in sorted(race.active_runners, key=lambda x: -x.win_prob):
        flag = "BET" if not r.no_bet else "no"
        print(f"{r.box:<4}{r.name:<14}{r.win_odds:>7.2f}{r.win_prob:>9.4f}{r.ev_pct:>7.1f}{r.confidence:>6.2f}  {flag}")

    print("\nTop 3 exacta combos:")
    for combo, p in sim.exacta[:3]:
        print(f"  {' -> '.join(combo)}  prob={p:.4f}")

    print("\nTop 3 trifecta combos:")
    for combo, p in sim.trifecta[:3]:
        print(f"  {' -> '.join(combo)}  prob={p:.4f}")

    # Check probabilities sum to 1
    total = sum(r.win_prob for r in race.active_runners)
    print(f"\nSum of P(win) for race: {total:.6f} (should be ~1.0)")

print("\n=== Testing learning tracker ===")
from racing_ai.storage import get_db
from racing_ai.layer7_learning import LearningTracker
db = get_db()

# Place a sample bet on the favourite + settle as winner
if results:
    race, _ = results[0]
    fav = max(race.active_runners, key=lambda x: x.win_prob)
    bid = db.record_bet(race.race_id, fav.runner_id, "win", 50.0, fav.win_odds)
    db.settle_bet(bid, won=True, odds_closing=fav.win_odds * 0.95)
    # Place a losing one too
    underdog = min(race.active_runners, key=lambda x: x.win_prob)
    bid2 = db.record_bet(race.race_id, underdog.runner_id, "win", 20.0, underdog.win_odds)
    db.settle_bet(bid2, won=False, odds_closing=underdog.win_odds * 1.10)

snap = LearningTracker().snapshot()
print(f"Snapshot: {snap}")
print("\n✅ End-to-end smoke test passed.")
