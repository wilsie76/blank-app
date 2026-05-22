# Racing AI — 7-Layer Prediction Stack + Learning Bot

A working racing prediction system with a self-learning bot. Greyhound *and*
thoroughbred. Every layer below is a real, testable Python module — not
pseudo-code.

## Architecture

| # | Module | Job |
|---|---|---|
| 1 | `racing_ai/layer1_intake/` | DataSource adapters. `synthetic` is fully working. **`ManualSource` + `parse_pasted_form()`** let you feed real form data with no scraper. SportsBet/TAB/Betfair are stubs. |
| 2 | `racing_ai/layer2_cleaning/` | Drop scratched/vacant runners, normalize odds, validate rows, discard malformed races. |
| 3 | `racing_ai/layer3_features/` | 13 features per runner, **race-type-aware** (greyhound vs thoroughbred): `implied_prob`, `market_rank`, `form_score`, `box_efficiency`, `speed_differential`, `fatigue_index`, `pace_pressure`, `track_bias`, `steam_index`, `closing_line_delta`, `volatility_score`, `weight_kg`, `days_since_run`. |
| 4 | `racing_ai/layer4_models/` | **`HeuristicModel`** (linear scorer with racing-wisdom priors, used by default — no training data required) **+** XGBoost / LightGBM / CatBoost / MLP `Ensemble` (opt-in via `use_ensemble_for_thoroughbreds=True` once you have real training data). Sklearn GBDT fallback when heavy libs are missing. Per-race softmax guarantees probabilities sum to 1. |
| 5 | `racing_ai/layer5_simulation/` | Plackett-Luce Monte Carlo (10k sims default) → win/place/exacta/trifecta/first-4 probabilities. |
| 6 | `racing_ai/layer6_value/` | EV%, fractional Kelly stake, no-bet flags (EV, confidence, volatility, market-disagreement). |
| 7 | `racing_ai/layer7_learning/` | ROI / hit-rate / CLV / miss-type tracker. **`RaceLearningBot`** — per-(track,distance,box) calibration table + online SGD on the heuristic weights + plain-English insight log. |

`racing_ai/pipeline.py` orchestrates everything. `racing_ai/storage/db.py` is
the SQLite DAO (Postgres-swappable — change the connection only).

## What "the bot" actually does

The `RaceLearningBot` sits between the model and the simulation. It holds:

1. **Per-race-type linear weights** (one `HeuristicModel` per type). After
   every settled race it does one cross-entropy SGD step against the actual
   winner: `update = -lr · Σᵢ(pᵢ - yᵢ) xᵢ`, clipped per-feature so a single
   race can't lurch the priors.
2. **Per-(race_type, track, distance, box) calibration buckets**. Every
   bucket tracks `(n_observed, sum_predicted, sum_observed)`. The applied
   multiplier is `(observed / predicted)`, smoothed toward 1.0 with a
   Bayesian prior so cold buckets don't explode.
3. **Insight log**. When a bucket has ≥ 6 observations and deviates more
   than ~22 % from the model's expectation, it publishes a finding such as
   *"Box 1 at Gawler 400m (greyhound) outperforms the model by +18 % over
   8 races"*.

The bot persists to `models/bot_state.pkl` and `bot_insights` in SQLite, so
it remembers what it learned across sessions.

## Running

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

In the UI:

* **Today's Card** — pick a race, see model probs + EV + sim combos +
  no-bet flags. Record bets and finishing orders.
* **Manual Race** — paste a race-card text dump *or* fill a structured
  table; the predictor runs immediately.
* **Bot Insights** — current heuristic weights, drift from initial,
  most-deviating per-context buckets, plain-English insights.
* **Backtest** — retrain the synthetic Ensemble.
* **Performance** — ROI / CLV / hit-rate / equity curve.

## Smoke tests

```bash
# End-to-end with synthetic source
python3 smoke_test.py

# Predict the 3 races from the user's seed dataset (Gawler R5 GH 400m,
# Pakenham R1 TB 1200m, Hatrick R9 GH 520m), then teach the bot the
# real outcomes and re-predict to show it learning.
python3 demo_user_races.py
```

## Programmatic use

```python
from racing_ai.layer1_intake import build_greyhound_race
from racing_ai.pipeline import Pipeline

race = build_greyhound_race(
    track="Gawler", race_number=5, distance_m=400, track_condition="good",
    runners=[
        dict(box=1, name="Aston Verona", trainer="Calicchio",
             last_6_form="542355", win_odds_open=4.40, win_odds=3.30,
             place_odds=1.75, weight_kg=27.2, days_since_run=4,
             win_pct=8.89, place_pct=46.67, best_time_s=23.03,
             td_starts=19, td_wins=0, td_places=7),
        # ... more runners
    ],
)
pipe = Pipeline(source="manual")
race, sim = pipe.run_one(race, n_simulations=8000)

# After the race runs, feed the actual finishing order back in:
pipe.record_result(race, finishing_order=[
    "<runner_id_of_winner>", "<runner_id_of_2nd>", ...
])
# The bot updates its weights + calibration tables and persists to disk.
```

## What's still NOT done (be honest)

- **Real data adapters.** SportsBet/TAB/Betfair are stubs by design — each
  is a multi-week project on its own and may have ToS restrictions.
  Betfair is the cleanest path (`betfairlightweight`).
- **Real historical training data.** The trained Ensemble fits on
  synthetic races so you can see it working end-to-end. That's why the
  default predictor is the bot-tuned `HeuristicModel`. Replace
  `layer4_models.train.generate_training_set` with a query against your
  historical DB before flipping `use_ensemble_for_thoroughbreds=True`.
- **Probability calibration.** Add Platt scaling or isotonic regression on
  a held-out set once you have real outcomes. The bot's per-context
  multipliers are a coarse calibration; they don't replace a global
  calibration step on a held-out validation set.
- **Backtesting harness.** Stub it out by replaying historical races
  through `Pipeline.run_one()` and feeding outcomes via
  `Pipeline.record_result()`.

## Honest expectations

Strong, well-calibrated racing models long-term hit 5-8 % ROI in good years
and break even in bad ones. The edge comes from disciplined EV filtering,
CLV tracking, and a model that's been continuously calibrated against the
specific tracks/distances you bet — not the model itself in isolation. The
skeleton above gives you the discipline machinery and the calibration
loop; the work is in the data and the time spent letting the bot
accumulate context.
