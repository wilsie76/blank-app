# Racing AI — 7-Layer Prediction Stack

A working skeleton for a value-driven horse racing prediction system. Every
layer below is a real, testable Python module — not pseudo-code.

## Architecture

| # | Module | Job |
|---|---|---|
| 1 | `racing_ai/layer1_intake/` | DataSource adapters. `synthetic` is fully working; `sportsbet` / `tab` / `betfair` are stubs you fill in with real clients. |
| 2 | `racing_ai/layer2_cleaning/` | Drop scratched/vacant runners, normalize odds, validate rows, discard malformed races. |
| 3 | `racing_ai/layer3_features/` | 13 features per runner: `implied_prob`, `market_rank`, `form_score`, `box_efficiency`, `speed_differential`, `fatigue_index`, `pace_pressure`, `track_bias`, `steam_index`, `closing_line_delta`, `volatility_score`, plus weight & days-since-run. |
| 4 | `racing_ai/layer4_models/` | XGBoost + LightGBM + CatBoost + MLP ensemble. Sklearn GBDT fallback if heavy libs missing. Per-race softmax normalisation guarantees probabilities sum to 1. |
| 5 | `racing_ai/layer5_simulation/` | Plackett-Luce Monte Carlo (10k sims default) → win/place/exacta/trifecta/first-4 probabilities. |
| 6 | `racing_ai/layer6_value/` | EV%, fractional Kelly stake, no-bet flags (EV, confidence, volatility, market-disagreement). |
| 7 | `racing_ai/layer7_learning/` | ROI / hit-rate / CLV / market-agreement / miss-type breakdown, backed by SQLite. |

`racing_ai/pipeline.py` is the orchestrator. `racing_ai/storage/db.py` is the
SQLite DAO (Postgres-swappable — change the connection only).

## Running

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

On first launch the ensemble auto-trains on synthetic data (~30 sec).

## Smoke test

```bash
python3 smoke_test.py
```

## What's NOT done (be honest)

- Real data adapters. Sportsbet/TAB/Betfair are stubs by design — each is a
  multi-week project on its own and may have ToS restrictions. Betfair is the
  cleanest path (`betfairlightweight`).
- Real historical training data. The pipeline trains on synthetic races so
  you can see it working end-to-end. Replace
  `layer4_models.train.generate_training_set` with a query against your
  historical DB.
- Probability calibration. Add Platt scaling or isotonic regression on a
  held-out set once you have real outcomes.
- Backtesting harness. Stub it out by replaying historical races through
  `Pipeline.run_one()` and feeding outcomes to the `LearningTracker`.

## Honest expectations

Strong, well-calibrated racing models long-term hit 5–8% ROI in good years
and break even in bad ones. The edge comes from disciplined EV filtering and
CLV tracking — not the model itself. The skeleton above gives you the
discipline machinery; the work is in the data and the calibration.
