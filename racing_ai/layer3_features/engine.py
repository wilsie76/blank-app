"""Layer 3 — Feature engine.

Each feature is computed per-runner relative to its race context.
This is where the real edge gets built; treat the formulas below as starting
points and refine them with backtested signal-to-noise analysis.
"""
from __future__ import annotations
import math
import statistics
from typing import Iterable
import numpy as np
import pandas as pd
from ..schema import Race, Runner


FEATURE_NAMES = [
    "implied_prob",
    "market_rank",
    "form_score",
    "days_since_run",
    "weight_kg",
    "box_efficiency",
    "speed_differential",
    "fatigue_index",
    "pace_pressure",          # race-level repeated per runner
    "track_bias",             # race-level
    "steam_index",
    "closing_line_delta",
    "volatility_score",
]


def feature_names() -> list[str]:
    return list(FEATURE_NAMES)


# --- per-runner primitives ---------------------------------------------------

def _form_score(form: str) -> float:
    """Higher is better. Most-recent run weighted heaviest."""
    if not form:
        return 0.5
    digits = [int(c) for c in form if c.isdigit()]
    if not digits:
        return 0.5
    weights = [1.0, 0.7, 0.5, 0.3, 0.2][: len(digits)]
    raw = sum((10 - min(d, 9)) * w for d, w in zip(digits, weights))
    norm = raw / sum(weights) / 10.0
    return float(np.clip(norm, 0.0, 1.0))


def _box_efficiency(box: int, distance_m: int) -> float:
    """Sprints punish wide boxes; routes less so. 0..1 (1 = best)."""
    if distance_m <= 1200:
        return float(np.clip(1.0 - (box - 1) * 0.06, 0.2, 1.0))
    if distance_m <= 1600:
        return float(np.clip(1.0 - (box - 1) * 0.03, 0.5, 1.0))
    return float(np.clip(1.0 - (box - 1) * 0.015, 0.7, 1.0))


def _fatigue_index(days_since_run: int) -> float:
    """0 = ideally spaced, 1 = badly spaced. Sweet spot ~14-28 days."""
    if days_since_run < 7:
        return 0.8
    if days_since_run <= 14:
        return 0.3
    if days_since_run <= 28:
        return 0.0
    if days_since_run <= 60:
        return 0.4
    return 0.7


def _steam_index(open_odds: float, current_odds: float) -> float:
    """Positive = price has shortened (money came). Negative = drifted."""
    if open_odds <= 1.0 or current_odds <= 1.0:
        return 0.0
    return float(np.log(open_odds / current_odds))


# --- race-level features -----------------------------------------------------

def _pace_pressure(race: Race) -> float:
    """Proxy: more front-runner-shaped form profiles => more pressure."""
    n = len(race.active_runners)
    if n == 0:
        return 0.0
    front_share = sum(1 for r in race.active_runners if r.last_5_form.startswith(("1", "2", "3"))) / n
    short_distance_boost = max(0.0, (1600 - race.distance_m) / 1600) * 0.5
    return float(np.clip(front_share + short_distance_boost, 0.0, 1.0))


def _track_bias(race: Race) -> float:
    """Coarse proxy from track condition. Replace with rolling sectional analysis."""
    base = {"firm": 0.6, "good": 0.5, "soft": 0.35, "heavy": 0.2}.get(race.track_condition, 0.5)
    if race.weather == "rainy":
        base -= 0.05
    return float(np.clip(base, 0.0, 1.0))


# --- public API --------------------------------------------------------------

def build_features(race: Race) -> Race:
    """Mutate race.runners[*].features and race-level fields."""
    runners = race.active_runners
    if not runners:
        return race

    pace = _pace_pressure(race)
    bias = _track_bias(race)
    race.pace_pressure = pace
    race.track_bias = bias

    implied = np.array([1.0 / r.win_odds for r in runners])
    overround = implied.sum()
    implied_norm = implied / overround if overround > 0 else implied
    ranks = (-implied_norm).argsort().argsort() + 1   # 1 = favourite

    # Per-race speed proxy: inverse market rank scaled
    field_strength = float(implied_norm.mean())
    speeds = implied_norm / max(field_strength, 1e-6)

    # Volatility = std of last-5 form digits (race-level repeated)
    all_digits: list[int] = []
    for r in runners:
        all_digits += [int(c) for c in r.last_5_form if c.isdigit()]
    volatility = float(np.clip(statistics.pstdev(all_digits) / 4.5 if all_digits else 0.5, 0.0, 1.0))

    for i, r in enumerate(runners):
        steam = _steam_index(r.win_odds_open, r.win_odds)
        clv_delta = steam   # placeholder until closing odds are known
        r.features = {
            "implied_prob": float(implied_norm[i]),
            "market_rank": int(ranks[i]),
            "form_score": _form_score(r.last_5_form),
            "days_since_run": int(r.days_since_run),
            "weight_kg": float(r.weight_kg),
            "box_efficiency": _box_efficiency(r.box, race.distance_m),
            "speed_differential": float(speeds[i] - 1.0),
            "fatigue_index": _fatigue_index(r.days_since_run),
            "pace_pressure": pace,
            "track_bias": bias,
            "steam_index": steam,
            "closing_line_delta": clv_delta,
            "volatility_score": volatility,
        }
    return race


def race_to_dataframe(race: Race) -> pd.DataFrame:
    """Vectorise a race's runners into a model-ready frame."""
    rows = []
    for r in race.active_runners:
        row = {"runner_id": r.runner_id, **r.features}
        rows.append(row)
    return pd.DataFrame(rows).set_index("runner_id")[FEATURE_NAMES]


def races_to_dataframe(races: Iterable[Race]) -> pd.DataFrame:
    frames = [race_to_dataframe(r).assign(race_id=r.race_id) for r in races]
    return pd.concat(frames) if frames else pd.DataFrame(columns=FEATURE_NAMES)
