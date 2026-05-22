"""Layer 3 — Feature engine.

Same 13 feature *names* across race types so downstream models don't care,
but each feature is computed in a race-type-aware way. Greyhound and
thoroughbred have very different drivers (box vs. weight, sectional time vs.
form-class), so the dispatch matters.
"""
from __future__ import annotations
import statistics
from typing import Iterable
import numpy as np
import pandas as pd
from ..schema import Race, Runner, RACE_TYPE_GREYHOUND


FEATURE_NAMES = [
    "implied_prob",
    "market_rank",
    "form_score",
    "days_since_run",
    "weight_kg",
    "box_efficiency",
    "speed_differential",
    "fatigue_index",
    "pace_pressure",          # race-level, repeated per runner
    "track_bias",             # race-level
    "steam_index",
    "closing_line_delta",
    "volatility_score",
]


def feature_names() -> list[str]:
    return list(FEATURE_NAMES)


# ============================================================================
# Shared primitives
# ============================================================================

def _form_digits(form: str) -> list[int]:
    """Extract digits from a form string. 'X' / 'x' (scratched/unraced) skipped."""
    return [int(c) for c in form if c.isdigit()]


def _form_score(form: str, recent_first: bool = True) -> float:
    """Higher = better. Recent runs weighted heavier.

    Form is read most-recent-first by convention (e.g. "542355" means last
    finish was 5, before that 4, etc.).
    """
    digits = _form_digits(form)
    if not digits:
        return 0.5
    weights = [1.0, 0.7, 0.5, 0.35, 0.25, 0.18][: len(digits)]
    raw = sum((10 - min(d, 9)) * w for d, w in zip(digits, weights))
    norm = raw / sum(weights) / 10.0
    return float(np.clip(norm, 0.0, 1.0))


def _steam_index(open_odds: float, current_odds: float) -> float:
    """Positive = price has shortened (money came). Negative = drifted."""
    if open_odds <= 1.0 or current_odds <= 1.0:
        return 0.0
    return float(np.log(open_odds / current_odds))


# ============================================================================
# Thoroughbred-specific primitives
# ============================================================================

def _tb_box_efficiency(box: int, distance_m: int) -> float:
    """Sprints punish wide barriers; routes less so. 0..1 (1 = best)."""
    if distance_m <= 1200:
        return float(np.clip(1.0 - (box - 1) * 0.06, 0.2, 1.0))
    if distance_m <= 1600:
        return float(np.clip(1.0 - (box - 1) * 0.03, 0.5, 1.0))
    return float(np.clip(1.0 - (box - 1) * 0.015, 0.7, 1.0))


def _tb_fatigue_index(days_since_run: int) -> float:
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


def _tb_pace_pressure(race: Race) -> float:
    """Proxy: more leader-shaped form profiles => more pressure."""
    n = len(race.active_runners)
    if n == 0:
        return 0.0
    front_share = sum(
        1 for r in race.active_runners
        if (r.last_5_form or "").lstrip().startswith(("1", "2", "3"))
    ) / n
    short_distance_boost = max(0.0, (1600 - race.distance_m) / 1600) * 0.5
    return float(np.clip(front_share + short_distance_boost, 0.0, 1.0))


def _tb_track_bias(race: Race) -> float:
    base = {"firm": 0.6, "good": 0.5, "soft": 0.35, "heavy": 0.2}.get(race.track_condition, 0.5)
    if race.weather == "rainy":
        base -= 0.05
    return float(np.clip(base, 0.0, 1.0))


# ============================================================================
# Greyhound-specific primitives
# ============================================================================

def _gh_box_efficiency(box: int, distance_m: int, n_runners: int) -> float:
    """In greyhound racing the inside boxes (1-3) are a real edge, especially
    at sprints. Effect attenuates at longer distances and where there are
    vacant boxes outside us.
    """
    # Base table (520m / 400m sprint):
    #   box 1 -> ~0.95, box 5 -> ~0.6, box 8 -> ~0.45
    base = max(0.30, 1.0 - (box - 1) * 0.07)
    if distance_m >= 600:
        base = max(0.45, 1.0 - (box - 1) * 0.045)   # less impact in stayers
    if distance_m <= 400:
        base = max(0.30, 1.0 - (box - 1) * 0.085)   # MORE impact in 300-400m
    # Field-size compensation: a wide box with a small field plays smaller
    if n_runners <= 6:
        base = base + 0.05
    return float(np.clip(base, 0.20, 1.0))


def _gh_fatigue_index(days_since_run: int) -> float:
    """Greyhounds: best window is 5-14 days. >30 days = freshness/spell risk."""
    if days_since_run < 4:
        return 0.7
    if days_since_run <= 10:
        return 0.0
    if days_since_run <= 21:
        return 0.15
    if days_since_run <= 45:
        return 0.4
    if days_since_run <= 90:
        return 0.65
    return 0.85   # long layoff


def _gh_pace_pressure(race: Race) -> float:
    """Greyhound 'pace' is dominated by early speed off the boxes.
    Multiple front-running types in low boxes -> higher pressure.
    """
    if not race.active_runners:
        return 0.0
    early_speed_count = 0
    for r in race.active_runners:
        digits = _form_digits(r.last_6_form or r.last_5_form)
        if digits and digits[0] in (1, 2, 3) and r.box <= 4:
            early_speed_count += 1
    return float(np.clip(early_speed_count / 3.0, 0.0, 1.0))


def _gh_track_bias(race: Race) -> float:
    base = {"fast": 0.7, "good": 0.55, "slow": 0.4, "heavy": 0.25}.get(
        (race.track_condition or "").lower(), 0.55
    )
    return float(np.clip(base, 0.0, 1.0))


def _gh_speed_differential(runner: Runner, field_best: float | None) -> float:
    """How much faster than the field-median best-time is this dog?
    Returns ~ -0.5..+0.5 (normalised seconds advantage).
    """
    if not runner.best_time_s or not field_best:
        return 0.0
    delta = field_best - runner.best_time_s     # positive means faster than median
    return float(np.clip(delta / 0.5, -0.5, 0.5))


# ============================================================================
# Public API — race-type dispatch
# ============================================================================

def build_features(race: Race) -> Race:
    """Mutate race.runners[*].features and race-level fields.

    Dispatches on race.race_type. Always emits the same 13-key feature dict
    so downstream models don't have to branch.
    """
    runners = race.active_runners
    if not runners:
        return race

    if race.race_type == RACE_TYPE_GREYHOUND:
        return _build_features_greyhound(race)
    return _build_features_thoroughbred(race)


def _build_features_thoroughbred(race: Race) -> Race:
    runners = race.active_runners
    pace = _tb_pace_pressure(race)
    bias = _tb_track_bias(race)
    race.pace_pressure = pace
    race.track_bias = bias

    implied = np.array([1.0 / r.win_odds for r in runners])
    overround = implied.sum()
    implied_norm = implied / overround if overround > 0 else implied
    ranks = (-implied_norm).argsort().argsort() + 1     # 1 = favourite

    field_strength = float(implied_norm.mean())
    speeds = implied_norm / max(field_strength, 1e-6)

    all_digits: list[int] = []
    for r in runners:
        all_digits += _form_digits(r.last_5_form)
    volatility = float(np.clip(
        statistics.pstdev(all_digits) / 4.5 if all_digits else 0.5, 0.0, 1.0,
    ))

    for i, r in enumerate(runners):
        steam = _steam_index(r.win_odds_open, r.win_odds)
        r.features = {
            "implied_prob": float(implied_norm[i]),
            "market_rank": int(ranks[i]),
            "form_score": _form_score(r.last_5_form),
            "days_since_run": int(r.days_since_run),
            "weight_kg": float(r.weight_kg),
            "box_efficiency": _tb_box_efficiency(r.box, race.distance_m),
            "speed_differential": float(speeds[i] - 1.0),
            "fatigue_index": _tb_fatigue_index(r.days_since_run),
            "pace_pressure": pace,
            "track_bias": bias,
            "steam_index": steam,
            "closing_line_delta": steam,
            "volatility_score": volatility,
        }
    return race


def _build_features_greyhound(race: Race) -> Race:
    runners = race.active_runners
    n = len(runners)
    pace = _gh_pace_pressure(race)
    bias = _gh_track_bias(race)
    race.pace_pressure = pace
    race.track_bias = bias

    implied = np.array([1.0 / r.win_odds for r in runners])
    overround = implied.sum()
    implied_norm = implied / overround if overround > 0 else implied
    ranks = (-implied_norm).argsort().argsort() + 1

    # Field-best benchmark for time advantage
    times = [r.best_time_s for r in runners if r.best_time_s]
    field_best = float(np.median(times)) if times else None

    # Volatility: spread of last-6 form digits across the field
    all_digits: list[int] = []
    for r in runners:
        all_digits += _form_digits(r.last_6_form or r.last_5_form)
    volatility = float(np.clip(
        statistics.pstdev(all_digits) / 4.5 if all_digits else 0.5, 0.0, 1.0,
    ))

    for i, r in enumerate(runners):
        steam = _steam_index(r.win_odds_open, r.win_odds)

        # Form score: prefer the deeper last-6 if present, else last_5
        form_str = r.last_6_form or r.last_5_form
        form = _form_score(form_str)

        # Boost form_score by win% / place% if available (data-driven prior)
        if r.win_pct is not None and r.place_pct is not None:
            history = (r.win_pct * 0.6 + r.place_pct * 0.4) / 100.0   # 0..1
            form = float(np.clip(0.6 * form + 0.4 * history, 0.0, 1.0))

        # Track-and-distance specialism — places/start at this t/d
        td_strength = 0.0
        if r.td_starts and r.td_starts >= 3:
            td_places = (r.td_wins or 0) + (r.td_places or 0)
            td_strength = td_places / r.td_starts        # 0..1

        # Speed differential vs field-best time at this distance
        speed_diff = _gh_speed_differential(r, field_best)

        r.features = {
            "implied_prob": float(implied_norm[i]),
            "market_rank": int(ranks[i]),
            "form_score": form,
            "days_since_run": int(r.days_since_run),
            "weight_kg": float(r.weight_kg),
            "box_efficiency": _gh_box_efficiency(r.box, race.distance_m, n),
            "speed_differential": float(speed_diff + 0.5 * td_strength),
            "fatigue_index": _gh_fatigue_index(r.days_since_run),
            "pace_pressure": pace,
            "track_bias": bias,
            "steam_index": steam,
            "closing_line_delta": steam,
            "volatility_score": volatility,
        }
    return race


def race_to_dataframe(race: Race) -> pd.DataFrame:
    """Vectorise a race's runners into a model-ready frame."""
    rows = []
    for r in race.active_runners:
        if not r.features:
            # Defensive: build features lazily if caller forgot
            build_features(race)
            break
    for r in race.active_runners:
        rows.append({"runner_id": r.runner_id, **r.features})
    return pd.DataFrame(rows).set_index("runner_id")[FEATURE_NAMES]


def races_to_dataframe(races: Iterable[Race]) -> pd.DataFrame:
    frames = [race_to_dataframe(r).assign(race_id=r.race_id) for r in races]
    return pd.concat(frames) if frames else pd.DataFrame(columns=FEATURE_NAMES)
