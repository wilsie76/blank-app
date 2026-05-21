"""Layer 6 — Value engine.

For each runner:
  ev_pct       = (model_prob * (decimal_odds - 1) - (1 - model_prob)) * 100
  kelly_stake  = max(0, fractional_kelly * (b*p - q) / b)   where b=odds-1, q=1-p
  no_bet flag  triggered when:
                - EV below MIN_EV_PCT
                - confidence below MIN_CONFIDENCE
                - volatility above MAX_VOLATILITY
                - model & market disagree wildly (sanity check)
"""
from __future__ import annotations
import math
import numpy as np
from ..config import (
    MIN_EV_PCT, MAX_VOLATILITY, MIN_CONFIDENCE, KELLY_FRACTION,
)
from ..schema import Race


def kelly_fraction(p: float, decimal_odds: float, fraction: float = KELLY_FRACTION) -> float:
    b = max(decimal_odds - 1.0, 1e-6)
    q = 1.0 - p
    full = (b * p - q) / b
    return float(max(0.0, full * fraction))


def _ev_pct(p: float, decimal_odds: float) -> float:
    return (p * (decimal_odds - 1.0) - (1.0 - p)) * 100.0


def evaluate_race(race: Race, confidence: float) -> Race:
    """Mutates race.runners with ev_pct, no_bet, kelly_stake, confidence."""
    runners = race.active_runners
    if not runners:
        return race

    # Implied (vig-removed) market prob for sanity check
    implied = np.array([1.0 / r.win_odds for r in runners])
    overround = implied.sum()
    fair_market = implied / overround if overround > 0 else implied

    for i, r in enumerate(runners):
        ev = _ev_pct(r.win_prob, r.win_odds)
        r.ev_pct = ev
        r.confidence = confidence
        vol = float(r.features.get("volatility_score", 0.5))
        market_p = float(fair_market[i])
        # If model says <50% of market price, treat as suspect
        wild_disagree = (market_p > 0.15 and r.win_prob < market_p * 0.4)

        reasons = []
        if ev < MIN_EV_PCT:
            reasons.append(f"EV {ev:.1f}% < {MIN_EV_PCT}%")
        if confidence < MIN_CONFIDENCE:
            reasons.append(f"low ensemble confidence {confidence:.2f}")
        if vol > MAX_VOLATILITY:
            reasons.append(f"volatility {vol:.2f} > {MAX_VOLATILITY}")
        if wild_disagree:
            reasons.append("model disagrees sharply with market")

        r.no_bet = bool(reasons)
        r.no_bet_reason = "; ".join(reasons)
        r.kelly_stake = 0.0 if r.no_bet else kelly_fraction(r.win_prob, r.win_odds)

    return race
