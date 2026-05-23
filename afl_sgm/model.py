"""Per-leg probability model.

For numeric-threshold legs (e.g. "25+ Disposals"):
    Gaussian on last-5 with half-integer continuity correction, blended with
    the de-vigged market-implied probability via a mild Beta-Binomial-style
    prior. This uses the *magnitude* of historical stats, not just hits.

For binary legs (H2H, "1+ Goal"-style):
    Beta-Binomial on wins/hits with a uniform-ish prior.

We deliberately produce a *conservative* probability when the last-5 sample
is small or noisy. For the SGM use case ("best winning outcome") that's a
feature: it biases the optimiser toward props that visibly cleared the
threshold by a margin in recent form.
"""
from __future__ import annotations
import math
import statistics
from typing import Iterable

from .schema import Leg, PLAYER_STAT_MARKETS, MARKET_H2H


# Probability bounds. n=5 is too small to ever justify >97% from form alone.
PROB_MIN = 0.05
PROB_MAX = 0.97

# Bookmaker margin per leg for de-vigging (used to derive a prior). The real
# margin varies (usually 3-6% depending on market depth) but 4% is a
# reasonable default.
DEFAULT_VIG = 0.04

# How many "prior games" of weight to give the market-implied probability.
# n_observed=5 vs prior=4 => 56% form weight, 44% market weight when n=5.
DEFAULT_PRIOR_STRENGTH = 4.0

# Floor on stdev to avoid divide-by-zero / overconfident projections when
# the last-5 happens to be flat (e.g. McCartin marks: 7,7,12,8,7).
SIGMA_FLOOR = 0.5


def _normal_cdf(z: float) -> float:
    """Standard normal CDF. Stdlib only -- no numpy required."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _clip(x: float, lo: float = PROB_MIN, hi: float = PROB_MAX) -> float:
    return max(lo, min(hi, x))


def _market_implied(price: float, vig: float = DEFAULT_VIG) -> float:
    """De-vigged market probability."""
    if price <= 1.0:
        return 0.99
    raw = 1.0 / price
    # Per-leg vig adjustment is approximate; assumes 1+vig two-way overround.
    return _clip(raw / (1.0 + vig), 0.01, 0.99)


def _ev_pct(p: float, price: float) -> float:
    return (p * price - 1.0) * 100.0


def _score_h2h(leg: Leg) -> None:
    """Team head-to-head. Last-5 form is already priced in by the market;
    we use a lightly-smoothed market-implied probability as the model prob,
    then read confidence off the form (lopsided W/L => more confident).
    """
    history = [h for h in leg.history if h in ("W", "L", "D")]
    n = len(history)
    hits = sum(1 for h in history if h == "W")
    leg.hits = hits
    leg.n_games = n
    leg.raw_hit_rate = (hits / n) if n else 0.0

    market_p = _market_implied(leg.price)
    # Mild Beta(2, 2) blend with form -- but only as a tie-breaker.
    if n:
        beta_p = (2.0 + hits) / (2.0 + 2.0 + n)
        leg.model_prob = _clip(0.7 * market_p + 0.3 * beta_p)
    else:
        leg.model_prob = _clip(market_p)

    leg.confidence = abs(2.0 * leg.raw_hit_rate - 1.0) if n else 0.3
    leg.ev_pct = _ev_pct(leg.model_prob, leg.price)
    if n < 5:
        leg.flags.append("partial form sample")


def _score_player_threshold(leg: Leg, prior_strength: float, vig: float) -> None:
    """Player numeric prop (disposals, fantasy, marks, etc.).

    Uses a Gaussian over last-5 raw values with half-integer continuity
    correction, blended with de-vigged market-implied probability.
    """
    if not leg.history or leg.threshold is None:
        leg.model_prob = _market_implied(leg.price, vig)
        leg.confidence = 0.2
        leg.ev_pct = _ev_pct(leg.model_prob, leg.price)
        leg.flags.append("no history")
        return

    nums = [float(h) for h in leg.history]
    n = len(nums)
    threshold = float(leg.threshold)
    hits = sum(1 for v in nums if v >= threshold)
    leg.hits = hits
    leg.n_games = n
    leg.raw_hit_rate = hits / n

    if n >= 2:
        mu = statistics.fmean(nums)
        sigma = max(statistics.stdev(nums), SIGMA_FLOOR)
    else:
        mu = nums[0]
        sigma = max(mu * 0.25, SIGMA_FLOOR)

    # Half-integer continuity correction: P(X >= k) ~ P(Y > k - 0.5).
    adj = threshold - 0.5
    z = (adj - mu) / sigma
    p_form = 1.0 - _normal_cdf(z)
    p_form = _clip(p_form, 0.02, 0.98)

    market_p = _market_implied(leg.price, vig)
    blended = (prior_strength * market_p + n * p_form) / (prior_strength + n)
    leg.model_prob = _clip(blended)

    # Confidence:
    #   - high if z-distance from threshold is large in the favourable direction
    #   - high if last-5 is consistent (low coef of variation)
    cv = sigma / max(abs(mu), 1.0)
    consistency = max(0.0, 1.0 - cv)
    z_strength = min(1.0, abs(z) / 1.5)         # |z| >= 1.5 saturates
    leg.confidence = round(max(0.15, 0.5 * consistency + 0.5 * z_strength), 3)

    if n < 5:
        leg.flags.append("partial form sample")
    if hits == 0:
        leg.flags.append("0 hits in last 5")
    elif hits == n and threshold <= mu * 0.5:
        leg.flags.append("threshold trivially below mean")
    if cv > 0.6:
        leg.flags.append("volatile form")

    leg.ev_pct = _ev_pct(leg.model_prob, leg.price)


def score_leg(
    leg: Leg,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    vig: float = DEFAULT_VIG,
) -> Leg:
    """Mutate a single leg with model_prob / confidence / ev_pct in place."""
    leg.flags = []
    if leg.is_h2h() or leg.market == MARKET_H2H:
        _score_h2h(leg)
    elif leg.market in PLAYER_STAT_MARKETS:
        _score_player_threshold(leg, prior_strength, vig)
    else:
        # Unknown market -- fall back to market-implied
        leg.model_prob = _market_implied(leg.price, vig)
        leg.confidence = 0.2
        leg.ev_pct = _ev_pct(leg.model_prob, leg.price)
        leg.flags.append("unknown market")
    return leg


def score_legs(
    legs: Iterable[Leg],
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    vig: float = DEFAULT_VIG,
) -> list[Leg]:
    """Score every leg. Returns the same list (mutated)."""
    out = list(legs)
    for leg in out:
        score_leg(leg, prior_strength=prior_strength, vig=vig)
    return out
