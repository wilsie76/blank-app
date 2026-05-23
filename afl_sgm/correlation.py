"""Leg-to-leg correlation engine.

Three levels of dependence between legs:

1. Fully redundant (rho = 1.0). Same player, same market, but the higher
   threshold mathematically implies the lower one. Example:
       Heeney 30+ disposals  ->  Heeney 20+ disposals (always lands too).
   These pairs add zero lift to combined probability and the bookie prices
   them as if only the higher-threshold leg existed. We collapse them.

2. Same player, related markets. Disposals + fantasy points + kicks all
   move together because they share an underlying "good game for the
   player" signal. We use pairwise correlations from RELATED_MARKETS in
   schema.py.

3. Cross-player or team-thesis. A small positive lift (~0.10) applies when
   legs trend in the same team direction (e.g. several Sydney players
   exceeding form-line implies Sydney is on top, which the H2H leg
   already implies). Slight negative for opposing-team props.

We do NOT try to reproduce Sportsbet's exact correlation discount because
that's proprietary. We do produce a model-fair price and a model-implied
combined probability the user can compare against the bookie's actual SGM
quote to spot edges.
"""
from __future__ import annotations
from typing import Iterable

from .schema import Leg, RELATED_MARKETS, MARKET_H2H


# Default within-group correlation between RELATED markets on the same player.
# Used when no entry exists in RELATED_MARKETS.
SAME_PLAYER_DEFAULT_RHO = 0.40

# Cross-player correlation defaults.
SAME_TEAM_RHO  =  0.10  # mild positive: e.g. two Geelong mids both having a day
OPP_TEAM_RHO   = -0.05  # very mild negative: opposing team success conflicts
NEUTRAL_RHO    =  0.0

# H2H correlation with team-friendly props.
H2H_PLAYER_RHO = 0.30   # Sydney H2H vs a Sydney player going big


def _redundancy_key(leg: Leg) -> tuple | None:
    """Two legs share this key iff one's threshold ladder implies the other.

    Same player + same canonical market => same ladder. Threshold itself is
    NOT part of the key (so 20+ and 25+ disposals share the key).
    """
    if leg.is_player_leg() and leg.player and leg.market and leg.threshold is not None:
        return ("ladder", leg.player.lower(), leg.market)
    return None


def redundant_pairs(legs: Iterable[Leg]) -> list[tuple[str, str]]:
    """Return pairs of leg_ids where one implies the other.

    The first id in each tuple is the IMPLIED leg (lower threshold, redundant
    when paired with the higher threshold of the same ladder).
    """
    by_ladder: dict[tuple, list[Leg]] = {}
    for leg in legs:
        key = _redundancy_key(leg)
        if key:
            by_ladder.setdefault(key, []).append(leg)
    pairs: list[tuple[str, str]] = []
    for ladder_legs in by_ladder.values():
        if len(ladder_legs) < 2:
            continue
        # sort ascending by threshold; lower implied by higher
        ladder_legs.sort(key=lambda l: l.threshold or 0.0)
        for i in range(len(ladder_legs) - 1):
            for j in range(i + 1, len(ladder_legs)):
                pairs.append((ladder_legs[i].leg_id, ladder_legs[j].leg_id))
    return pairs


def collapse_redundant(legs: list[Leg]) -> list[Leg]:
    """Within each (player, market) ladder keep only the highest threshold.

    The lower-threshold leg has higher probability but is implied by the
    higher one when both are picked, so for combined-probability purposes
    we use the higher threshold (which is the binding constraint).
    """
    by_ladder: dict[tuple, list[Leg]] = {}
    other: list[Leg] = []
    for leg in legs:
        key = _redundancy_key(leg)
        if key:
            by_ladder.setdefault(key, []).append(leg)
        else:
            other.append(leg)
    out = list(other)
    for ladder_legs in by_ladder.values():
        ladder_legs.sort(key=lambda l: l.threshold or 0.0, reverse=True)
        out.append(ladder_legs[0])
    return out


def pair_correlation(a: Leg, b: Leg) -> float:
    """Heuristic correlation in [-1, 1] between two legs."""
    if a.leg_id == b.leg_id:
        return 1.0

    # Ladder redundancy on same player + same market
    if (a.is_player_leg() and b.is_player_leg()
        and a.player and b.player and a.player.lower() == b.player.lower()
        and a.market == b.market):
        return 1.0

    # Same player, different but related markets
    if (a.is_player_leg() and b.is_player_leg()
        and a.player and b.player and a.player.lower() == b.player.lower()):
        rel = RELATED_MARKETS.get(a.market or "", {}).get(b.market or "")
        if rel is not None:
            return rel
        return SAME_PLAYER_DEFAULT_RHO

    # H2H combined with player from same team
    if a.is_h2h() and b.is_player_leg():
        # we don't carry team-of-player in the schema; use leg description
        # heuristic: if the team name appears in description? defensive default.
        return H2H_PLAYER_RHO * 0.6  # uncertain attribution
    if b.is_h2h() and a.is_player_leg():
        return H2H_PLAYER_RHO * 0.6

    # Both H2H (e.g. Geelong + Sydney): mutually exclusive
    if a.is_h2h() and b.is_h2h() and (a.team or "").lower() != (b.team or "").lower():
        return -1.0

    # Default: near-independent
    return NEUTRAL_RHO


def correlation_matrix(legs: list[Leg]) -> list[list[float]]:
    """Symmetric correlation matrix in the order legs are passed."""
    n = len(legs)
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        M[i][i] = 1.0
        for j in range(i + 1, n):
            rho = pair_correlation(legs[i], legs[j])
            M[i][j] = M[j][i] = rho
    return M


def avg_pair_correlation(legs: list[Leg]) -> float:
    """Average pairwise correlation across all unique pairs (for warnings)."""
    if len(legs) < 2:
        return 0.0
    s = 0.0
    n = 0
    for i in range(len(legs)):
        for j in range(i + 1, len(legs)):
            s += pair_correlation(legs[i], legs[j])
            n += 1
    return s / n if n else 0.0
