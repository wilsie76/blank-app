"""Combine many legs into a single SGM ticket.

Given an ordered set of legs with model probabilities and a correlation
function, return:

  * naive_price       -- product of leg prices (fictional; the bookie won't
                         offer this)
  * combined_prob     -- model's view of P(all legs land), under correlation
  * fair_price        -- 1 / combined_prob (zero-edge price implied by model)
  * estimated_book_price -- naive_price collapsed for redundant ladders +
                            heuristic correlation discount on related-market
                            pairs in the same player. This is a *rough*
                            approximation of what Sportsbet would offer.

Combination math
----------------

For two events A, B with marginals p_a, p_b and correlation rho, we use the
Frechet-Hoeffding bound interpolation:

    P(A and B | rho) =
        (1 - rho) * p_a * p_b  +  rho * min(p_a, p_b)             if rho >= 0
        (1 + rho) * p_a * p_b  +  (-rho) * max(0, p_a + p_b - 1)  if rho < 0

This collapses to:
    rho =  0   ->  independence:  p_a * p_b
    rho =  1   ->  perfect:       min(p_a, p_b)        [B implied by A]
    rho = -1   ->  perfect-neg:   max(0, p_a + p_b - 1) [mutually exclusive]

For >2 legs we apply this pairwise and chain. We process redundant ladders
first (rho=1), then same-player related markets, then cross-player. This
matches a Gaussian-copula upper-tail approximation closely enough for SGM
ranking purposes; we are NOT trying to reproduce Sportsbet's exact pricing
because their correlation matrix is proprietary.
"""
from __future__ import annotations
from typing import Optional

from .schema import Leg, SGMTicket
from .correlation import (
    pair_correlation,
    collapse_redundant,
    avg_pair_correlation,
)


def _frechet_combine(p_a: float, p_b: float, rho: float) -> float:
    """Pairwise probability under Frechet-Hoeffding linear interpolation."""
    p_a = max(0.0, min(1.0, p_a))
    p_b = max(0.0, min(1.0, p_b))
    rho = max(-1.0, min(1.0, rho))
    # Mutually exclusive (e.g. opposing-team H2H legs) is a hard zero.
    if rho <= -0.999:
        return 0.0
    indep = p_a * p_b
    if rho >= 0:
        upper = min(p_a, p_b)
        return (1.0 - rho) * indep + rho * upper
    else:
        lower = max(0.0, p_a + p_b - 1.0)
        return (1.0 + rho) * indep + (-rho) * lower


def _greedy_combine_probs(legs: list[Leg]) -> float:
    """Compose joint P(all land) by chaining pairwise.

    Order of composition matters slightly; we process highest-correlation
    pairs first (so redundancy is absorbed before independent multiplication
    inflates uncertainty).
    """
    if not legs:
        return 1.0
    if len(legs) == 1:
        return legs[0].model_prob

    # Build a working list of (combined_so_far, member_legs_proxy)
    # We treat the chain's running probability as a single pseudo-leg. The
    # correlation between the running combo and the next leg is the MAX of
    # the next leg's correlations against any leg already in the combo --
    # this approximates "how much of the new leg's signal is already in the
    # combo".

    remaining = list(legs)
    # Sort initial: pick the safest leg first to anchor
    remaining.sort(key=lambda l: -l.model_prob)
    combo = [remaining.pop(0)]
    p = combo[0].model_prob

    while remaining:
        # Find the next leg with the strongest connection to the existing combo
        best_idx = 0
        best_rho = -2.0
        for i, cand in enumerate(remaining):
            rho = max((pair_correlation(cand, c) for c in combo), default=0.0)
            if rho > best_rho:
                best_rho = rho
                best_idx = i
        next_leg = remaining.pop(best_idx)
        rho = best_rho
        p = _frechet_combine(p, next_leg.model_prob, rho)
        combo.append(next_leg)

    return max(0.0, min(1.0, p))


def _estimate_book_price(legs: list[Leg], naive: float) -> float:
    """Estimate Sportsbet's combined price after correlation discount.

    Approach:
    1. Collapse fully-redundant ladders (lower thresholds drop out -- they're
       implied by the higher one).
    2. For remaining same-player related-market pairs, apply a discount on
       the second leg's contribution proportional to correlation.
    3. Return the resulting product.

    This is heuristic. The bookie's true matrix is unknown.
    """
    collapsed = collapse_redundant(legs)
    if not collapsed:
        return 0.0
    if len(collapsed) == 1:
        return collapsed[0].price

    # Anchor on the highest-priced leg (most informative) and chain
    sorted_legs = sorted(collapsed, key=lambda l: -l.price)
    chained_prob = sorted_legs[0].model_prob
    chained_price = sorted_legs[0].price
    used = [sorted_legs[0]]

    for leg in sorted_legs[1:]:
        rho = max((pair_correlation(leg, u) for u in used), default=0.0)
        # Effective probability of this leg given the running combo
        # already contains correlated information.
        p_new = leg.model_prob
        joint = _frechet_combine(chained_prob, p_new, rho)
        # Marginal contribution to the combined probability
        # = joint / chained_prob, capped to [0, 1]
        if chained_prob > 0:
            marginal = max(p_new, joint / chained_prob)
        else:
            marginal = p_new
        marginal = max(0.01, min(0.999, marginal))
        # Convert to a price for this leg given prior context
        # (this is what the book effectively is pricing it at)
        eff_price = 1.0 / marginal
        chained_price *= eff_price
        chained_prob = joint
        used.append(leg)

    # Apply a small bookmaker margin on top (compounded ~5% per leg
    # collapses to roughly 8-12% on a 5-leg SGM, less on shorter)
    margin = min(0.12, 0.015 * len(collapsed))
    return chained_price * (1.0 - margin)


def combine_legs(legs: list[Leg], mode: str = "banker") -> SGMTicket:
    """Bundle a set of legs into a priced SGM ticket."""
    legs = list(legs)
    if not legs:
        return SGMTicket(legs=[], n_legs=0, mode=mode)

    naive_price = 1.0
    for leg in legs:
        naive_price *= leg.price

    # Joint probability under correlation
    p_combined = _greedy_combine_probs(legs)
    fair_price = (1.0 / p_combined) if p_combined > 0 else 0.0
    book_price = _estimate_book_price(legs, naive_price)

    ev_pct = (p_combined * book_price - 1.0) * 100.0 if book_price > 0 else 0.0

    warnings: list[str] = []
    avg_rho = avg_pair_correlation(legs)
    if avg_rho > 0.55:
        warnings.append(f"high average correlation ({avg_rho:+.2f}) -- legs share signal")
    if avg_rho < -0.10:
        warnings.append(f"negative average correlation ({avg_rho:+.2f}) -- conflicting legs")

    # Flag fully redundant pairs
    seen_keys: set[tuple] = set()
    for leg in legs:
        if leg.is_player_leg() and leg.player and leg.market:
            k = (leg.player.lower(), leg.market)
            if k in seen_keys:
                warnings.append(f"redundant ladder: multiple {leg.player} {leg.market} thresholds")
            seen_keys.add(k)

    if len(legs) > 15:
        warnings.append(f"more than 15 legs ({len(legs)}) -- consider trimming")

    return SGMTicket(
        legs=legs,
        n_legs=len(legs),
        naive_price=round(naive_price, 2),
        estimated_book_price=round(book_price, 2),
        combined_prob=round(p_combined, 4),
        combined_ev_pct=round(ev_pct, 2),
        expected_value=round(p_combined * book_price - 1.0, 4),
        fair_price=round(fair_price, 2),
        mode=mode,
        warnings=warnings,
    )
