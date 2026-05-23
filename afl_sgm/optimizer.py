"""15-leg SGM optimiser.

Two modes
---------

Banker  (default; "ensure best winning outcome")
    Maximise combined_prob. Below a price floor we use a smooth penalty so
    the optimiser still climbs toward at-least-modest payouts but never
    sacrifices much probability for dollars. This is the recommended mode
    when the goal is to actually cash the ticket.

Value
    Maximise EV = combined_prob * estimated_book_price - 1. Picks legs with
    real edge even when probability is moderate. Higher variance.

Algorithm
---------

Greedy with full ticket re-scoring at each step:

    selected = []
    while len(selected) < target_legs:
        for cand in remaining_pool:
            ticket = combine_legs(selected + [cand])
            score  = objective(ticket)
        pick cand with highest score
    return combine_legs(selected)

The combiner internally collapses redundant ladders, so the optimiser
cannot game its own price by stacking same-player threshold legs.
Filtering rules:
    - drop legs with EV below min_per_leg_ev (default -10%)
    - skip legs that would create a hard contradiction with the existing
      combo (mutual-exclusive H2H pairs)
"""
from __future__ import annotations
from dataclasses import dataclass, field

from .schema import Leg, SGMTicket, SGMMode
from .combiner import combine_legs
from .correlation import pair_correlation


@dataclass
class OptimizerConfig:
    target_legs: int = 15           # how many legs to build
    min_legs: int = 4               # don't return shorter tickets even if score plateaus
    mode: str = SGMMode.BANKER.value
    min_book_price: float = 3.00    # Banker price floor: below this, soft penalty kicks in
    min_per_leg_ev_pct: float = -10.0  # filter out legs the model thinks are seriously over-priced
    min_per_leg_prob: float = 0.0   # optional probability floor for individual legs
    allow_redundant_ladder: bool = False  # keep one threshold per (player, market)
    # The optimiser internally never picks legs that the model deems impossible
    # (e.g. opposing-team H2H both selected); this is hard-coded.


# ---------------------------------------------------------------------------
# Objective functions

def _banker_score(ticket: SGMTicket, cfg: OptimizerConfig) -> float:
    """Maximise probability, with a smooth penalty below the price floor.

    Below floor: score = combined_prob * (book_price / floor)^0.5
    At/above floor: score = combined_prob

    The 0.5 exponent makes the price-pull mild so we don't sacrifice 5% of
    win probability to gain a marginal extra dollar of payout.
    """
    p = ticket.combined_prob
    if p <= 0:
        return 0.0
    if cfg.min_book_price <= 0 or ticket.estimated_book_price >= cfg.min_book_price:
        return p
    ratio = max(0.0, ticket.estimated_book_price / cfg.min_book_price)
    return p * (ratio ** 0.5)


def _value_score(ticket: SGMTicket, cfg: OptimizerConfig) -> float:
    """Maximise EV. Negative EV is heavily penalised (returns ~0)."""
    if ticket.combined_prob <= 0 or ticket.estimated_book_price <= 1.0:
        return 0.0
    ev = ticket.combined_prob * ticket.estimated_book_price - 1.0
    # Cap at zero from below; the search prefers slightly +EV combos.
    return max(0.0, ev)


def _scorer(cfg: OptimizerConfig):
    return _banker_score if cfg.mode == SGMMode.BANKER.value else _value_score


# ---------------------------------------------------------------------------
# Pre-filter & infeasibility checks

def _is_compatible(cand: Leg, selected: list[Leg]) -> bool:
    """True if cand can be added without contradiction."""
    for s in selected:
        rho = pair_correlation(cand, s)
        # Mutually exclusive (both teams' H2H)
        if rho <= -0.999:
            return False
    return True


def _violates_ladder(cand: Leg, selected: list[Leg], allow_redundant: bool) -> bool:
    """If allow_redundant=False, reject same player + same market threshold pairs."""
    if allow_redundant:
        return False
    if not (cand.is_player_leg() and cand.player and cand.market):
        return False
    for s in selected:
        if s.is_player_leg() and s.player and s.market:
            if (s.player.lower() == cand.player.lower()
                    and s.market == cand.market
                    and s.threshold != cand.threshold):
                return True
    return False


def _filter_pool(legs: list[Leg], cfg: OptimizerConfig) -> list[Leg]:
    return [
        leg for leg in legs
        if leg.ev_pct >= cfg.min_per_leg_ev_pct
        and leg.model_prob >= cfg.min_per_leg_prob
        and leg.price > 1.0
    ]


# ---------------------------------------------------------------------------
# Search

def optimise_sgm(
    legs: list[Leg],
    mode: str | SGMMode = SGMMode.BANKER,
    target_legs: int = 15,
    min_book_price: float = 3.00,
    min_per_leg_ev_pct: float = -10.0,
    allow_redundant_ladder: bool = False,
    initial_legs: list[Leg] | None = None,
) -> SGMTicket:
    """Greedy search for an N-leg SGM.

    ``initial_legs`` lets the caller pre-pin specific legs (e.g. user
    selections in the UI); the optimiser fills the remaining slots.
    """
    cfg = OptimizerConfig(
        target_legs=target_legs,
        mode=mode.value if isinstance(mode, SGMMode) else str(mode),
        min_book_price=min_book_price,
        min_per_leg_ev_pct=min_per_leg_ev_pct,
        allow_redundant_ladder=allow_redundant_ladder,
    )

    pool = _filter_pool(legs, cfg)
    pool_by_id = {leg.leg_id: leg for leg in pool}

    selected: list[Leg] = list(initial_legs or [])
    selected_ids: set[str] = {leg.leg_id for leg in selected}

    score_fn = _scorer(cfg)
    target = max(1, cfg.target_legs)

    while len(selected) < target:
        best_leg: Leg | None = None
        best_score = -1.0
        # Score the current ticket as a baseline (in case nothing helps)
        for cand in pool:
            if cand.leg_id in selected_ids:
                continue
            if not _is_compatible(cand, selected):
                continue
            if _violates_ladder(cand, selected, cfg.allow_redundant_ladder):
                continue
            ticket = combine_legs(selected + [cand])
            s = score_fn(ticket, cfg)
            if s > best_score:
                best_score = s
                best_leg = cand
        if best_leg is None:
            break
        selected.append(best_leg)
        selected_ids.add(best_leg.leg_id)

    final = combine_legs(selected, mode=cfg.mode)
    return final


def ticket_ladder(
    legs: list[Leg],
    mode: str | SGMMode = SGMMode.BANKER,
    leg_counts: list[int] | None = None,
    **kwargs,
) -> list[SGMTicket]:
    """Run the optimiser at multiple leg counts so the user can compare.

    Useful for the UI: shows how prob / price / EV change as legs are added,
    so the user can pick the sweet spot rather than locking in 15.
    """
    leg_counts = leg_counts or [3, 5, 8, 10, 12, 15]
    out: list[SGMTicket] = []
    for n in leg_counts:
        ticket = optimise_sgm(legs, mode=mode, target_legs=n, **kwargs)
        out.append(ticket)
    return out
