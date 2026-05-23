"""Domain objects for the SGM pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class SGMMode(str, Enum):
    """Optimisation objective."""
    BANKER = "banker"   # maximise P(SGM wins). Recommended for "best winning outcome".
    VALUE = "value"     # maximise EV. Higher variance.


# Canonical market keys. Adapter code translates raw paste text to these.
MARKET_DISPOSALS = "disposals"
MARKET_GOALS     = "goals"
MARKET_FANTASY   = "fantasy"
MARKET_MARKS     = "marks"
MARKET_TACKLES   = "tackles"
MARKET_KICKS     = "kicks"
MARKET_HANDBALLS = "handballs"
MARKET_HITOUTS   = "hitouts"
MARKET_CLEARANCES = "clearances"
MARKET_H2H       = "h2h"

PLAYER_STAT_MARKETS = {
    MARKET_DISPOSALS, MARKET_GOALS, MARKET_FANTASY, MARKET_MARKS,
    MARKET_TACKLES, MARKET_KICKS, MARKET_HANDBALLS, MARKET_HITOUTS,
    MARKET_CLEARANCES,
}

# Markets that share underlying signal -- used by the correlation engine.
RELATED_MARKETS = {
    MARKET_DISPOSALS:  {MARKET_KICKS: 0.85, MARKET_HANDBALLS: 0.80, MARKET_FANTASY: 0.80, MARKET_MARKS: 0.45, MARKET_TACKLES: 0.30},
    MARKET_KICKS:      {MARKET_DISPOSALS: 0.85, MARKET_FANTASY: 0.65, MARKET_MARKS: 0.50},
    MARKET_HANDBALLS:  {MARKET_DISPOSALS: 0.80, MARKET_FANTASY: 0.55, MARKET_CLEARANCES: 0.40},
    MARKET_FANTASY:    {MARKET_DISPOSALS: 0.80, MARKET_MARKS: 0.55, MARKET_TACKLES: 0.45, MARKET_GOALS: 0.40, MARKET_HITOUTS: 0.50},
    MARKET_MARKS:      {MARKET_DISPOSALS: 0.45, MARKET_FANTASY: 0.55, MARKET_GOALS: 0.30},
    MARKET_TACKLES:    {MARKET_DISPOSALS: 0.30, MARKET_FANTASY: 0.45, MARKET_CLEARANCES: 0.35},
    MARKET_GOALS:      {MARKET_FANTASY: 0.40, MARKET_MARKS: 0.30},
    MARKET_HITOUTS:    {MARKET_FANTASY: 0.50, MARKET_CLEARANCES: 0.25},
    MARKET_CLEARANCES: {MARKET_HANDBALLS: 0.40, MARKET_TACKLES: 0.35, MARKET_HITOUTS: 0.25},
}


@dataclass
class Leg:
    """One bookmaker market that could be added to an SGM."""

    leg_id: str                       # stable hash of (player|team, market, threshold)
    leg_type: str                     # "player_threshold" | "team_h2h"
    description: str                  # human label e.g. "Bailey Smith 25+ Disposals"

    # market identity
    player: Optional[str] = None
    team: Optional[str] = None
    jersey: Optional[int] = None
    market: Optional[str] = None      # canonical market key (see constants above)
    threshold: Optional[float] = None # numeric threshold (e.g. 25 for "25+ Disposals")

    # raw history (numeric for stat markets, "W"/"L"/"D" for H2H)
    history: list = field(default_factory=list)

    # price
    price: float = 0.0                # decimal odds

    # populated by the model
    hits: int = 0                     # n times threshold was met in last_5
    n_games: int = 0
    raw_hit_rate: float = 0.0         # hits / n_games (no smoothing)
    model_prob: float = 0.0           # smoothed probability
    ev_pct: float = 0.0               # (model_prob * price - 1) * 100
    confidence: float = 0.0           # 0..1, lower with smaller samples / more variance
    flags: list = field(default_factory=list)   # warnings ("small sample", "stale", etc.)

    def to_dict(self) -> dict:
        return asdict(self)

    def is_player_leg(self) -> bool:
        return self.leg_type == "player_threshold"

    def is_h2h(self) -> bool:
        return self.leg_type == "team_h2h"


@dataclass
class SGMTicket:
    """A combined SGM bet."""

    legs: list[Leg]
    naive_price: float = 0.0          # product of leg prices (NOT what the book offers)
    estimated_book_price: float = 0.0 # after correlation discount approximation
    combined_prob: float = 0.0        # P(all legs land) under correlation
    combined_ev_pct: float = 0.0      # using estimated_book_price
    expected_value: float = 0.0       # combined_prob * estimated_book_price - 1
    fair_price: float = 0.0           # 1 / combined_prob (zero-edge price)
    n_legs: int = 0
    mode: str = ""
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["legs"] = [leg.to_dict() for leg in self.legs]
        return d
