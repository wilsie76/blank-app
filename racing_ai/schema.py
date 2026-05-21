"""Core domain objects passed between layers.

Kept as plain dataclasses (not pydantic) to stay light and serialization-friendly.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


@dataclass
class Runner:
    runner_id: str
    name: str
    box: int                       # barrier / box number
    weight_kg: float
    jockey: str
    trainer: str
    last_5_form: str               # e.g. "1-3-2-5-1"
    days_since_run: int
    win_odds_open: float           # opening market price (decimal)
    win_odds: float                # current price (decimal)
    place_odds: float
    scratched: bool = False

    # Populated downstream
    features: dict = field(default_factory=dict)
    raw_score: float = 0.0         # ensemble logit / score
    win_prob: float = 0.0          # calibrated probability
    place_prob: float = 0.0
    ev_pct: float = 0.0
    confidence: float = 0.0
    no_bet: bool = True
    no_bet_reason: str = ""
    kelly_stake: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Race:
    race_id: str
    track: str
    race_number: int
    distance_m: int
    surface: str                   # "turf" / "synthetic" / "dirt"
    track_condition: str           # firm / good / soft / heavy
    weather: str
    start_time: datetime
    runners: list[Runner] = field(default_factory=list)

    # Populated by simulation
    exacta: list[tuple] = field(default_factory=list)   # [((id1,id2), prob), ...]
    trifecta: list[tuple] = field(default_factory=list)
    first4: list[tuple] = field(default_factory=list)
    pace_pressure: float = 0.0
    track_bias: float = 0.0

    @property
    def active_runners(self) -> list[Runner]:
        return [r for r in self.runners if not r.scratched]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start_time"] = self.start_time.isoformat()
        return d
