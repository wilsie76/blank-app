"""Core domain objects passed between layers.

Kept as plain dataclasses (not pydantic) to stay light and serialization-friendly.
Schema is shared between thoroughbreds and greyhounds; greyhound-specific
fields are optional and default to None so old code keeps working unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


# --- enums (kept as strings for SQLite friendliness) -------------------------
RACE_TYPE_THOROUGHBRED = "thoroughbred"
RACE_TYPE_GREYHOUND = "greyhound"
RACE_TYPE_HARNESS = "harness"   # supported via the thoroughbred path for now


@dataclass
class Runner:
    # --- Core identity (required) -------------------------------------------
    runner_id: str
    name: str
    box: int                       # barrier / box number
    weight_kg: float
    jockey: str                    # "" for greyhounds
    trainer: str
    last_5_form: str               # last 5 finishing positions, e.g. "1-3-2-5-1" or "13251"
    days_since_run: int
    win_odds_open: float           # opening market price (decimal)
    win_odds: float                # current price (decimal)
    place_odds: float

    # --- Status -------------------------------------------------------------
    scratched: bool = False

    # --- Populated downstream (model output) --------------------------------
    features: dict = field(default_factory=dict)
    raw_score: float = 0.0
    win_prob: float = 0.0          # final calibrated probability
    place_prob: float = 0.0
    ev_pct: float = 0.0
    confidence: float = 0.0
    no_bet: bool = True
    no_bet_reason: str = ""
    kelly_stake: float = 0.0

    # --- Optional rich form (used by greyhound path; safe for thoroughbreds) -
    last_6_form: str = ""
    best_time_s: Optional[float] = None
    win_pct: Optional[float] = None          # 0-100 scale (so "23.81%" -> 23.81)
    place_pct: Optional[float] = None        # 0-100
    career_starts: Optional[int] = None
    career_wins: Optional[int] = None
    career_places: Optional[int] = None      # 2nds + 3rds
    td_starts: Optional[int] = None          # track + distance starts
    td_wins: Optional[int] = None
    td_places: Optional[int] = None
    sb_rating: Optional[float] = None
    sire: str = ""
    dam: str = ""

    # --- Provenance for the bot to attribute learnings ----------------------
    raw_model_prob: float = 0.0     # uncalibrated model probability
    bot_adjustment: float = 0.0     # log-multiplier applied by the bot

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Race:
    race_id: str
    track: str
    race_number: int
    distance_m: int
    surface: str                   # "turf" / "synthetic" / "dirt" / "track"
    track_condition: str           # firm / good / soft / heavy / fast etc.
    weather: str
    start_time: datetime
    runners: list[Runner] = field(default_factory=list)

    # Race classification — drives feature/model dispatch
    race_type: str = RACE_TYPE_THOROUGHBRED

    # Populated by simulation / feature engine
    exacta: list[tuple] = field(default_factory=list)   # [((id1,id2), prob), ...]
    trifecta: list[tuple] = field(default_factory=list)
    first4: list[tuple] = field(default_factory=list)
    pace_pressure: float = 0.0
    track_bias: float = 0.0

    # Populated when actual finishing order is recorded (powers the bot)
    finishing_order: list[str] = field(default_factory=list)
    settled: bool = False

    @property
    def active_runners(self) -> list[Runner]:
        return [r for r in self.runners if not r.scratched]

    @property
    def context_key(self) -> str:
        """Coarse bucket the bot uses for per-context calibration."""
        return f"{self.race_type}|{self.track}|{self.distance_m}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start_time"] = self.start_time.isoformat()
        return d
