"""Central configuration. Tweak thresholds here."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DB_PATH = DATA_DIR / "racing.db"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# --- Feature engine ----------------------------------------------------------
RECENT_FORM_RACES = 5

# --- Simulation --------------------------------------------------------------
DEFAULT_SIMULATIONS = 10_000

# --- Value engine ------------------------------------------------------------
MIN_EV_PCT = 5.0          # below this => no-bet
MAX_VOLATILITY = 0.65     # above this => no-bet
MIN_CONFIDENCE = 0.55     # ensemble agreement floor
KELLY_FRACTION = 0.25     # fractional Kelly (conservative)

# --- Learning ----------------------------------------------------------------
MIN_BANKROLL = 1000.0     # for stake sizing display
