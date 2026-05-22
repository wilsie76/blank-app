"""Linear heuristic model — works for any race type.

The greyhound version was the original; we generalised it because:
  - the 13-feature schema is shared across race types,
  - the Ensemble shipped in this repo is trained only on synthetic data so its
    predictions on real thoroughbred form are unreliable until you replace
    `generate_training_set()` with real history,
  - a calibrated linear scorer with racing-wisdom priors is a *much* safer
    default than an ML model trained on the wrong distribution.

The bot's online SGD updates these weights as real outcomes come in; over
time the model becomes specifically tuned to whatever circuits you're
actually backing.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..layer3_features.engine import FEATURE_NAMES
from ..schema import RACE_TYPE_GREYHOUND, RACE_TYPE_THOROUGHBRED


# ----------------------------------------------------------------------------
# Default weights per race type.  Positive = better runner.
# ----------------------------------------------------------------------------
_DEFAULTS_GH: dict[str, float] = {
    "implied_prob":        4.0,
    "form_score":          1.6,
    "box_efficiency":      1.5,
    "speed_differential":  1.2,
    "fatigue_index":      -0.9,
    "steam_index":         0.8,
    "pace_pressure":      -0.2,
    "track_bias":          0.2,
    "volatility_score":   -0.3,
    "market_rank":        -0.05,
    "days_since_run":      0.0,
    "weight_kg":           0.0,
    "closing_line_delta":  0.5,
}

_DEFAULTS_TB: dict[str, float] = {
    # Thoroughbreds: form, days-since-run and steam matter more; box less so.
    "implied_prob":        4.5,    # market is *very* informative for TB
    "form_score":          2.0,
    "box_efficiency":      0.6,
    "speed_differential":  0.8,
    "fatigue_index":      -0.7,
    "steam_index":         1.2,
    "pace_pressure":       0.0,
    "track_bias":          0.3,
    "volatility_score":   -0.2,
    "market_rank":        -0.05,
    "days_since_run":      0.0,
    "weight_kg":          -0.05,   # heavier weight = small headwind
    "closing_line_delta":  0.7,
}


def default_weights(race_type: str) -> dict[str, float]:
    if race_type == RACE_TYPE_GREYHOUND:
        return dict(_DEFAULTS_GH)
    return dict(_DEFAULTS_TB)


class HeuristicModel:
    """Drop-in replacement for Ensemble.  Linear scorer + softmax-within-race."""

    def __init__(self, race_type: str = RACE_TYPE_GREYHOUND,
                 weights: dict[str, float] | None = None):
        self.race_type = race_type
        self.weights: dict[str, float] = default_weights(race_type)
        if weights:
            for k, v in weights.items():
                if k in self.weights:
                    self.weights[k] = float(v)
        self.feature_names = list(FEATURE_NAMES)

    @property
    def NAME(self) -> str:
        return f"heuristic_{self.race_type}"

    # ------------------------------------------------------------------
    def _score_row(self, row: pd.Series) -> float:
        return float(sum(self.weights[f] * row[f] for f in self.feature_names))

    def predict_raw(self, X: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        Xv = X[self.feature_names].astype(float)
        scores = np.array([self._score_row(Xv.iloc[i]) for i in range(len(Xv))])
        s = scores - scores.max()
        e = np.exp(s)
        probs = e / e.sum() if e.sum() > 0 else np.full_like(e, 1.0 / len(e))
        return probs, {self.NAME: probs}

    def predict_race_probs(self, X: pd.DataFrame) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
        probs, per_model = self.predict_raw(X)
        n = len(probs)
        if n <= 1:
            return probs, 1.0, per_model
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-9, 1.0)))
        max_entropy = float(np.log(n))
        confidence = float(np.clip(1.0 - entropy / max_entropy, 0.0, 1.0))
        confidence = 0.4 + 0.6 * confidence    # soften
        return probs, confidence, per_model

    # ------------------------------------------------------------------
    def update_weights(self, deltas: dict[str, float]) -> None:
        for k, dv in deltas.items():
            if k in self.weights:
                self.weights[k] += dv

    def state(self) -> dict:
        return {"weights": dict(self.weights), "race_type": self.race_type}

    @classmethod
    def from_state(cls, state: dict) -> "HeuristicModel":
        return cls(race_type=state.get("race_type", RACE_TYPE_GREYHOUND),
                   weights=state.get("weights"))


# ----------------------------------------------------------------------------
# Backwards-compatible alias used elsewhere in the codebase.
# ----------------------------------------------------------------------------
class GreyhoundHeuristicModel(HeuristicModel):
    """Alias kept so any callers that imported the old name still work."""
    def __init__(self, weights: dict[str, float] | None = None):
        super().__init__(race_type=RACE_TYPE_GREYHOUND, weights=weights)


# Compatibility constant
DEFAULT_WEIGHTS = _DEFAULTS_GH
