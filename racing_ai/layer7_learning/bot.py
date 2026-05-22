"""RaceLearningBot — the 'AI that learns every race'.

Two complementary learning loops, run *per race type* (greyhound / thoroughbred):

  1. Online SGD on the heuristic model's feature weights.
     After each race we observe a one-hot winner; we backprop a single
     cross-entropy step into the linear scorer used by HeuristicModel.
     This is intentionally tiny so a few mis-predicted races don't blow up
     the priors — but over time the bot moves toward weights that fit the
     tracks you actually bet.

  2. Per-context calibration table.
     For every (race_type, track, distance, box) we store:
       - n_observed         total settled races contributing to this bucket
       - sum_predicted      sum of the model's predicted P(win) for runners
                            that occupied this box at this context
       - sum_observed       sum of one-hot winners
     The calibration multiplier is `(observed_rate / predicted_rate)`, smoothed
     toward 1.0 with a Bayesian prior so cold buckets don't explode. We apply
     this as a log-add to the model's logits before softmax — exactly the
     'this dog gets a +12% bump because Box 1 at Gawler 400m wins 12% more
     than the model expected' semantics.

  3. Insight log.
     Every learn() call also writes any meaningful new findings into the
     bot_insights table so the dashboard's 'Bot Insights' tab can surface
     them in plain English.

Persistence: the bot pickles its state to `models/bot_state.pkl` and writes
insights to SQLite via storage.db.
"""
from __future__ import annotations
import logging
import math
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import MODELS_DIR
from ..layer3_features.engine import FEATURE_NAMES
from ..layer4_models.heuristic import HeuristicModel
from ..schema import Race, RACE_TYPE_GREYHOUND, RACE_TYPE_THOROUGHBRED
from ..storage import get_db

log = logging.getLogger(__name__)


# Smoothing prior for per-context multipliers (higher = more cautious updates)
CALIBRATION_PRIOR_STRENGTH = 8.0
# Minimum observations before we trust a bucket enough to publish an insight
INSIGHT_MIN_N = 6
# Online learning rate for the heuristic feature weights
LEARNING_RATE = 0.04


@dataclass
class _ContextStats:
    n: int = 0
    sum_predicted: float = 0.0
    sum_observed: float = 0.0

    def update(self, predicted: float, observed: int) -> None:
        self.n += 1
        self.sum_predicted += float(predicted)
        self.sum_observed += float(observed)

    def multiplier(self) -> float:
        prior = CALIBRATION_PRIOR_STRENGTH
        pred = self.sum_predicted + prior * 0.5
        obs = self.sum_observed + prior * 0.5
        if pred <= 0:
            return 1.0
        return float(obs / pred)


@dataclass
class _BotState:
    # Track *deltas* applied per race type so we can report drift.
    weight_deltas: dict[str, dict[str, float]] = field(default_factory=dict)
    initial_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    # Per-context stats.  Key = "race_type|track|distance|box"
    context_stats: dict[str, _ContextStats] = field(default_factory=dict)
    n_races_seen: int = 0
    last_updated: str = ""


class RaceLearningBot:
    """Calibrates predictions and learns from settled races."""

    STATE_FILE = MODELS_DIR / "bot_state.pkl"

    def __init__(self, models: dict[str, HeuristicModel] | None = None):
        self.state = _BotState()
        self.models: dict[str, HeuristicModel] = models or {
            RACE_TYPE_GREYHOUND: HeuristicModel(race_type=RACE_TYPE_GREYHOUND),
            RACE_TYPE_THOROUGHBRED: HeuristicModel(race_type=RACE_TYPE_THOROUGHBRED),
        }
        for rt, m in self.models.items():
            self.state.initial_weights.setdefault(rt, dict(m.weights))
            self.state.weight_deltas.setdefault(rt, {f: 0.0 for f in FEATURE_NAMES})
        self.db = get_db()

    # ------------------------------------------------------------------
    # Backwards-compat shim: older code referenced `bot.gh_model`.
    # ------------------------------------------------------------------
    @property
    def gh_model(self) -> HeuristicModel:
        return self.models[RACE_TYPE_GREYHOUND]

    @property
    def tb_model(self) -> HeuristicModel:
        return self.models[RACE_TYPE_THOROUGHBRED]

    def model_for(self, race_type: str) -> HeuristicModel:
        if race_type not in self.models:
            self.models[race_type] = HeuristicModel(race_type=race_type)
            self.state.initial_weights.setdefault(race_type, dict(self.models[race_type].weights))
            self.state.weight_deltas.setdefault(race_type, {f: 0.0 for f in FEATURE_NAMES})
        return self.models[race_type]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        path = path or self.STATE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "state": self.state,
                "model_states": {rt: m.state() for rt, m in self.models.items()},
            }, f)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "RaceLearningBot":
        path = path or cls.STATE_FILE
        if not path.exists():
            return cls()
        with open(path, "rb") as f:
            blob = pickle.load(f)

        models: dict[str, HeuristicModel] = {}
        for rt, ms in (blob.get("model_states") or {}).items():
            models[rt] = HeuristicModel.from_state(ms)
        # Backwards compat with v1 bot pickles that only had gh_weights
        if not models and "gh_weights" in blob:
            models[RACE_TYPE_GREYHOUND] = HeuristicModel(
                race_type=RACE_TYPE_GREYHOUND, weights=blob["gh_weights"],
            )
        bot = cls(models=models or None)
        if blob.get("state"):
            bot.state = blob["state"]
            # Ensure new race-type buckets are initialised
            for rt in bot.models:
                bot.state.initial_weights.setdefault(rt, dict(bot.models[rt].weights))
                bot.state.weight_deltas.setdefault(rt, {f: 0.0 for f in FEATURE_NAMES})
        return bot

    @classmethod
    def get(cls) -> "RaceLearningBot":
        return cls.load()

    # ------------------------------------------------------------------
    # Calibration: applied at predict time
    # ------------------------------------------------------------------
    def _bucket_key(self, race: Race, box: int) -> str:
        return f"{race.race_type}|{race.track}|{race.distance_m}|{box}"

    def calibrate(self, race: Race, raw_probs: np.ndarray) -> np.ndarray:
        """Apply per-context multipliers to raw model probs and re-softmax.

        Mutates each runner's `bot_adjustment` for explainability and returns
        the new probabilities (sum = 1.0).
        """
        active = race.active_runners
        if len(active) != len(raw_probs):
            return raw_probs

        logits = np.log(np.clip(raw_probs, 1e-9, 1.0))
        adjustments = np.zeros_like(logits)
        for i, runner in enumerate(active):
            stats = self.state.context_stats.get(self._bucket_key(race, runner.box))
            if stats and stats.n >= 3:
                mult = stats.multiplier()
                adj = float(np.clip(np.log(mult), -1.0, 1.0))     # ±2.7x cap
                adjustments[i] = adj
                runner.bot_adjustment = adj
            else:
                runner.bot_adjustment = 0.0
            runner.raw_model_prob = float(raw_probs[i])

        adjusted = logits + adjustments
        adjusted -= adjusted.max()
        e = np.exp(adjusted)
        return e / e.sum() if e.sum() > 0 else raw_probs

    # ------------------------------------------------------------------
    # Learning: called once outcome is known
    # ------------------------------------------------------------------
    def learn(self, race: Race, finishing_order: list[str],
              feature_frame: pd.DataFrame | None = None) -> list[str]:
        """Update calibration table + model weights for this race type.

        finishing_order: list of runner_ids, position 0 = winner.
        Returns the list of new insight strings published.
        """
        if not finishing_order:
            return []

        winner_id = finishing_order[0]
        active = race.active_runners
        if not active:
            return []

        runners_by_id = {r.runner_id: r for r in active}
        if winner_id not in runners_by_id:
            log.warning("Winner %s not in active runners — skipping learn()", winner_id)
            return []

        # 1) Update per-context calibration
        for r in active:
            key = self._bucket_key(race, r.box)
            stats = self.state.context_stats.setdefault(key, _ContextStats())
            stats.update(predicted=r.raw_model_prob or r.win_prob,
                         observed=int(r.runner_id == winner_id))

        # 2) One online-SGD step on whichever heuristic model serves this race type
        if feature_frame is not None and not feature_frame.empty:
            self._sgd_step(race, feature_frame, winner_id)

        # 3) Bookkeeping
        self.state.n_races_seen += 1
        self.state.last_updated = datetime.utcnow().isoformat()

        new_insights = self._maybe_publish_insights(race)

        race.finishing_order = list(finishing_order)
        race.settled = True
        self.db.record_result(race.race_id, finishing_order)

        self.save()
        return new_insights

    # ------------------------------------------------------------------
    # Online SGD for the linear scorer of this race type
    # ------------------------------------------------------------------
    def _sgd_step(self, race: Race, frame: pd.DataFrame, winner_id: str) -> None:
        """One cross-entropy gradient step on the per-race-type weights.

            p_i = exp(w·x_i) / Σ exp(w·x_j)
            ∂L/∂w_k = Σ_i (p_i - y_i) x_i,k
        """
        if frame.empty or winner_id not in frame.index:
            return
        model = self.model_for(race.race_type)

        feats = frame[FEATURE_NAMES].astype(float).values        # (n, k)
        runner_ids = list(frame.index)
        scores = feats @ np.array([model.weights[f] for f in FEATURE_NAMES])
        s = scores - scores.max()
        e = np.exp(s)
        p = e / e.sum() if e.sum() > 0 else np.full_like(e, 1.0 / len(e))

        y = np.zeros_like(p)
        y[runner_ids.index(winner_id)] = 1.0
        grad = (p - y) @ feats
        update = -LEARNING_RATE * grad
        update = np.clip(update, -0.10, 0.10)    # per-feature clip

        deltas = {f: float(update[i]) for i, f in enumerate(FEATURE_NAMES)}
        model.update_weights(deltas)
        agg = self.state.weight_deltas.setdefault(
            race.race_type, {f: 0.0 for f in FEATURE_NAMES},
        )
        for k, v in deltas.items():
            agg[k] = agg.get(k, 0.0) + v

    # ------------------------------------------------------------------
    # Insights
    # ------------------------------------------------------------------
    def _maybe_publish_insights(self, race: Race) -> list[str]:
        new: list[str] = []
        for r in race.active_runners:
            key = self._bucket_key(race, r.box)
            stats = self.state.context_stats.get(key)
            if not stats or stats.n < INSIGHT_MIN_N:
                continue
            mult = stats.multiplier()
            if abs(math.log(max(mult, 1e-9))) < 0.20:
                continue
            direction = "outperforms" if mult > 1.0 else "underperforms"
            pct = (mult - 1.0) * 100.0
            text = (
                f"Box {r.box} at {race.track} {race.distance_m}m ({race.race_type}) "
                f"{direction} the model by {pct:+.0f}% over {stats.n} races."
            )
            sev = "high" if abs(pct) >= 30 else "medium"
            self.db.insert_insight(text, severity=sev, context_key=key)
            new.append(text)
        return new

    # ------------------------------------------------------------------
    # Public read API used by the UI
    # ------------------------------------------------------------------
    def summary(self) -> dict:
        n_buckets = len(self.state.context_stats)
        active_buckets = sum(
            1 for s in self.state.context_stats.values() if s.n >= INSIGHT_MIN_N
        )
        weights_now = {rt: dict(m.weights) for rt, m in self.models.items()}
        weights_drift = {
            rt: {
                f: round(self.models[rt].weights[f] -
                         self.state.initial_weights.get(rt, {}).get(f, 0.0), 4)
                for f in FEATURE_NAMES
            }
            for rt in self.models
        }
        return {
            "races_seen": self.state.n_races_seen,
            "last_updated": self.state.last_updated,
            "context_buckets": n_buckets,
            "mature_buckets": active_buckets,
            "weights_now": weights_now,
            "weights_drift": weights_drift,
            # Backwards-compat shorthand for the greyhound-only UI
            "greyhound_weights": weights_now.get(RACE_TYPE_GREYHOUND, {}),
            "greyhound_weight_drift": weights_drift.get(RACE_TYPE_GREYHOUND, {}),
        }

    def top_calibration_buckets(self, top_n: int = 20) -> list[dict]:
        rows = []
        for key, stats in self.state.context_stats.items():
            if stats.n < INSIGHT_MIN_N:
                continue
            race_type, track, dist, box = key.split("|")
            mult = stats.multiplier()
            rows.append({
                "race_type": race_type,
                "track": track,
                "distance_m": int(dist),
                "box": int(box),
                "n": stats.n,
                "predicted_rate": stats.sum_predicted / stats.n,
                "observed_rate": stats.sum_observed / stats.n,
                "multiplier": mult,
                "deviation_pct": (mult - 1.0) * 100.0,
            })
        rows.sort(key=lambda r: -abs(r["deviation_pct"]))
        return rows[:top_n]

    def recent_insights(self, limit: int = 30) -> list[dict]:
        return self.db.recent_insights(limit=limit)
