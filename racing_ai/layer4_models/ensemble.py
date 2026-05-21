"""Layer 4 — Ensemble.

Stacks XGBoost + LightGBM + CatBoost + a small NN (MLP).
Each model outputs P(win) for a runner. Final probability is the *softmax-
within-race* of the average raw score, which guarantees per-race probabilities
sum to 1.0 — exactly what the simulation engine needs.

If any of the heavy libs aren't installed the ensemble degrades gracefully:
the missing model is skipped. If none are available we fall back to
sklearn GradientBoostingClassifier.
"""
from __future__ import annotations
import logging
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from ..config import MODELS_DIR
from ..layer3_features.engine import FEATURE_NAMES

log = logging.getLogger(__name__)

# Optional heavy deps — detect once at import.
MODELS_AVAILABLE: dict[str, bool] = {
    "xgboost": False, "lightgbm": False, "catboost": False, "mlp": True, "gbdt": True,
}
try:
    import xgboost as xgb  # type: ignore
    MODELS_AVAILABLE["xgboost"] = True
except Exception:
    xgb = None
try:
    import lightgbm as lgb  # type: ignore
    MODELS_AVAILABLE["lightgbm"] = True
except Exception:
    lgb = None
try:
    from catboost import CatBoostClassifier  # type: ignore
    MODELS_AVAILABLE["catboost"] = True
except Exception:
    CatBoostClassifier = None  # type: ignore


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class Ensemble:
    """Train/predict facade. Models are trained per-instance and pickled."""

    MODEL_FILE = MODELS_DIR / "ensemble.pkl"

    def __init__(self):
        self.models: dict[str, object] = {}
        self.feature_names = list(FEATURE_NAMES)

    # --- training ----------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
        """Train all available models. Returns dict of train accuracy per model."""
        Xdf = X[self.feature_names].astype(float)
        Xv = Xdf.values
        scores: dict[str, float] = {}

        if MODELS_AVAILABLE["xgboost"]:
            m = xgb.XGBClassifier(
                n_estimators=400, max_depth=5, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85,
                eval_metric="logloss", n_jobs=-1, verbosity=0,
            )
            m.fit(Xdf, y)
            self.models["xgboost"] = m
            scores["xgboost"] = float(m.score(Xdf, y))

        if MODELS_AVAILABLE["lightgbm"]:
            m = lgb.LGBMClassifier(
                n_estimators=400, num_leaves=31, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85, n_jobs=-1, verbose=-1,
            )
            m.fit(Xdf, y)
            self.models["lightgbm"] = m
            scores["lightgbm"] = float(m.score(Xdf, y))

        if MODELS_AVAILABLE["catboost"]:
            m = CatBoostClassifier(
                iterations=400, depth=6, learning_rate=0.05,
                verbose=False, allow_writing_files=False,
            )
            m.fit(Xdf, y)
            self.models["catboost"] = m
            scores["catboost"] = float(m.score(Xdf, y))

        # Always train a small NN + a sklearn GBDT (cheap insurance)
        nn = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
        nn.fit(Xv, y)
        self.models["mlp"] = nn
        scores["mlp"] = float(nn.score(Xv, y))

        if not any(k in self.models for k in ("xgboost", "lightgbm", "catboost")):
            gbdt = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
            gbdt.fit(Xv, y)
            self.models["gbdt"] = gbdt
            scores["gbdt"] = float(gbdt.score(Xv, y))

        log.info("Ensemble trained: %s", scores)
        return scores

    # --- inference ---------------------------------------------------------
    def _model_proba(self, name: str, X) -> np.ndarray:
        m = self.models[name]
        if hasattr(m, "predict_proba"):
            return np.asarray(m.predict_proba(X))[:, 1]
        return np.asarray(m.predict(X)).astype(float)

    def predict_raw(self, X: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return (averaged P(win) per runner, {model_name: per-model probs})."""
        if not self.models:
            raise RuntimeError("Ensemble is not trained. Run train_and_save() first.")
        Xdf = X[self.feature_names].astype(float)
        # MLP/sklearn want numpy; tree libs handle either but keep names for LGBM
        Xv = Xdf.values
        per_model: dict[str, np.ndarray] = {}
        for name in self.models:
            arr = Xdf if name in ("xgboost", "lightgbm", "catboost") else Xv
            per_model[name] = self._model_proba(name, arr)
        avg = np.mean(np.stack(list(per_model.values())), axis=0)
        return avg, per_model

    def predict_race_probs(self, X: pd.DataFrame) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
        """Return (per-runner P(win), confidence, per-model probs).

        Confidence = 1 - mean pairwise std across model outputs (higher = models agree).
        """
        avg, per_model = self.predict_raw(X)
        # Softmax-normalise within race so ∑ P(win) = 1
        race_probs = _softmax(np.log(np.clip(avg, 1e-6, 1.0)))
        # Confidence: how much do the models agree on the favourite?
        stack = np.stack(list(per_model.values()))   # (n_models, n_runners)
        std_per_runner = stack.std(axis=0)
        confidence = float(np.clip(1.0 - std_per_runner.mean() * 5.0, 0.0, 1.0))
        return race_probs, confidence, per_model

    # --- persistence -------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        path = path or self.MODEL_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"models": self.models, "feature_names": self.feature_names}, f)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "Ensemble":
        path = path or cls.MODEL_FILE
        with open(path, "rb") as f:
            blob = pickle.load(f)
        e = cls()
        e.models = blob["models"]
        e.feature_names = blob["feature_names"]
        return e

    @classmethod
    def exists(cls, path: Path | None = None) -> bool:
        return (path or cls.MODEL_FILE).exists()
