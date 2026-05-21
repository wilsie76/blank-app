"""Training entry point.

Generates a synthetic training set, builds features, samples winners using
a hidden 'true' weight vector, and fits the ensemble. In production you'd
replace generate_training_set() with a query against your historical DB.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

from ..layer1_intake.synthetic import SyntheticSource
from ..layer2_cleaning import clean_races
from ..layer3_features.engine import build_features, race_to_dataframe, FEATURE_NAMES
from .ensemble import Ensemble

log = logging.getLogger(__name__)


def generate_training_set(n_meetings: int = 60, seed: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    """Build (X, y) where y=1 marks the actual winner of each race."""
    rng = np.random.default_rng(seed)
    src = SyntheticSource(seed=seed)
    all_X: list[pd.DataFrame] = []
    all_y: list[np.ndarray] = []

    # Hidden 'true' coefficient vector — the signal the models must recover.
    true_w = rng.normal(0, 1, size=len(FEATURE_NAMES))
    # Bias the most predictive features
    feat_idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    true_w[feat_idx["implied_prob"]] = 3.0
    true_w[feat_idx["form_score"]] = 1.2
    true_w[feat_idx["box_efficiency"]] = 0.6
    true_w[feat_idx["fatigue_index"]] = -0.8
    true_w[feat_idx["steam_index"]] = 1.5
    true_w[feat_idx["speed_differential"]] = 1.0

    for _ in range(n_meetings):
        races = src.fetch_races()
        races = clean_races(races)
        for race in races:
            build_features(race)
            df = race_to_dataframe(race)
            if df.empty:
                continue
            scores = df.values @ true_w + rng.normal(0, 0.4, size=len(df))
            probs = np.exp(scores - scores.max())
            probs = probs / probs.sum()
            winner_idx = rng.choice(len(df), p=probs)
            y = np.zeros(len(df), dtype=int)
            y[winner_idx] = 1
            all_X.append(df.reset_index(drop=True))
            all_y.append(y)

    X = pd.concat(all_X, ignore_index=True)
    y = np.concatenate(all_y)
    return X, y


def train_and_save(n_meetings: int = 60) -> tuple[Ensemble, dict[str, float], int]:
    log.info("Generating training set: %d meetings", n_meetings)
    X, y = generate_training_set(n_meetings=n_meetings)
    log.info("Training rows: %d  positives: %d", len(y), int(y.sum()))
    ens = Ensemble()
    scores = ens.fit(X, y)
    ens.save()
    return ens, scores, len(y)
