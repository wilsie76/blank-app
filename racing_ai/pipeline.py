"""Pipeline orchestrator — wires all 7 layers."""
from __future__ import annotations
import logging
from datetime import date
from typing import Iterable

from .layer1_intake import get_source, DataSource
from .layer2_cleaning import clean_races
from .layer3_features.engine import build_features, race_to_dataframe
from .layer4_models import Ensemble, train_and_save
from .layer5_simulation import simulate_race, SimulationResult
from .layer6_value import evaluate_race
from .storage import get_db
from .schema import Race
from .config import DEFAULT_SIMULATIONS

log = logging.getLogger(__name__)


class Pipeline:
    """End-to-end: intake -> clean -> features -> ensemble -> sim -> EV -> store."""

    def __init__(self, source: str | DataSource = "synthetic", auto_train: bool = True):
        self.source = get_source(source) if isinstance(source, str) else source
        if not Ensemble.exists():
            if auto_train:
                log.info("No trained ensemble found — training on synthetic data...")
                train_and_save(n_meetings=60)
            else:
                raise RuntimeError("No trained ensemble. Call train_and_save() first.")
        self.ensemble = Ensemble.load()
        self.db = get_db()

    # ---------------------------------------------------------------------
    def run(
        self,
        target_date: date | None = None,
        n_simulations: int = DEFAULT_SIMULATIONS,
        persist: bool = True,
    ) -> list[tuple[Race, SimulationResult]]:
        races = self.source.fetch_races(target_date)
        races = clean_races(races)
        out: list[tuple[Race, SimulationResult]] = []

        for race in races:
            build_features(race)
            df = race_to_dataframe(race)
            if df.empty:
                continue

            probs, confidence, _per_model = self.ensemble.predict_race_probs(df)

            # Write probs back onto runners (df row order == active_runners order)
            for runner, p in zip(race.active_runners, probs):
                runner.win_prob = float(p)

            sim = simulate_race(
                runner_ids=df.index.tolist(),
                win_probs=probs,
                n_simulations=n_simulations,
            )
            # Store simulation outputs on race for the UI
            race.exacta = sim.exacta
            race.trifecta = sim.trifecta
            race.first4 = sim.first4

            # Use simulated place probabilities as place_prob
            for runner in race.active_runners:
                runner.place_prob = sim.place.get(runner.runner_id, 0.0)

            evaluate_race(race, confidence=confidence)

            if persist:
                self.db.save_race_with_predictions(race)

            out.append((race, sim))

        return out

    # ---------------------------------------------------------------------
    def run_one(self, race: Race, n_simulations: int = DEFAULT_SIMULATIONS) -> tuple[Race, SimulationResult]:
        cleaned = clean_races([race])
        if not cleaned:
            raise ValueError("Race failed validation")
        race = cleaned[0]
        build_features(race)
        df = race_to_dataframe(race)
        probs, confidence, _ = self.ensemble.predict_race_probs(df)
        for runner, p in zip(race.active_runners, probs):
            runner.win_prob = float(p)
        sim = simulate_race(df.index.tolist(), probs, n_simulations=n_simulations)
        race.exacta, race.trifecta, race.first4 = sim.exacta, sim.trifecta, sim.first4
        for runner in race.active_runners:
            runner.place_prob = sim.place.get(runner.runner_id, 0.0)
        evaluate_race(race, confidence=confidence)
        return race, sim
