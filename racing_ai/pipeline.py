"""Pipeline orchestrator — wires all 7 layers and the learning bot.

By default both race types route through the bot's learnable HeuristicModel
(linear scorer with racing-wisdom priors that the bot tunes online). The
trained Ensemble is available via `use_ensemble_for_thoroughbreds=True`,
but it's only safe once you've replaced the synthetic training set with
real history.
"""
from __future__ import annotations
import logging
from datetime import date

from .layer1_intake import get_source, DataSource
from .layer2_cleaning import clean_races
from .layer3_features.engine import build_features, race_to_dataframe
from .layer4_models import Ensemble, train_and_save
from .layer5_simulation import simulate_race, SimulationResult
from .layer6_value import evaluate_race
from .layer7_learning import RaceLearningBot
from .storage import get_db
from .schema import Race, RACE_TYPE_GREYHOUND, RACE_TYPE_THOROUGHBRED
from .config import DEFAULT_SIMULATIONS

log = logging.getLogger(__name__)


class Pipeline:
    """End-to-end: intake -> clean -> features -> model -> bot -> sim -> EV -> store."""

    def __init__(
        self,
        source: str | DataSource = "synthetic",
        auto_train: bool = True,
        bot: RaceLearningBot | None = None,
        use_ensemble_for_thoroughbreds: bool = False,
    ):
        self.source = get_source(source) if isinstance(source, str) else source
        self.use_ensemble_for_thoroughbreds = use_ensemble_for_thoroughbreds

        # Trained thoroughbred ensemble. Lazy-loaded; only required when
        # use_ensemble_for_thoroughbreds=True.
        self._ensemble: Ensemble | None = None
        if use_ensemble_for_thoroughbreds and not Ensemble.exists():
            if auto_train:
                log.info("No trained ensemble found — training on synthetic data...")
                train_and_save(n_meetings=60)
            else:
                raise RuntimeError(
                    "use_ensemble_for_thoroughbreds=True but no trained ensemble. "
                    "Call train_and_save() first."
                )

        # Persistent learning bot. Owns one HeuristicModel per race type.
        self.bot: RaceLearningBot = bot or RaceLearningBot.load()
        self.db = get_db()

    # ------------------------------------------------------------------
    @property
    def ensemble(self) -> Ensemble:
        if self._ensemble is None:
            if not Ensemble.exists():
                raise RuntimeError("Ensemble not trained. Call train_and_save() first.")
            self._ensemble = Ensemble.load()
        return self._ensemble

    def _predictor_for(self, race: Race):
        """Pick the model whose predict_race_probs we'll use for this race."""
        if (race.race_type == RACE_TYPE_THOROUGHBRED
                and self.use_ensemble_for_thoroughbreds):
            return self.ensemble
        return self.bot.model_for(race.race_type)

    # ------------------------------------------------------------------
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
            result = self._predict_one(race, n_simulations=n_simulations, persist=persist)
            if result is not None:
                out.append(result)
        return out

    def run_one(self, race: Race, n_simulations: int = DEFAULT_SIMULATIONS,
                persist: bool = False) -> tuple[Race, SimulationResult]:
        cleaned = clean_races([race])
        if not cleaned:
            raise ValueError("Race failed validation")
        result = self._predict_one(cleaned[0], n_simulations=n_simulations, persist=persist)
        if result is None:
            raise ValueError("Race produced no predictions")
        return result

    # ------------------------------------------------------------------
    def _predict_one(
        self, race: Race, n_simulations: int, persist: bool,
    ) -> tuple[Race, SimulationResult] | None:
        build_features(race)
        df = race_to_dataframe(race)
        if df.empty:
            return None

        predictor = self._predictor_for(race)
        raw_probs, confidence, _per_model = predictor.predict_race_probs(df)

        # Bot calibration — applies per-(track,distance,box) multipliers learned online
        calibrated = self.bot.calibrate(race, raw_probs)

        # Write final probs onto runners (df row order == active_runners order)
        for runner, p in zip(race.active_runners, calibrated):
            runner.win_prob = float(p)

        sim = simulate_race(
            runner_ids=df.index.tolist(),
            win_probs=calibrated,
            n_simulations=n_simulations,
        )
        race.exacta, race.trifecta, race.first4 = sim.exacta, sim.trifecta, sim.first4

        for runner in race.active_runners:
            runner.place_prob = sim.place.get(runner.runner_id, 0.0)

        evaluate_race(race, confidence=confidence)

        if persist:
            self.db.save_race_with_predictions(race)

        return race, sim

    # ------------------------------------------------------------------
    def record_result(self, race: Race, finishing_order: list[str]) -> list[str]:
        """Tell the bot the actual finishing order. Returns new insights."""
        df = race_to_dataframe(race)
        return self.bot.learn(race, finishing_order, feature_frame=df)
