"""Synthetic race generator.

Produces realistic-shaped data so every downstream layer can run end-to-end.
The 'true' win probability is a hidden function of features — this lets the
ensemble actually learn something during training.
"""
from __future__ import annotations
import random
from datetime import date, datetime, timedelta
from ..schema import Race, Runner
from .base import DataSource

_TRACKS = ["Randwick", "Flemington", "Eagle Farm", "Caulfield", "Rosehill"]
_CONDITIONS = ["firm", "good", "soft", "heavy"]
_SURFACES = ["turf", "synthetic"]
_WEATHER = ["clear", "overcast", "rainy", "windy"]


def _form_string(rng: random.Random) -> str:
    return "-".join(str(rng.randint(1, 9)) for _ in range(5))


class SyntheticSource(DataSource):
    name = "synthetic"
    is_live = False

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def fetch_races(self, target_date: date | None = None) -> list[Race]:
        target_date = target_date or date.today()
        rng = self._rng
        n_races = rng.randint(6, 9)
        races: list[Race] = []
        base_time = datetime.combine(target_date, datetime.min.time()) + timedelta(hours=12)

        for race_no in range(1, n_races + 1):
            track = rng.choice(_TRACKS)
            n_runners = rng.randint(8, 14)
            distance = rng.choice([1000, 1200, 1400, 1600, 2000, 2400])
            cond = rng.choices(_CONDITIONS, weights=[2, 5, 2, 1])[0]

            runners = []
            # Hidden "true" strength per runner — used to set realistic odds
            true_strengths = [rng.gauss(0, 1) for _ in range(n_runners)]
            # Softmax-ish to probabilities
            import math
            exps = [math.exp(s) for s in true_strengths]
            tot = sum(exps)
            true_probs = [e / tot for e in exps]

            for i, p in enumerate(true_probs):
                # Bookmaker odds = 1/p inflated by 15% overround + noise
                fair = 1.0 / max(p, 0.01)
                book = fair * rng.uniform(1.10, 1.20)
                open_odds = book * rng.uniform(0.95, 1.10)
                runners.append(Runner(
                    runner_id=f"R{race_no}_{i+1}",
                    name=f"Runner {i+1}",
                    box=i + 1,
                    weight_kg=round(rng.uniform(52.0, 60.5), 1),
                    jockey=f"J. Jockey{rng.randint(1,40)}",
                    trainer=f"T. Trainer{rng.randint(1,25)}",
                    last_5_form=_form_string(rng),
                    days_since_run=rng.randint(7, 60),
                    win_odds_open=round(open_odds, 2),
                    win_odds=round(book, 2),
                    place_odds=round(book * 0.35 + 1.0, 2),
                    scratched=rng.random() < 0.06,
                ))

            races.append(Race(
                race_id=f"{target_date.isoformat()}_{track[:3].upper()}_{race_no}",
                track=track,
                race_number=race_no,
                distance_m=distance,
                surface=rng.choice(_SURFACES),
                track_condition=cond,
                weather=rng.choice(_WEATHER),
                start_time=base_time + timedelta(minutes=30 * race_no),
                runners=runners,
            ))
        return races

    def fetch_closing_odds(self, race_id: str) -> dict[str, float]:
        # Synthetic CLV: small drift from current odds
        return {}
