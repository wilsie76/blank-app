"""AFL Same Game Multi (SGM) builder.

Pipeline:
    paste text -> parser -> Leg[] -> probability model -> correlation engine
                                          -> SGM combiner -> 15-leg optimiser

Two optimisation modes:
    * Banker (default) -- maximise P(SGM wins). Picks safest legs that share
      a positive-correlation thesis. This is what you want when the goal is
      "ensure best winning outcome".
    * Value -- maximise expected value. Picks +EV legs even if win
      probability is lower.
"""
from __future__ import annotations

__version__ = "0.1.0"

# Lazy re-exports: import sub-modules directly (e.g. `from afl_sgm.parser
# import parse_paste`) so partial installs don't blow up the package.
__all__ = [
    "Leg",
    "SGMTicket",
    "SGMMode",
    "parse_paste",
    "score_leg",
    "score_legs",
    "correlation_matrix",
    "redundant_pairs",
    "combine_legs",
    "optimise_sgm",
]


def __getattr__(name):
    if name in {"Leg", "SGMTicket", "SGMMode"}:
        from .schema import Leg, SGMTicket, SGMMode
        return {"Leg": Leg, "SGMTicket": SGMTicket, "SGMMode": SGMMode}[name]
    if name == "parse_paste":
        from .parser import parse_paste
        return parse_paste
    if name in {"score_leg", "score_legs"}:
        from .model import score_leg, score_legs
        return {"score_leg": score_leg, "score_legs": score_legs}[name]
    if name in {"correlation_matrix", "redundant_pairs"}:
        from .correlation import correlation_matrix, redundant_pairs
        return {
            "correlation_matrix": correlation_matrix,
            "redundant_pairs": redundant_pairs,
        }[name]
    if name == "combine_legs":
        from .combiner import combine_legs
        return combine_legs
    if name == "optimise_sgm":
        from .optimizer import optimise_sgm
        return optimise_sgm
    raise AttributeError(f"module 'afl_sgm' has no attribute {name!r}")
