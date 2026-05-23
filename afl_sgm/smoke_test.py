"""Smoke test for the AFL SGM pipeline.

Runs end-to-end on the Geelong v Sydney fixture and prints:
  * parse summary
  * top 10 safest single legs
  * Banker mode: best 15-leg SGM (max P(win) with price floor)
  * Value mode:  best 15-leg SGM (max EV)
  * Ticket ladder for both modes (3, 5, 8, 10, 12, 15 legs)

Run from the project root:
    python3 -m afl_sgm.smoke_test
"""
from __future__ import annotations
from pathlib import Path

from .parser import parse_paste
from .model import score_legs
from .schema import SGMMode
from .optimizer import optimise_sgm, ticket_ladder


def _print_ticket(ticket, label=""):
    print(f"\n{'=' * 78}")
    print(f"  {label}  ({ticket.n_legs} legs, mode={ticket.mode})")
    print("=" * 78)
    if ticket.warnings:
        print("Warnings:")
        for w in ticket.warnings:
            print(f"  - {w}")
    print(f"\n{'Leg':<46} {'Price':>6} {'Hits':>5} {'Model%':>7} {'EV%':>7}")
    print("-" * 78)
    for leg in ticket.legs:
        print(f"{leg.description[:46]:<46} {leg.price:>6.2f} "
              f"{leg.hits}/{leg.n_games:<3} {leg.model_prob*100:>6.1f}%  {leg.ev_pct:>+5.1f}%")
    print("-" * 78)
    print(f"  Naive product price : ${ticket.naive_price:.2f}  "
          f"(theoretical -- bookie won't offer this)")
    print(f"  Estimated book price: ${ticket.estimated_book_price:.2f}  "
          f"(after correlation discount)")
    print(f"  Model fair price    : ${ticket.fair_price:.2f}  "
          f"(zero-edge price; if Sportsbet offers above this, +EV)")
    print(f"  Combined P(wins)    : {ticket.combined_prob*100:.1f}%")
    print(f"  Combined EV         : {ticket.combined_ev_pct:+.1f}%  "
          f"(at estimated book price)")


def main():
    fixture = Path(__file__).parent / "fixtures" / "geelong_v_sydney.txt"
    text = fixture.read_text()

    legs = parse_paste(text)
    score_legs(legs)

    print(f"Parsed and scored {len(legs)} legs across "
          f"{len({l.player for l in legs if l.player})} players "
          f"and {len({l.team for l in legs if l.team})} teams.\n")

    print("Top 10 safest single legs (by model probability):")
    print("-" * 78)
    for leg in sorted(legs, key=lambda l: -l.model_prob)[:10]:
        print(f"  {leg.description[:44]:<44} @ {leg.price:>5.2f}  "
              f"P={leg.model_prob*100:>5.1f}%  EV={leg.ev_pct:>+5.1f}%")

    # ---- Banker mode --------------------------------------------------------
    banker = optimise_sgm(
        legs, mode=SGMMode.BANKER, target_legs=15,
        min_book_price=3.0, min_per_leg_ev_pct=-10.0,
    )
    _print_ticket(banker, "BANKER MODE -- best 15-leg SGM for max P(win)")

    # ---- Value mode ---------------------------------------------------------
    value = optimise_sgm(
        legs, mode=SGMMode.VALUE, target_legs=15,
        min_per_leg_ev_pct=0.0,
    )
    _print_ticket(value, "VALUE MODE -- best 15-leg SGM for max EV")

    # ---- Ladder -------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("  BANKER ladder (compare across leg counts)")
    print("=" * 78)
    print(f"{'Legs':>4}  {'P(win)':>7}  {'Book$':>7}  {'Fair$':>7}  {'EV%':>6}")
    for ticket in ticket_ladder(legs, mode=SGMMode.BANKER, min_book_price=3.0,
                                min_per_leg_ev_pct=-10.0):
        print(f"{ticket.n_legs:>4}  {ticket.combined_prob*100:>6.1f}%  "
              f"${ticket.estimated_book_price:>6.2f}  ${ticket.fair_price:>6.2f}  "
              f"{ticket.combined_ev_pct:>+5.1f}%")

    print(f"\n{'=' * 78}")
    print("  VALUE ladder")
    print("=" * 78)
    print(f"{'Legs':>4}  {'P(win)':>7}  {'Book$':>7}  {'Fair$':>7}  {'EV%':>6}")
    for ticket in ticket_ladder(legs, mode=SGMMode.VALUE, min_per_leg_ev_pct=0.0):
        print(f"{ticket.n_legs:>4}  {ticket.combined_prob*100:>6.1f}%  "
              f"${ticket.estimated_book_price:>6.2f}  ${ticket.fair_price:>6.2f}  "
              f"{ticket.combined_ev_pct:>+5.1f}%")


if __name__ == "__main__":
    main()
