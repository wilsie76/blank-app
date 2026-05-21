"""Layer 7 — Learning tracker.

Aggregates settled bets to give you the metrics that actually matter:
  ROI, hit rate, CLV (closing line value), market-agreement, miss types.

Designed to be called from the dashboard or a scheduled job.
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from ..storage import get_db


@dataclass
class PerformanceSnapshot:
    n_bets: int
    n_settled: int
    hit_rate: float
    roi_pct: float
    avg_clv_pct: float
    profit: float
    market_agreement: float       # fraction of bets where we agreed with favourite-direction
    by_miss_type: dict[str, int]  # {"value_lost":n, "favourite_beat_us":n, ...}


def _classify_miss(row: pd.Series) -> str:
    if row.get("won") == 1:
        return "win"
    if row.get("odds_closing") and row["odds_closing"] > row["odds_taken"]:
        return "drift_loss"             # odds drifted out — we were on the wrong horse
    if row.get("odds_closing") and row["odds_closing"] < row["odds_taken"]:
        return "steam_loss"             # market agreed but it lost anyway -> bad luck
    return "flat_loss"


class LearningTracker:
    def __init__(self):
        self.db = get_db()

    def snapshot(self) -> PerformanceSnapshot:
        df = self.db.all_bets_df()
        if df.empty:
            return PerformanceSnapshot(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, {})

        settled = df[df["settled"] == 1].copy()
        n_bets = len(df)
        n_settled = len(settled)

        if n_settled == 0:
            return PerformanceSnapshot(n_bets, 0, 0.0, 0.0, 0.0, 0.0, 0.0, {})

        settled["payout"] = settled["payout"].fillna(0.0)
        profit = float(settled["payout"].sum() - settled["stake"].sum())
        roi = profit / float(settled["stake"].sum()) * 100.0 if settled["stake"].sum() > 0 else 0.0
        hit = float((settled["won"] == 1).mean())

        # CLV = how much better/worse our taken odds were than closing
        clv_rows = settled.dropna(subset=["odds_closing"])
        if not clv_rows.empty:
            clv = float(((clv_rows["odds_taken"] - clv_rows["odds_closing"]) / clv_rows["odds_closing"]).mean() * 100.0)
        else:
            clv = 0.0

        # Market agreement proxy: did we bet at *better* than closing odds?
        agree = float((clv_rows["odds_taken"] > clv_rows["odds_closing"]).mean()) if not clv_rows.empty else 0.0

        miss_types = settled.apply(_classify_miss, axis=1).value_counts().to_dict()

        return PerformanceSnapshot(
            n_bets=n_bets, n_settled=n_settled,
            hit_rate=hit, roi_pct=roi, avg_clv_pct=clv,
            profit=profit, market_agreement=agree,
            by_miss_type={str(k): int(v) for k, v in miss_types.items()},
        )

    def equity_curve(self) -> pd.DataFrame:
        df = self.db.all_bets_df()
        if df.empty:
            return pd.DataFrame(columns=["placed_at", "cumulative_pnl"])
        settled = df[df["settled"] == 1].copy()
        if settled.empty:
            return pd.DataFrame(columns=["placed_at", "cumulative_pnl"])
        settled = settled.sort_values("placed_at")
        settled["pnl"] = settled["payout"].fillna(0.0) - settled["stake"]
        settled["cumulative_pnl"] = settled["pnl"].cumsum()
        return settled[["placed_at", "cumulative_pnl"]]
