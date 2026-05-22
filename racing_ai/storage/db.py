"""SQLite DAO. Schema kept neutral so swapping to PostgreSQL is just changing
the connection string + a few type names (TEXT->TEXT, REAL->DOUBLE PRECISION).
"""
from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator
from ..config import DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS races (
    race_id        TEXT PRIMARY KEY,
    track          TEXT NOT NULL,
    race_number    INTEGER NOT NULL,
    distance_m     INTEGER NOT NULL,
    surface        TEXT,
    track_condition TEXT,
    weather        TEXT,
    start_time     TEXT NOT NULL,
    pace_pressure  REAL,
    track_bias     REAL,
    raw_json       TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id        TEXT NOT NULL,
    runner_id      TEXT NOT NULL,
    win_prob       REAL NOT NULL,
    confidence     REAL NOT NULL,
    win_odds       REAL NOT NULL,
    ev_pct         REAL NOT NULL,
    no_bet         INTEGER NOT NULL,
    no_bet_reason  TEXT,
    kelly_stake    REAL,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_predictions_race ON predictions(race_id);

CREATE TABLE IF NOT EXISTS bets (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id        TEXT NOT NULL,
    runner_id      TEXT NOT NULL,
    market         TEXT NOT NULL,            -- win | place | exacta | trifecta | first4
    stake          REAL NOT NULL,
    odds_taken     REAL NOT NULL,
    odds_closing   REAL,                     -- filled later for CLV
    settled        INTEGER DEFAULT 0,
    won            INTEGER,                  -- 0/1, NULL until settled
    payout         REAL,
    placed_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_bets_race ON bets(race_id);

CREATE TABLE IF NOT EXISTS results (
    race_id        TEXT PRIMARY KEY,
    finishing_order_json TEXT NOT NULL,      -- ["R1_3","R1_7",...]
    settled_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bot_insights (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    text           TEXT NOT NULL,
    severity       TEXT NOT NULL,            -- low | medium | high
    context_key    TEXT,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_bot_insights_ctx ON bot_insights(context_key);
"""


class Database:
    def __init__(self, path: str | Path = DB_PATH):
        self.path = Path(path)
        self._init_schema()

    def _init_schema(self) -> None:
        with self.connect() as cx:
            cx.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        cx = sqlite3.connect(self.path)
        cx.row_factory = sqlite3.Row
        cx.execute("PRAGMA foreign_keys = ON")
        try:
            yield cx
            cx.commit()
        finally:
            cx.close()

    # --- writes ----------------------------------------------------------
    def save_race_with_predictions(self, race) -> None:
        now = datetime.utcnow().isoformat()
        with self.connect() as cx:
            cx.execute(
                "INSERT OR REPLACE INTO races VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    race.race_id, race.track, race.race_number, race.distance_m,
                    race.surface, race.track_condition, race.weather,
                    race.start_time.isoformat(),
                    race.pace_pressure, race.track_bias,
                    json.dumps(race.to_dict(), default=str),
                ),
            )
            for r in race.active_runners:
                cx.execute(
                    "INSERT INTO predictions "
                    "(race_id, runner_id, win_prob, confidence, win_odds, ev_pct, "
                    "no_bet, no_bet_reason, kelly_stake, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (race.race_id, r.runner_id, r.win_prob, r.confidence,
                     r.win_odds, r.ev_pct, int(r.no_bet), r.no_bet_reason,
                     r.kelly_stake, now),
                )

    def record_bet(self, race_id: str, runner_id: str, market: str,
                   stake: float, odds_taken: float) -> int:
        with self.connect() as cx:
            cur = cx.execute(
                "INSERT INTO bets (race_id, runner_id, market, stake, odds_taken, placed_at) "
                "VALUES (?,?,?,?,?,?)",
                (race_id, runner_id, market, stake, odds_taken, datetime.utcnow().isoformat()),
            )
            return int(cur.lastrowid)

    def settle_bet(self, bet_id: int, won: bool, odds_closing: float | None = None) -> None:
        with self.connect() as cx:
            row = cx.execute("SELECT stake, odds_taken FROM bets WHERE id=?", (bet_id,)).fetchone()
            if not row:
                return
            payout = row["stake"] * row["odds_taken"] if won else 0.0
            cx.execute(
                "UPDATE bets SET settled=1, won=?, payout=?, odds_closing=? WHERE id=?",
                (int(won), payout, odds_closing, bet_id),
            )

    def record_result(self, race_id: str, finishing_order: list[str]) -> None:
        with self.connect() as cx:
            cx.execute(
                "INSERT OR REPLACE INTO results VALUES (?,?,?)",
                (race_id, json.dumps(finishing_order), datetime.utcnow().isoformat()),
            )

    def insert_insight(self, text: str, severity: str = "medium",
                       context_key: str | None = None) -> int:
        with self.connect() as cx:
            cur = cx.execute(
                "INSERT INTO bot_insights (text, severity, context_key, created_at) "
                "VALUES (?,?,?,?)",
                (text, severity, context_key, datetime.utcnow().isoformat()),
            )
            return int(cur.lastrowid)

    # --- reads -----------------------------------------------------------
    def all_bets_df(self):
        import pandas as pd
        with self.connect() as cx:
            return pd.read_sql_query("SELECT * FROM bets", cx)

    def predictions_for_race(self, race_id: str):
        import pandas as pd
        with self.connect() as cx:
            return pd.read_sql_query(
                "SELECT * FROM predictions WHERE race_id=? ORDER BY win_prob DESC",
                cx, params=(race_id,),
            )

    def recent_insights(self, limit: int = 30) -> list[dict]:
        with self.connect() as cx:
            rows = cx.execute(
                "SELECT id, text, severity, context_key, created_at "
                "FROM bot_insights ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]


_db: Database | None = None


def get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db
