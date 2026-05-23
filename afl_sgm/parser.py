"""Parse a copied SGM market dump into Leg objects.

Designed for the Sportsbet "Hot Legs" tab format. Each leg looks like:

    9
    Max Holmes
    25+ Disposals
    28
    25
    29
    33
    26

    1.65

with a blank line between the history and the price. Team H2H legs use the
same shape but with W/L history and "Head to Head" as the market line.

The parser is deliberately defensive -- it walks the text line by line and
treats any malformed block as garbage to discard, so partial pastes still
yield the legs that did parse cleanly.
"""
from __future__ import annotations
import hashlib
import re
import unicodedata
from typing import Iterable, Optional

from .schema import (
    Leg,
    PLAYER_STAT_MARKETS,
    MARKET_DISPOSALS, MARKET_GOALS, MARKET_FANTASY, MARKET_MARKS,
    MARKET_TACKLES, MARKET_KICKS, MARKET_HANDBALLS, MARKET_HITOUTS,
    MARKET_CLEARANCES, MARKET_H2H,
)


# Map raw stat words (singular, lowercased) to canonical market keys.
_STAT_WORD_MAP = {
    "disposal":   MARKET_DISPOSALS,
    "goal":       MARKET_GOALS,
    "fantasy":    MARKET_FANTASY,
    "mark":       MARKET_MARKS,
    "tackle":     MARKET_TACKLES,
    "kick":       MARKET_KICKS,
    "handball":   MARKET_HANDBALLS,
    "hitout":     MARKET_HITOUTS,
    "clearance":  MARKET_CLEARANCES,
}

# Lines that are tab navigation / section headers and never part of a leg.
_NOISE = {
    "all", "popular", "hot legs", "same game multi", "match preview",
    "top markets", "show all", "show less", "show more",
    "new player most groups", "pick your own disposals",
    "disposal markets", "pick your own goals", "goal scorer markets",
    "player afl fantasy", "player marks", "player tackles", "player kicks",
    "player handballs", "player hitouts", "player clearances",
    "player goal combos", "player disposal combos",
    "match goals in every quarter", "team goals in every quarter",
    "quick bet markets", "margin markets", "handicap markets",
    "total points markets", "total goals markets",
    "1st and last to score markets", "\"1st to\" markets",
    "quarter/half markets", "doubles markets",
    "pick your line", "pick your own total", "head to head",
    # Stat tab headers that must NOT be eaten when they're not part of a leg
    # (the parser will only consume these inside a valid block).
    "disposals", "goals", "fantasy points", "hitouts", "marks", "tackles",
    "kicks", "handballs", "clearances",
    # Misc
    "first disposal", "1st goal", "1st or 2nd goal", "1st, 2nd or 3rd goal",
    "big win little win", "line", "total game points - over/under",
    # NB: do NOT noise-filter threshold strings like "20+ Disposals" or
    # "1+ Goal" -- they're legitimate market lines inside leg blocks. The
    # block-shape validator will reject any tab-bar instance on its own
    # because tab-bar entries don't have 5 numeric history lines around them.
}

# Tab-bar lists that cluster between sections; lines with just team names are
# OK because they only become a leg when followed by "Head to Head".
_NOISE_PASSTHROUGH_TEAMS = {"geelong", "sydney"}

_PRICE_RE = re.compile(r"^\$?(\d{1,4}(?:\.\d{1,2}))$")
_THRESHOLD_RE = re.compile(r"^(\d+)\+\s+([A-Za-z][A-Za-z ]+?)s?$")
_INT_RE = re.compile(r"^\d{1,3}$")


def _normalise(text: str) -> list[str]:
    """Strip control chars / NBSPs and split into trimmed lines."""
    text = unicodedata.normalize("NFKC", text).replace("\xa0", " ")
    return [line.strip() for line in text.split("\n")]


def _is_noise(line: str) -> bool:
    low = line.lower()
    if low in _NOISE:
        return True
    # Match-preview style "8m" countdowns
    if re.fullmatch(r"\d{1,2}[hms]", low):
        return True
    return False


def _classify_market(line: str) -> Optional[tuple[str, float]]:
    """Map a market line ('25+ Disposals') -> (canonical_key, threshold)."""
    m = _THRESHOLD_RE.match(line)
    if not m:
        return None
    threshold = float(m.group(1))
    stat_word = m.group(2).strip().lower()
    # Strip "points" / "point" suffix for fantasy
    stat_word = re.sub(r"\s+points?$", "", stat_word).strip()
    # Drop trailing 's' if our singular map doesn't already include it
    if stat_word.endswith("s") and stat_word[:-1] in _STAT_WORD_MAP:
        stat_word = stat_word[:-1]
    if stat_word in _STAT_WORD_MAP:
        return _STAT_WORD_MAP[stat_word], threshold
    return None


def _make_leg_id(*parts: str) -> str:
    raw = "|".join(p.lower() for p in parts if p is not None)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def _build_player_leg(buf: list[str], price: float) -> Optional[Leg]:
    """Block layout: [maybe jersey, name, market, h1..h5]."""
    if len(buf) < 7:
        return None
    history_raw = buf[-5:]
    market_line = buf[-6]
    # Validate history is all numeric
    if not all(_INT_RE.match(h) for h in history_raw):
        return None
    market = _classify_market(market_line)
    if not market:
        return None
    market_key, threshold = market

    # Identify name + (optional) jersey
    if len(buf) >= 8 and _INT_RE.match(buf[-8]):
        jersey = int(buf[-8])
        name = buf[-7]
    else:
        jersey = None
        name = buf[-7]

    if not name or _is_noise(name):
        return None

    # Allow apostrophes, hyphens, spaces; reject if it looks like a stat label
    if not re.search(r"[A-Za-z]", name) or re.search(r"\d", name):
        return None

    history = [float(h) for h in history_raw]
    description = f"{name} {market_line}"

    return Leg(
        leg_id=_make_leg_id(name, market_key, str(threshold)),
        leg_type="player_threshold",
        description=description,
        player=name,
        jersey=jersey,
        market=market_key,
        threshold=threshold,
        history=history,
        price=price,
    )


def _build_h2h_leg(buf: list[str], price: float) -> Optional[Leg]:
    """Block layout: [team, 'Head to Head', h1..h5] -- 5 W/L/D values."""
    if len(buf) < 7:
        return None
    history_raw = buf[-5:]
    market_line = buf[-6]
    if market_line.lower() != "head to head":
        return None
    if not all(h in {"W", "L", "D"} for h in history_raw):
        return None
    team = buf[-7]
    if not team or _is_noise(team) and team.lower() not in _NOISE_PASSTHROUGH_TEAMS:
        # team must look like a real word
        return None
    if not re.search(r"[A-Za-z]", team):
        return None
    description = f"{team} to win (H2H)"
    return Leg(
        leg_id=_make_leg_id(team, "h2h"),
        leg_type="team_h2h",
        description=description,
        team=team,
        market=MARKET_H2H,
        threshold=None,
        history=list(history_raw),
        price=price,
    )


def _flush_block(buf: list[str], price: float) -> Optional[Leg]:
    """Try player-leg first, then H2H."""
    leg = _build_player_leg(buf, price)
    if leg:
        return leg
    return _build_h2h_leg(buf, price)


def parse_paste(text: str) -> list[Leg]:
    """Top-level entry: extract every parseable Leg from raw paste text.

    Duplicates (same player + market + threshold) are deduped, keeping the
    first occurrence (which in the Hot Legs section is the lowest-priced /
    most-recent quote).
    """
    lines = _normalise(text)

    legs: list[Leg] = []
    seen_ids: set[str] = set()

    buf: list[str] = []
    expecting_price = False

    for raw in lines:
        line = raw.strip()

        if expecting_price:
            m = _PRICE_RE.match(line)
            if m:
                try:
                    price = float(m.group(1))
                except ValueError:
                    price = 0.0
                if price > 1.0:
                    leg = _flush_block(buf, price)
                    if leg and leg.leg_id not in seen_ids:
                        legs.append(leg)
                        seen_ids.add(leg.leg_id)
                buf = []
                expecting_price = False
                continue
            else:
                # Missed the price -- abandon this block and treat current line
                # as the start of a new buffer (if it's content).
                buf = []
                expecting_price = False
                if line and not _is_noise(line):
                    buf.append(line)
                continue

        if line == "":
            # Block terminator. If buffer looks plausibly leg-shaped (>=6 lines
            # so far) we expect a price next; otherwise discard.
            if len(buf) >= 6:
                expecting_price = True
            else:
                buf = []
            continue

        # Filter pure-noise lines.
        if _is_noise(line):
            # A market line like "Head to Head" is noise outside a leg context
            # but is meaningful inside. Heuristic: only keep when buf already
            # has at least 1 plausible content line (a team name).
            if line.lower() == "head to head" and buf and re.search(r"[A-Za-z]", buf[-1]):
                buf.append(line)
            else:
                buf = []
            continue

        buf.append(line)

    return legs


def parse_paste_with_metadata(text: str) -> dict:
    """Convenience: parse + summary (for the UI)."""
    legs = parse_paste(text)
    by_player: dict[str, int] = {}
    by_market: dict[str, int] = {}
    teams: set[str] = set()
    for leg in legs:
        if leg.player:
            by_player[leg.player] = by_player.get(leg.player, 0) + 1
        if leg.team:
            teams.add(leg.team)
        if leg.market:
            by_market[leg.market] = by_market.get(leg.market, 0) + 1
    return {
        "legs": legs,
        "n_legs": len(legs),
        "n_players": len(by_player),
        "by_market": by_market,
        "teams": sorted(teams),
    }
