"""HTTP clients for MLB Stats API and Baseball Savant."""

from __future__ import annotations

import csv
import io
import json
import logging
import sqlite3
import time
from datetime import date, timedelta

import requests

from bts import cache
from bts.config import (
    MLB_API_BASE,
    RATE_LIMIT_DELAY,
    REQUEST_TIMEOUT,
    RETRY_ATTEMPTS,
    RETRY_BACKOFF,
    SAVANT_XBA_URL,
)
from bts.models import (
    BattingStats,
    GameInfo,
    LineupEntry,
    PitcherBio,
    PitchingStats,
    PlayerBio,
    StartingPitcher,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Low-level HTTP
# ---------------------------------------------------------------------------

_last_request_time = 0.0


def _rate_limit() -> None:
    """Enforce minimum delay between requests."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < RATE_LIMIT_DELAY:
        time.sleep(RATE_LIMIT_DELAY - elapsed)
    _last_request_time = time.time()


def _get(url: str, *, timeout: int = REQUEST_TIMEOUT) -> requests.Response:
    """GET with retry + backoff."""
    for attempt in range(RETRY_ATTEMPTS):
        _rate_limit()
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt == RETRY_ATTEMPTS - 1:
                raise
            wait = RETRY_BACKOFF[attempt] if attempt < len(RETRY_BACKOFF) else RETRY_BACKOFF[-1]
            log.warning("Request failed (%s), retrying in %.1fs: %s", e, wait, url)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _mlb_get_json(path: str, conn: sqlite3.Connection, ttl: int) -> dict:
    """Fetch a MLB Stats API endpoint with caching."""
    url = f"{MLB_API_BASE}{path}"
    cached = cache.get_cached_response(conn, url)
    if cached is not None:
        return json.loads(cached)
    resp = _get(url)
    cache.put_cached_response(conn, url, resp.text, ttl)
    return resp.json()


# ---------------------------------------------------------------------------
# Schedule + Lineups
# ---------------------------------------------------------------------------

def get_schedule(
    conn: sqlite3.Connection, game_date: date
) -> list[GameInfo]:
    """Fetch the day's game schedule."""
    ds = game_date.isoformat()
    data = _mlb_get_json(
        f"/schedule?sportId=1&date={ds}&hydrate=probablePitcher,lineups,team",
        conn,
        ttl=1800,  # 30 min
    )
    games: list[GameInfo] = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            status = g.get("status", {}).get("detailedState", "Unknown")
            games.append(
                GameInfo(
                    game_pk=g["gamePk"],
                    game_date=game_date,
                    away_team=g["teams"]["away"]["team"]["name"],
                    home_team=g["teams"]["home"]["team"]["name"],
                    venue_name=g.get("venue", {}).get("name", "Unknown"),
                    status=status,
                )
            )
    return games


def get_lineups_and_pitchers(
    conn: sqlite3.Connection, game_pk: int, game_date: date
) -> tuple[list[LineupEntry], list[StartingPitcher]]:
    """Extract lineups and probable/starting pitchers from the schedule data."""
    ds = game_date.isoformat()
    data = _mlb_get_json(
        f"/schedule?sportId=1&date={ds}&hydrate=probablePitcher,lineups,team",
        conn,
        ttl=1800,
    )
    lineups: list[LineupEntry] = []
    pitchers: list[StartingPitcher] = []

    for d in data.get("dates", []):
        for g in d.get("games", []):
            if g["gamePk"] != game_pk:
                continue

            for side, is_home in [("away", False), ("home", True)]:
                team_data = g["teams"][side]
                team_name = team_data["team"]["name"]

                # Probable pitcher
                pp = team_data.get("probablePitcher")
                if pp:
                    # Fetch pitcher bio to get handedness
                    pbio = get_pitcher_bio(conn, pp["id"])
                    pitchers.append(
                        StartingPitcher(
                            player_id=pp["id"],
                            player_name=pp.get("fullName", "Unknown"),
                            pitch_hand=pbio.pitch_hand if pbio else "R",
                            is_home=is_home,
                            team=team_name,
                        )
                    )

                # Lineups (embedded in schedule when hydrated)
                lineup_data = g.get("lineups", {})
                side_key = "homePlayers" if is_home else "awayPlayers"
                players = lineup_data.get(side_key, [])
                for i, p in enumerate(players, start=1):
                    lineups.append(
                        LineupEntry(
                            player_id=p["id"],
                            player_name=p.get("fullName", "Unknown"),
                            lineup_slot=i,
                            is_home=is_home,
                            team=team_name,
                        )
                    )

    return lineups, pitchers


# ---------------------------------------------------------------------------
# Player bio
# ---------------------------------------------------------------------------

def get_player_bio(conn: sqlite3.Connection, player_id: int) -> PlayerBio | None:
    """Fetch player biographical data (handedness, position). Cached permanently."""
    cached = cache.get_player_bio(conn, player_id)
    if cached:
        return PlayerBio(
            player_id=player_id,
            full_name=cached["full_name"],
            bat_side=cached["bat_side"],
            primary_position=cached["primary_position"],
        )

    data = _mlb_get_json(f"/people/{player_id}", conn, ttl=365 * 86400)
    people = data.get("people", [])
    if not people:
        return None

    p = people[0]
    bat_side = p.get("batSide", {}).get("code", "R")
    position = p.get("primaryPosition", {}).get("abbreviation", "DH")
    full_name = p.get("fullName", "Unknown")

    cache.put_player_bio(conn, player_id, full_name, bat_side, position)
    return PlayerBio(
        player_id=player_id,
        full_name=full_name,
        bat_side=bat_side,
        primary_position=position,
    )


def get_pitcher_bio(conn: sqlite3.Connection, player_id: int) -> PitcherBio | None:
    """Fetch pitcher biographical data. Cached permanently."""
    cached = cache.get_pitcher_bio(conn, player_id)
    if cached:
        return PitcherBio(
            player_id=player_id,
            full_name=cached["full_name"],
            pitch_hand=cached["pitch_hand"],
        )

    data = _mlb_get_json(f"/people/{player_id}", conn, ttl=365 * 86400)
    people = data.get("people", [])
    if not people:
        return None

    p = people[0]
    pitch_hand = p.get("pitchHand", {}).get("code", "R")
    full_name = p.get("fullName", "Unknown")

    cache.put_pitcher_bio(conn, player_id, full_name, pitch_hand)
    return PitcherBio(
        player_id=player_id, full_name=full_name, pitch_hand=pitch_hand
    )


# ---------------------------------------------------------------------------
# Batting stats
# ---------------------------------------------------------------------------

def _parse_batting_stats(stat_dict: dict) -> BattingStats:
    """Parse a stats dict from the MLB API into a BattingStats."""
    return BattingStats(
        pa=int(stat_dict.get("plateAppearances", 0)),
        ab=int(stat_dict.get("atBats", 0)),
        hits=int(stat_dict.get("hits", 0)),
        bb=int(stat_dict.get("baseOnBalls", 0)),
        hbp=int(stat_dict.get("hitByPitch", 0)),
        k=int(stat_dict.get("strikeOuts", 0)),
    )


def get_season_batting(
    conn: sqlite3.Connection, player_id: int, season: int
) -> BattingStats:
    """Fetch season-level batting stats."""
    try:
        data = _mlb_get_json(
            f"/people/{player_id}/stats?stats=season&season={season}&group=hitting",
            conn,
            ttl=43200,  # 12 hours
        )
        for group in data.get("stats", []):
            for split in group.get("splits", []):
                return _parse_batting_stats(split.get("stat", {}))
    except Exception as e:
        log.debug("Failed to fetch season batting for player %s: %s", player_id, e)
    return BattingStats()


def get_recent_batting(
    conn: sqlite3.Connection,
    player_id: int,
    end_date: date,
    days: int = 14,
) -> BattingStats:
    """Fetch batting stats for a recent window ending at end_date."""
    start = end_date - timedelta(days=days)
    try:
        data = _mlb_get_json(
            f"/people/{player_id}/stats?stats=byDateRange"
            f"&startDate={start.isoformat()}&endDate={end_date.isoformat()}"
            f"&group=hitting",
            conn,
            ttl=21600,  # 6 hours
        )
        for group in data.get("stats", []):
            for split in group.get("splits", []):
                return _parse_batting_stats(split.get("stat", {}))
    except Exception as e:
        log.debug("Failed to fetch recent batting for player %s: %s", player_id, e)
    return BattingStats()


def get_platoon_splits(
    conn: sqlite3.Connection, player_id: int, season: int
) -> dict[str, BattingStats]:
    """Fetch platoon splits (vs LHP and vs RHP) for the current season.

    Returns a dict with keys "vl" and "vr".
    """
    try:
        data = _mlb_get_json(
            f"/people/{player_id}/stats?stats=statSplits"
            f"&season={season}&group=hitting"
            f"&sitCodes=vl,vr",
            conn,
            ttl=43200,
        )
    except Exception as e:
        log.debug("Failed to fetch platoon splits for player %s: %s", player_id, e)
        return {}
    splits: dict[str, BattingStats] = {}
    for group in data.get("stats", []):
        for split in group.get("splits", []):
            code = split.get("split", {}).get("code", "")
            if code in ("vl", "vr"):
                splits[code] = _parse_batting_stats(split.get("stat", {}))
    return splits


def get_career_platoon_splits(
    conn: sqlite3.Connection, player_id: int
) -> dict[str, BattingStats]:
    """Fetch career-level platoon splits as a fallback."""
    try:
        data = _mlb_get_json(
            f"/people/{player_id}/stats?stats=career&group=hitting",
            conn,
            ttl=86400,
        )
    except Exception as e:
        log.debug("Failed to fetch career splits for player %s: %s", player_id, e)
        return {}
    # Career endpoint doesn't support sitCodes — parse from splits if present
    splits: dict[str, BattingStats] = {}
    for group in data.get("stats", []):
        for split in group.get("splits", []):
            code = split.get("split", {}).get("code", "")
            if code in ("vl", "vr"):
                splits[code] = _parse_batting_stats(split.get("stat", {}))
    return splits


# ---------------------------------------------------------------------------
# Game log (for backtesting point-in-time reconstruction)
# ---------------------------------------------------------------------------

def get_game_log(
    conn: sqlite3.Connection, player_id: int, season: int
) -> list[dict]:
    """Fetch the full season game log for a batter.

    Returns a list of dicts, each with 'date' (str) and 'stat' (dict).
    """
    data = _mlb_get_json(
        f"/people/{player_id}/stats?stats=gameLog&season={season}&group=hitting",
        conn,
        ttl=86400,
    )
    entries: list[dict] = []
    for group in data.get("stats", []):
        for split in group.get("splits", []):
            entries.append({
                "date": split.get("date", ""),
                "stat": split.get("stat", {}),
                "game_pk": split.get("game", {}).get("gamePk"),
                "opponent": split.get("opponent", {}).get("name", ""),
                "is_home": split.get("isHome", False),
            })
    return entries


# ---------------------------------------------------------------------------
# Pitching stats
# ---------------------------------------------------------------------------

def _parse_pitching_stats(stat_dict: dict) -> PitchingStats:
    return PitchingStats(
        pa_faced=int(stat_dict.get("battersFaced", 0)),
        hits_allowed=int(stat_dict.get("hits", 0)),
        bb_allowed=int(stat_dict.get("baseOnBalls", 0)),
        k=int(stat_dict.get("strikeOuts", 0)),
        ip=float(stat_dict.get("inningsPitched", "0") or "0"),
    )


def get_season_pitching(
    conn: sqlite3.Connection, player_id: int, season: int
) -> PitchingStats:
    """Fetch season-level pitching stats."""
    try:
        data = _mlb_get_json(
            f"/people/{player_id}/stats?stats=season&season={season}&group=pitching",
            conn,
            ttl=43200,
        )
        for group in data.get("stats", []):
            for split in group.get("splits", []):
                return _parse_pitching_stats(split.get("stat", {}))
    except Exception as e:
        log.debug("Failed to fetch pitching stats for player %s: %s", player_id, e)
    return PitchingStats()


def get_prior_season_pitching(
    conn: sqlite3.Connection, player_id: int, current_season: int
) -> PitchingStats:
    """Fetch the prior season's pitching stats for stabilization."""
    return get_season_pitching(conn, player_id, current_season - 1)


# ---------------------------------------------------------------------------
# Boxscore (for backtesting — get actual results)
# ---------------------------------------------------------------------------

def get_boxscore(conn: sqlite3.Connection, game_pk: int) -> dict:
    """Fetch the full boxscore for a completed game."""
    return _mlb_get_json(f"/game/{game_pk}/boxscore", conn, ttl=365 * 86400)


def parse_boxscore_batting(boxscore: dict) -> list[dict]:
    """Extract batting lines from a boxscore.

    Returns list of dicts with player_id, player_name, batting_order,
    stats (H, AB, PA), team, is_home.
    """
    results: list[dict] = []
    for side, is_home in [("away", False), ("home", True)]:
        team_data = boxscore.get("teams", {}).get(side, {})
        team_name = team_data.get("team", {}).get("name", "Unknown")
        players = team_data.get("players", {})

        for key, pdata in players.items():
            batting = pdata.get("stats", {}).get("batting", {})
            order = pdata.get("battingOrder")
            if not order:
                continue  # wasn't in the batting lineup
            slot = int(str(order)[0])  # e.g. "100" → 1, "200" → 2

            results.append({
                "player_id": pdata.get("person", {}).get("id"),
                "player_name": pdata.get("person", {}).get("fullName", "Unknown"),
                "lineup_slot": slot,
                "is_home": is_home,
                "team": team_name,
                "hits": int(batting.get("hits", 0)),
                "ab": int(batting.get("atBats", 0)),
                "pa": int(batting.get("plateAppearances", 0)),
            })
    return results


# ---------------------------------------------------------------------------
# Baseball Savant — xBA leaderboard
# ---------------------------------------------------------------------------

def fetch_savant_xba(conn: sqlite3.Connection, season: int) -> dict[int, dict]:
    """Fetch the Baseball Savant xBA leaderboard for a season.

    Returns a dict keyed by MLBAM player_id with {pa, ba, xba}.
    """
    url = SAVANT_XBA_URL.format(year=season)
    cached = cache.get_cached_response(conn, url)
    if cached:
        text = cached
    else:
        resp = _get(url)
        text = resp.text
        cache.put_cached_response(conn, url, text, ttl=86400)

    result: dict[int, dict] = {}
    # Strip BOM that Baseball Savant includes
    text = text.lstrip("\ufeff")
    reader = csv.DictReader(io.StringIO(text))
    rows_for_cache: list[dict] = []

    for row in reader:
        try:
            player_id = int(row.get("player_id", 0))
            if player_id == 0:
                continue
            pa = int(row.get("pa", 0))
            ba = float(row.get("ba", 0))
            xba = float(row.get("est_ba", 0))
            entry = {"player_id": player_id, "pa": pa, "ba": ba, "xba": xba}
            result[player_id] = entry
            rows_for_cache.append(entry)
        except (ValueError, TypeError):
            continue

    if rows_for_cache:
        cache.put_xba_bulk(conn, season, rows_for_cache)

    return result


def get_player_xba(
    conn: sqlite3.Connection, player_id: int, season: int
) -> dict | None:
    """Get a single player's xBA data from the cache."""
    return cache.get_xba(conn, player_id, season)
