"""SQLite-backed cache for API responses and parsed data."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from bts.config import CACHE_DB, CACHE_DIR


def _ensure_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Return a connection to the cache database, creating tables if needed."""
    _ensure_dir()
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS api_cache (
            url           TEXT PRIMARY KEY,
            response_text TEXT NOT NULL,
            fetched_at    REAL NOT NULL,
            ttl_seconds   INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS player_bio (
            player_id        INTEGER PRIMARY KEY,
            full_name        TEXT NOT NULL,
            bat_side         TEXT NOT NULL,
            primary_position TEXT NOT NULL,
            fetched_at       REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pitcher_bio (
            player_id  INTEGER PRIMARY KEY,
            full_name  TEXT NOT NULL,
            pitch_hand TEXT NOT NULL,
            fetched_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS savant_xba (
            player_id  INTEGER,
            season     INTEGER,
            pa         INTEGER,
            ba         REAL,
            xba        REAL,
            fetched_at REAL NOT NULL,
            PRIMARY KEY (player_id, season)
        );
    """)


# ---------------------------------------------------------------------------
# Generic HTTP response cache
# ---------------------------------------------------------------------------

def get_cached_response(conn: sqlite3.Connection, url: str) -> str | None:
    """Return cached response text if present and not expired, else None."""
    row = conn.execute(
        "SELECT response_text, fetched_at, ttl_seconds FROM api_cache WHERE url = ?",
        (url,),
    ).fetchone()
    if row is None:
        return None
    response_text, fetched_at, ttl = row
    if time.time() - fetched_at > ttl:
        return None  # expired
    return response_text


def put_cached_response(
    conn: sqlite3.Connection, url: str, response_text: str, ttl: int
) -> None:
    """Store an API response in the cache."""
    conn.execute(
        """INSERT OR REPLACE INTO api_cache (url, response_text, fetched_at, ttl_seconds)
           VALUES (?, ?, ?, ?)""",
        (url, response_text, time.time(), ttl),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Player bio cache
# ---------------------------------------------------------------------------

def get_player_bio(conn: sqlite3.Connection, player_id: int) -> dict | None:
    """Return cached player bio dict or None."""
    row = conn.execute(
        "SELECT full_name, bat_side, primary_position FROM player_bio WHERE player_id = ?",
        (player_id,),
    ).fetchone()
    if row is None:
        return None
    return {"full_name": row[0], "bat_side": row[1], "primary_position": row[2]}


def put_player_bio(
    conn: sqlite3.Connection,
    player_id: int,
    full_name: str,
    bat_side: str,
    primary_position: str,
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO player_bio
           (player_id, full_name, bat_side, primary_position, fetched_at)
           VALUES (?, ?, ?, ?, ?)""",
        (player_id, full_name, bat_side, primary_position, time.time()),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Pitcher bio cache
# ---------------------------------------------------------------------------

def get_pitcher_bio(conn: sqlite3.Connection, player_id: int) -> dict | None:
    row = conn.execute(
        "SELECT full_name, pitch_hand FROM pitcher_bio WHERE player_id = ?",
        (player_id,),
    ).fetchone()
    if row is None:
        return None
    return {"full_name": row[0], "pitch_hand": row[1]}


def put_pitcher_bio(
    conn: sqlite3.Connection, player_id: int, full_name: str, pitch_hand: str
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO pitcher_bio
           (player_id, full_name, pitch_hand, fetched_at)
           VALUES (?, ?, ?, ?)""",
        (player_id, full_name, pitch_hand, time.time()),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Savant xBA cache
# ---------------------------------------------------------------------------

def get_xba(conn: sqlite3.Connection, player_id: int, season: int) -> dict | None:
    row = conn.execute(
        "SELECT pa, ba, xba FROM savant_xba WHERE player_id = ? AND season = ?",
        (player_id, season),
    ).fetchone()
    if row is None:
        return None
    return {"pa": row[0], "ba": row[1], "xba": row[2]}


def put_xba_bulk(
    conn: sqlite3.Connection, season: int, rows: list[dict]
) -> None:
    """Insert/update xBA data for an entire season leaderboard."""
    conn.executemany(
        """INSERT OR REPLACE INTO savant_xba
           (player_id, season, pa, ba, xba, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [(r["player_id"], season, r["pa"], r["ba"], r["xba"], time.time()) for r in rows],
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

def cache_stats(conn: sqlite3.Connection) -> dict:
    """Return cache statistics."""
    api_count = conn.execute("SELECT COUNT(*) FROM api_cache").fetchone()[0]
    player_count = conn.execute("SELECT COUNT(*) FROM player_bio").fetchone()[0]
    pitcher_count = conn.execute("SELECT COUNT(*) FROM pitcher_bio").fetchone()[0]
    xba_count = conn.execute("SELECT COUNT(*) FROM savant_xba").fetchone()[0]
    db_size = CACHE_DB.stat().st_size if CACHE_DB.exists() else 0
    return {
        "api_responses": api_count,
        "player_bios": player_count,
        "pitcher_bios": pitcher_count,
        "xba_entries": xba_count,
        "db_size_mb": round(db_size / 1_048_576, 2),
    }


def clear_cache(conn: sqlite3.Connection) -> None:
    """Delete all cached data."""
    for table in ("api_cache", "player_bio", "pitcher_bio", "savant_xba"):
        conn.execute(f"DELETE FROM {table}")
    conn.commit()
    conn.execute("VACUUM")
