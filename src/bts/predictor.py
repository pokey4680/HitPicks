"""Main prediction pipeline: fetch data → compute → rank."""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, timedelta

from bts import cache, client, factors, probability
from bts.config import ConfidenceTier, PLATOON_ENABLED, RECENCY_WINDOWS
from bts.models import (
    BattingStats,
    GameInfo,
    LineupEntry,
    Prediction,
    StartingPitcher,
)

log = logging.getLogger(__name__)


def predict_for_date(
    conn: sqlite3.Connection,
    game_date: date,
    *,
    include_unproven: bool = False,
) -> list[Prediction]:
    """Run predictions for all batters in all games on a given date.

    Returns predictions sorted by hit_probability descending.
    """
    games = client.get_schedule(conn, game_date)
    if not games:
        log.warning("No games found for %s", game_date)
        return []

    # Pre-fetch the Savant xBA leaderboard for the season (one call)
    season = game_date.year
    xba_data = client.fetch_savant_xba(conn, season)

    all_predictions: list[Prediction] = []

    for game in games:
        if game.status not in ("Preview", "Pre-Game", "Warmup", "Scheduled", "Final"):
            # Skip games already in progress (lineups locked but too late to pick)
            # "Final" is included for backtesting
            if game.status != "In Progress":
                continue

        preds = _predict_game(conn, game, season, xba_data, include_unproven)
        all_predictions.extend(preds)

    all_predictions.sort(key=lambda p: p.hit_probability, reverse=True)
    return all_predictions


def _predict_game(
    conn: sqlite3.Connection,
    game: GameInfo,
    season: int,
    xba_data: dict[int, dict],
    include_unproven: bool,
) -> list[Prediction]:
    """Generate predictions for all batters in a single game."""
    lineups, pitchers = client.get_lineups_and_pitchers(
        conn, game.game_pk, game.game_date
    )

    if not lineups:
        log.debug("No lineup data for game %s", game.game_pk)
        return []

    # Map: is_home → StartingPitcher for the OPPOSING team
    # Away batters face the home pitcher, home batters face the away pitcher
    opposing_sp: dict[bool, StartingPitcher | None] = {True: None, False: None}
    for sp in pitchers:
        # If pitcher is home, they oppose away batters (is_home=False)
        opposing_sp[not sp.is_home] = sp

    predictions: list[Prediction] = []

    for entry in lineups:
        pred = _predict_batter(
            conn, entry, game, season, opposing_sp.get(entry.is_home), xba_data
        )
        if pred is None:
            continue
        if not include_unproven and pred.confidence == ConfidenceTier.UNPROVEN:
            continue
        predictions.append(pred)

    return predictions


def _predict_batter(
    conn: sqlite3.Connection,
    entry: LineupEntry,
    game: GameInfo,
    season: int,
    opp_sp: StartingPitcher | None,
    xba_data: dict[int, dict],
) -> Prediction | None:
    """Compute the full prediction for a single batter."""
    player_id = entry.player_id

    # --- Player bio (handedness) ---
    bio = client.get_player_bio(conn, player_id)
    if bio is None:
        return None

    # --- Current season stats ---
    season_stats = client.get_season_batting(conn, player_id, season)

    # --- Prior-season stats (for stabilization) ---
    prior_stats = client.get_season_batting(conn, player_id, season - 1)
    prior_hpa = prior_stats.hpa if prior_stats.pa > 0 else None

    # --- Recent-window stats (multiple windows) ---
    # End date is the day before the game (what was known that morning)
    end_date = game.game_date - timedelta(days=1)
    recent_stats: dict[int, BattingStats] = {}
    for window in RECENCY_WINDOWS:
        days = window["days"]
        stats = client.get_recent_batting(conn, player_id, end_date, days)
        recent_stats[days] = stats

    # --- Blended batter H/PA ---
    batter_hpa = factors.blended_batter_hpa(season_stats, recent_stats, prior_hpa)
    if batter_hpa <= 0:
        return None

    # --- Confidence tier ---
    confidence = factors.determine_confidence(
        season_stats.pa, prior_hpa is not None
    )

    # --- Opposing pitcher data ---
    pitcher_hpa_allowed = 0.250  # neutral fallback if no SP data
    pitcher_name = "TBD"
    pitcher_hand = "R"

    if opp_sp:
        pitcher_name = opp_sp.player_name
        pitcher_hand = opp_sp.pitch_hand

        pitch_season = client.get_season_pitching(conn, opp_sp.player_id, season)
        pitch_prior = client.get_season_pitching(conn, opp_sp.player_id, season - 1)
        prior_pitch_hpa = pitch_prior.hpa_allowed if pitch_prior.pa_faced > 0 else None

        pitcher_hpa_allowed = factors.stabilized_pitcher_hpa(
            pitch_season, prior_pitch_hpa
        )

    # --- Platoon split ---
    platoon_mult = 1.0
    if PLATOON_ENABLED:
        split_key = "vl" if pitcher_hand == "L" else "vr"
        current_splits = client.get_platoon_splits(conn, player_id, season)
        split_stats = current_splits.get(split_key)
        split_hpa = split_stats.hpa if split_stats and split_stats.pa > 0 else None
        split_pa = split_stats.pa if split_stats else 0

        career_splits = client.get_career_platoon_splits(conn, player_id)
        career_split = career_splits.get(split_key)
        career_split_hpa = career_split.hpa if career_split and career_split.pa > 0 else None
        career_split_pa = career_split.pa if career_split else 0

        platoon_mult = factors.platoon_multiplier(
            overall_hpa=season_stats.hpa if season_stats.pa > 0 else (prior_hpa or batter_hpa),
            split_hpa=split_hpa,
            career_split_hpa=career_split_hpa,
            split_pa=split_pa,
            career_split_pa=career_split_pa,
        )

    # --- xBA adjustment ---
    player_xba = xba_data.get(player_id)
    xba_adj = factors.xba_adjustment(
        xba=player_xba["xba"] if player_xba else None,
        actual_ba=player_xba["ba"] if player_xba else None,
    )

    # --- Park factor ---
    pf = factors.park_factor(game.venue_name)

    # --- Full prediction ---
    prob, breakdown = probability.full_prediction(
        batter_hpa=batter_hpa,
        pitcher_hpa_allowed=pitcher_hpa_allowed,
        platoon_mult=platoon_mult,
        xba_adj=xba_adj,
        park_factor=pf,
        lineup_slot=entry.lineup_slot,
        is_home=entry.is_home,
        pitcher_name=pitcher_name,
        pitcher_hand=pitcher_hand,
    )

    return Prediction(
        player_id=player_id,
        player_name=entry.player_name,
        team=entry.team,
        game_pk=game.game_pk,
        game_date=game.game_date,
        hit_probability=prob,
        confidence=confidence,
        factors=breakdown,
        current_season_pa=season_stats.pa,
        lineup_slot=entry.lineup_slot,
        is_home=entry.is_home,
    )
