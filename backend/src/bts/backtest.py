"""Backtesting engine for BTS.

Reconstructs point-in-time data for historical dates, runs predictions,
and compares to actual results. Measures calibration and predictive accuracy.

PERFORMANCE: Uses pre-fetched game logs for all stat computation.
A full-season backtest requires ~4500 API calls on first run (to warm the
cache), then runs entirely from local data in seconds.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from bts import cache as cache_mod
from bts import client, factors, probability
from bts.config import ConfidenceTier, RECENCY_WINDOWS
from bts.models import (
    BacktestResult,
    BattingStats,
    FactorBreakdown,
    LineupEntry,
    Prediction,
    StartingPitcher,
)

log = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Game-log based stat computation (local, no API calls)
# ---------------------------------------------------------------------------

def _aggregate_game_log(
    entries: list[dict], before_date: date
) -> BattingStats:
    """Aggregate game log entries into BattingStats for games before a date."""
    stats = BattingStats()
    for e in entries:
        game_date = _parse_date(e["date"])
        if game_date is None or game_date >= before_date:
            continue
        s = e["stat"]
        stats.pa += int(s.get("plateAppearances", 0))
        stats.ab += int(s.get("atBats", 0))
        stats.hits += int(s.get("hits", 0))
        stats.bb += int(s.get("baseOnBalls", 0))
        stats.hbp += int(s.get("hitByPitch", 0))
        stats.k += int(s.get("strikeOuts", 0))
    return stats


def _aggregate_game_log_window(
    entries: list[dict], before_date: date, window_days: int
) -> BattingStats:
    """Aggregate game log entries within a recent window before a date."""
    window_start = before_date - timedelta(days=window_days)
    stats = BattingStats()
    for e in entries:
        game_date = _parse_date(e["date"])
        if game_date is None:
            continue
        if game_date < window_start or game_date >= before_date:
            continue
        s = e["stat"]
        stats.pa += int(s.get("plateAppearances", 0))
        stats.ab += int(s.get("atBats", 0))
        stats.hits += int(s.get("hits", 0))
        stats.bb += int(s.get("baseOnBalls", 0))
        stats.hbp += int(s.get("hitByPitch", 0))
        stats.k += int(s.get("strikeOuts", 0))
    return stats


def _parse_date(s: str) -> date | None:
    try:
        return date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Cache warming — pre-fetch all data for a season
# ---------------------------------------------------------------------------

def warm_cache(conn: sqlite3.Connection, season: int) -> dict[str, int]:
    """Pre-fetch all game logs, boxscores, and player bios for a season.

    Returns counts of items fetched.
    """
    counts = {"games": 0, "boxscores": 0, "players": 0, "pitchers": 0}

    # 1. Fetch every game day in the season
    console.print(f"[bold]Warming cache for {season} season...[/bold]")
    season_start = date(season, 3, 20)  # Spring training / opening day area
    season_end = date(season, 10, 5)    # End of regular season
    current = season_start

    all_game_pks: list[int] = []
    all_player_ids: set[int] = set()
    all_pitcher_ids: set[int] = set()

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        # Phase 1: Collect all games and player IDs from boxscores
        days = (season_end - season_start).days + 1
        task1 = progress.add_task("Fetching schedules + boxscores", total=days)

        while current <= season_end:
            try:
                games = client.get_schedule(conn, current)
                for game in games:
                    if game.status == "Final":
                        all_game_pks.append(game.game_pk)
                        counts["games"] += 1
                        try:
                            boxscore = client.get_boxscore(conn, game.game_pk)
                            counts["boxscores"] += 1
                            batting_lines = client.parse_boxscore_batting(boxscore)
                            for line in batting_lines:
                                if line.get("player_id"):
                                    all_player_ids.add(line["player_id"])
                        except Exception as e:
                            log.debug("Boxscore fetch failed for %s: %s", game.game_pk, e)
            except Exception as e:
                log.debug("Schedule fetch failed for %s: %s", current, e)
            current += timedelta(days=1)
            progress.update(task1, advance=1)

        # Phase 2: Fetch player bios and game logs
        task2 = progress.add_task(
            "Fetching player bios + game logs", total=len(all_player_ids)
        )
        for pid in all_player_ids:
            try:
                bio = client.get_player_bio(conn, pid)
                counts["players"] += 1
            except Exception:
                pass
            try:
                client.get_game_log(conn, pid, season)
            except Exception:
                pass
            # Also fetch prior-season stats for stabilization
            try:
                client.get_season_batting(conn, pid, season - 1)
            except Exception:
                pass
            progress.update(task2, advance=1)

        # Phase 3: Fetch Savant xBA
        progress.add_task("Fetching Savant xBA leaderboard", total=1)
        try:
            client.fetch_savant_xba(conn, season)
        except Exception as e:
            log.warning("Failed to fetch Savant xBA: %s", e)

    console.print(
        f"[green]Cache warmed:[/green] {counts['games']} games, "
        f"{counts['boxscores']} boxscores, {counts['players']} players"
    )
    return counts


# ---------------------------------------------------------------------------
# Backtest runner (game-log based, fast)
# ---------------------------------------------------------------------------

def run_backtest(
    conn: sqlite3.Connection,
    start_date: date,
    end_date: date,
    *,
    recency_weights: list[dict] | None = None,
    disable_platoon: bool = False,
    disable_xba: bool = False,
    disable_park: bool = False,
) -> list[BacktestResult]:
    """Run the model over a date range using pre-cached game logs.

    Optional overrides for ablation studies:
    - recency_weights: override RECENCY_WINDOWS config
    - disable_platoon/xba/park: set factor to 1.0 for marginal lift testing
    """
    season = start_date.year
    effective_windows = recency_weights or RECENCY_WINDOWS

    # Pre-load Savant xBA for the season (one call, likely cached)
    xba_data = client.fetch_savant_xba(conn, season)

    # Cache of game logs: player_id → list of game entries
    game_log_cache: dict[int, list[dict]] = {}

    # Cache of prior-season H/PA: player_id → float|None
    prior_hpa_cache: dict[int, float | None] = {}

    # Cache of pitcher season stats (full season, not point-in-time for now)
    pitcher_season_cache: dict[int, Any] = {}
    pitcher_prior_cache: dict[int, Any] = {}

    results: list[BacktestResult] = []
    current = start_date
    total_days = (end_date - start_date).days + 1

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Backtesting", total=total_days)

        while current <= end_date:
            day_results = _backtest_date_fast(
                conn, current, season, xba_data,
                game_log_cache, prior_hpa_cache,
                pitcher_season_cache, pitcher_prior_cache,
                effective_windows,
                disable_platoon, disable_xba, disable_park,
            )
            results.extend(day_results)
            current += timedelta(days=1)
            progress.update(task, advance=1)

    return results


def _backtest_date_fast(
    conn: sqlite3.Connection,
    game_date: date,
    season: int,
    xba_data: dict[int, dict],
    game_log_cache: dict[int, list[dict]],
    prior_hpa_cache: dict[int, float | None],
    pitcher_season_cache: dict,
    pitcher_prior_cache: dict,
    effective_windows: list[dict],
    disable_platoon: bool,
    disable_xba: bool,
    disable_park: bool,
) -> list[BacktestResult]:
    """Backtest a single date using game-log local computation."""
    games = client.get_schedule(conn, game_date)
    results: list[BacktestResult] = []

    for game in games:
        if game.status != "Final":
            continue

        # Get actual boxscore (lineups + results)
        try:
            boxscore = client.get_boxscore(conn, game.game_pk)
        except Exception:
            continue

        batting_lines = client.parse_boxscore_batting(boxscore)
        if not batting_lines:
            continue

        # Build actual results lookup
        actuals: dict[int, dict] = {
            line["player_id"]: line for line in batting_lines
        }

        # Get starting pitchers from boxscore
        pitchers = _extract_starters_from_boxscore(conn, boxscore)

        # Map: is_home → opposing SP
        opposing_sp: dict[bool, StartingPitcher | None] = {True: None, False: None}
        for sp in pitchers:
            opposing_sp[not sp.is_home] = sp

        # Predict each batter using game logs
        for line in batting_lines:
            player_id = line["player_id"]
            if player_id is None or line["pa"] == 0:
                continue

            pred = _predict_batter_from_logs(
                conn, player_id, line, game, season, xba_data,
                game_log_cache, prior_hpa_cache,
                pitcher_season_cache, pitcher_prior_cache,
                opposing_sp.get(line["is_home"]),
                effective_windows,
                disable_platoon, disable_xba, disable_park,
            )

            if pred is None:
                continue
            if pred.confidence == ConfidenceTier.UNPROVEN:
                continue

            results.append(BacktestResult(
                prediction=pred,
                actual_got_hit=line["hits"] > 0,
                actual_hits=line["hits"],
                actual_ab=line["ab"],
                actual_pa=line["pa"],
            ))

    return results


def _predict_batter_from_logs(
    conn: sqlite3.Connection,
    player_id: int,
    line: dict,
    game: Any,
    season: int,
    xba_data: dict[int, dict],
    game_log_cache: dict[int, list[dict]],
    prior_hpa_cache: dict[int, float | None],
    pitcher_season_cache: dict,
    pitcher_prior_cache: dict,
    opp_sp: StartingPitcher | None,
    effective_windows: list[dict],
    disable_platoon: bool,
    disable_xba: bool,
    disable_park: bool,
) -> Prediction | None:
    """Predict a single batter using game-log local computation."""

    # --- Get or cache game log ---
    if player_id not in game_log_cache:
        game_log_cache[player_id] = client.get_game_log(conn, player_id, season)
    game_log = game_log_cache[player_id]

    # --- Season stats through yesterday (local computation) ---
    season_stats = _aggregate_game_log(game_log, game.game_date)
    if season_stats.pa == 0:
        # Check prior season
        if player_id not in prior_hpa_cache:
            prior = client.get_season_batting(conn, player_id, season - 1)
            prior_hpa_cache[player_id] = prior.hpa if prior.pa > 0 else None
        if prior_hpa_cache[player_id] is None:
            return None

    # --- Prior-season H/PA ---
    if player_id not in prior_hpa_cache:
        prior = client.get_season_batting(conn, player_id, season - 1)
        prior_hpa_cache[player_id] = prior.hpa if prior.pa > 0 else None
    prior_hpa = prior_hpa_cache[player_id]

    # --- Recent-window stats (local computation) ---
    recent_stats: dict[int, BattingStats] = {}
    for window in effective_windows:
        days = window["days"]
        recent_stats[days] = _aggregate_game_log_window(
            game_log, game.game_date, days
        )

    # --- Blended batter H/PA ---
    batter_hpa = factors.blended_batter_hpa(
        season_stats, recent_stats, prior_hpa,
        recency_windows=effective_windows,
    )
    if batter_hpa <= 0:
        return None

    # --- Confidence tier ---
    confidence = factors.determine_confidence(season_stats.pa, prior_hpa is not None)

    # --- Player bio ---
    bio = client.get_player_bio(conn, player_id)
    if bio is None:
        return None

    # --- Opposing pitcher ---
    pitcher_hpa_allowed = 0.250
    pitcher_name = "TBD"
    pitcher_hand = "R"

    if opp_sp:
        pitcher_name = opp_sp.player_name
        pitcher_hand = opp_sp.pitch_hand

        if opp_sp.player_id not in pitcher_season_cache:
            pitcher_season_cache[opp_sp.player_id] = client.get_season_pitching(
                conn, opp_sp.player_id, season
            )
        if opp_sp.player_id not in pitcher_prior_cache:
            prior_p = client.get_season_pitching(conn, opp_sp.player_id, season - 1)
            pitcher_prior_cache[opp_sp.player_id] = (
                prior_p.hpa_allowed if prior_p.pa_faced > 0 else None
            )

        pitcher_hpa_allowed = factors.stabilized_pitcher_hpa(
            pitcher_season_cache[opp_sp.player_id],
            pitcher_prior_cache[opp_sp.player_id],
        )

    # --- Platoon ---
    platoon_mult = 1.0
    if not disable_platoon:
        split_key = "vl" if pitcher_hand == "L" else "vr"
        current_splits = client.get_platoon_splits(conn, player_id, season)
        split_stats = current_splits.get(split_key)
        split_hpa = split_stats.hpa if split_stats and split_stats.pa > 0 else None
        split_pa = split_stats.pa if split_stats else 0

        career_splits = client.get_career_platoon_splits(conn, player_id)
        career_split = career_splits.get(split_key)
        career_split_hpa = career_split.hpa if career_split and career_split.pa > 0 else None
        career_split_pa = career_split.pa if career_split else 0

        overall = season_stats.hpa if season_stats.pa > 0 else (prior_hpa or batter_hpa)
        platoon_mult = factors.platoon_multiplier(
            overall, split_hpa, career_split_hpa, split_pa, career_split_pa,
        )

    # --- xBA ---
    xba_adj = 1.0
    if not disable_xba:
        player_xba = xba_data.get(player_id)
        xba_adj = factors.xba_adjustment(
            xba=player_xba["xba"] if player_xba else None,
            actual_ba=player_xba["ba"] if player_xba else None,
        )

    # --- Park factor ---
    pf = 1.0 if disable_park else factors.park_factor(game.venue_name)

    # --- Full prediction ---
    prob, breakdown = probability.full_prediction(
        batter_hpa=batter_hpa,
        pitcher_hpa_allowed=pitcher_hpa_allowed,
        platoon_mult=platoon_mult,
        xba_adj=xba_adj,
        park_factor=pf,
        lineup_slot=line["lineup_slot"],
        is_home=line["is_home"],
        pitcher_name=pitcher_name,
        pitcher_hand=pitcher_hand,
    )

    return Prediction(
        player_id=player_id,
        player_name=line["player_name"],
        team=line["team"],
        game_pk=game.game_pk,
        game_date=game.game_date,
        hit_probability=prob,
        confidence=confidence,
        factors=breakdown,
        current_season_pa=season_stats.pa,
        lineup_slot=line["lineup_slot"],
        is_home=line["is_home"],
    )


def _extract_starters_from_boxscore(
    conn: sqlite3.Connection, boxscore: dict
) -> list[StartingPitcher]:
    """Extract starting pitchers from a boxscore."""
    pitchers: list[StartingPitcher] = []
    for side, is_home in [("away", False), ("home", True)]:
        team_data = boxscore.get("teams", {}).get(side, {})
        team_name = team_data.get("team", {}).get("name", "Unknown")
        pitcher_ids = team_data.get("pitchers", [])
        if not pitcher_ids:
            continue
        # First pitcher listed is the starter
        starter_id = pitcher_ids[0]
        bio = client.get_pitcher_bio(conn, starter_id)
        if bio:
            pitchers.append(StartingPitcher(
                player_id=starter_id,
                player_name=bio.full_name,
                pitch_hand=bio.pitch_hand,
                is_home=is_home,
                team=team_name,
            ))
    return pitchers


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class BacktestMetrics:
    """Computed metrics from a backtest run."""

    total_predictions: int = 0
    brier_score: float = 0.0
    top1_hit_rate: float = 0.0
    top5_hit_rate: float = 0.0
    top10_hit_rate: float = 0.0
    calibration_bins: list[dict] = field(default_factory=list)
    streak_analysis: dict = field(default_factory=dict)


def compute_metrics(results: list[BacktestResult]) -> BacktestMetrics:
    """Compute calibration and accuracy metrics from backtest results."""
    if not results:
        return BacktestMetrics()

    metrics = BacktestMetrics(total_predictions=len(results))

    # Brier score: mean of (predicted - actual)²
    brier_sum = sum(
        (r.prediction.hit_probability - (1.0 if r.actual_got_hit else 0.0)) ** 2
        for r in results
    )
    metrics.brier_score = brier_sum / len(results)

    # Top-N accuracy by date
    by_date: dict[date, list[BacktestResult]] = defaultdict(list)
    for r in results:
        by_date[r.prediction.game_date].append(r)

    top1_hits = 0
    top5_hits = 0
    top5_total = 0
    top10_hits = 0
    top10_total = 0
    dates_with_data = 0

    # Streak simulation: pick top-1 each day
    streak_lengths: list[int] = []
    current_streak = 0

    for d in sorted(by_date.keys()):
        day_results = sorted(
            by_date[d], key=lambda r: r.prediction.hit_probability, reverse=True
        )
        if not day_results:
            continue

        dates_with_data += 1

        # Top-1
        if day_results[0].actual_got_hit:
            top1_hits += 1
            current_streak += 1
        else:
            if current_streak > 0:
                streak_lengths.append(current_streak)
            current_streak = 0

        # Top-5
        for r in day_results[:5]:
            top5_total += 1
            if r.actual_got_hit:
                top5_hits += 1

        # Top-10
        for r in day_results[:10]:
            top10_total += 1
            if r.actual_got_hit:
                top10_hits += 1

    if current_streak > 0:
        streak_lengths.append(current_streak)

    metrics.top1_hit_rate = top1_hits / dates_with_data if dates_with_data > 0 else 0.0
    metrics.top5_hit_rate = top5_hits / top5_total if top5_total > 0 else 0.0
    metrics.top10_hit_rate = top10_hits / top10_total if top10_total > 0 else 0.0

    metrics.streak_analysis = {
        "total_days": dates_with_data,
        "longest_streak": max(streak_lengths) if streak_lengths else 0,
        "avg_streak": (
            sum(streak_lengths) / len(streak_lengths) if streak_lengths else 0.0
        ),
        "streak_count": len(streak_lengths),
        "streaks": streak_lengths,
    }

    # Calibration bins (deciles)
    metrics.calibration_bins = _calibration_bins(results)

    return metrics


def _calibration_bins(results: list[BacktestResult], n_bins: int = 10) -> list[dict]:
    """Compute calibration bins."""
    sorted_results = sorted(results, key=lambda r: r.prediction.hit_probability)
    bin_size = max(1, len(sorted_results) // n_bins)
    bins: list[dict] = []

    for i in range(0, len(sorted_results), bin_size):
        chunk = sorted_results[i : i + bin_size]
        if not chunk:
            continue
        avg_predicted = sum(r.prediction.hit_probability for r in chunk) / len(chunk)
        actual_rate = sum(1 for r in chunk if r.actual_got_hit) / len(chunk)
        bins.append({
            "predicted_avg": round(avg_predicted, 3),
            "actual_rate": round(actual_rate, 3),
            "count": len(chunk),
            "range_low": round(chunk[0].prediction.hit_probability, 3),
            "range_high": round(chunk[-1].prediction.hit_probability, 3),
        })

    return bins


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_backtest_report(
    metrics: BacktestMetrics,
    start_date: date,
    end_date: date,
    *,
    label: str = "",
) -> None:
    """Print a formatted backtest report."""
    console.print()
    title = f"Backtest Report: {start_date.isoformat()} → {end_date.isoformat()}"
    if label:
        title += f"  ({label})"
    console.print(f"[bold]{title}[/bold]", justify="center")
    console.print()

    # Summary
    console.print("[bold]Summary[/bold]")
    console.print(f"  Total predictions:  {metrics.total_predictions}")
    console.print(f"  Brier score:        {metrics.brier_score:.4f}")
    console.print()

    # Top-N accuracy
    console.print("[bold]Pick Accuracy[/bold]")
    console.print(f"  Top-1 daily hit rate:   {metrics.top1_hit_rate:.1%}")
    console.print(f"  Top-5 avg hit rate:     {metrics.top5_hit_rate:.1%}")
    console.print(f"  Top-10 avg hit rate:    {metrics.top10_hit_rate:.1%}")
    console.print()

    # Streak simulation
    sa = metrics.streak_analysis
    if sa:
        console.print("[bold]Streak Simulation (top-1 pick daily)[/bold]")
        console.print(f"  Days simulated:   {sa['total_days']}")
        console.print(f"  Longest streak:   {sa['longest_streak']}")
        console.print(f"  Average streak:   {sa['avg_streak']:.1f}")
        console.print(f"  Number of streaks: {sa['streak_count']}")
        if sa.get("streaks"):
            top_streaks = sorted(sa["streaks"], reverse=True)[:5]
            console.print(f"  Top 5 streaks:    {top_streaks}")
        console.print()

    # Calibration table
    if metrics.calibration_bins:
        console.print("[bold]Calibration[/bold]")
        cal_table = Table(show_header=True, header_style="bold", border_style="dim")
        cal_table.add_column("Predicted Range", width=16)
        cal_table.add_column("Avg Predicted", justify="right", width=14)
        cal_table.add_column("Actual Hit Rate", justify="right", width=14)
        cal_table.add_column("Δ", justify="right", width=8)
        cal_table.add_column("Count", justify="right", width=6)

        for b in metrics.calibration_bins:
            delta = b["actual_rate"] - b["predicted_avg"]
            delta_str = f"{delta:+.3f}"
            delta_color = (
                "green" if abs(delta) < 0.05
                else "yellow" if abs(delta) < 0.10
                else "red"
            )
            cal_table.add_row(
                f"{b['range_low']:.3f}–{b['range_high']:.3f}",
                f"{b['predicted_avg']:.3f}",
                f"{b['actual_rate']:.3f}",
                f"[{delta_color}]{delta_str}[/{delta_color}]",
                str(b["count"]),
            )

        console.print(cal_table)
        console.print()
