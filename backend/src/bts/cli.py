"""CLI entry point for BTS."""

from __future__ import annotations

import logging
from datetime import date, datetime

import click
from rich.console import Console

from bts import cache as cache_mod
from bts import strategy as strategy_mod
from bts.backtest import compute_metrics, print_backtest_report, run_backtest, warm_cache
from bts.predictor import predict_for_date
from bts.report import render_predictions

console = Console()


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def main(debug: bool) -> None:
    """BTS — Beat the Streak hit probability predictor."""
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


@main.command()
@click.option(
    "--date", "game_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Date to predict (YYYY-MM-DD). Defaults to today.",
)
@click.option("--top", "top_n", type=int, default=25, help="Show top N predictions.")
@click.option(
    "--min-prob", type=float, default=None,
    help="Only show batters above this probability (e.g. 0.75).",
)
@click.option(
    "--all-tiers", is_flag=True, default=False,
    help="Include unproven players (default: excluded).",
)
def predict(
    game_date: datetime | None,
    top_n: int,
    min_prob: float | None,
    all_tiers: bool,
) -> None:
    """Generate hit probability predictions for a date."""
    d = game_date.date() if game_date else date.today()

    conn = cache_mod.get_connection()
    try:
        with console.status(f"[bold]Fetching data for {d.isoformat()}..."):
            predictions = predict_for_date(
                conn, d, include_unproven=all_tiers
            )

        if not predictions:
            console.print(f"[yellow]No predictions available for {d.isoformat()}.[/yellow]")
            console.print("[dim]Lineups may not be posted yet. Try again closer to game time.[/dim]")
            return

        render_predictions(
            predictions, d, top_n=top_n, min_prob=min_prob, console=console
        )

    finally:
        conn.close()


@main.command()
@click.option(
    "--start", "start_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=True,
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--end", "end_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=True,
    help="End date (YYYY-MM-DD).",
)
def backtest(start_date: datetime, end_date: datetime) -> None:
    """Backtest the model against historical results."""
    sd = start_date.date()
    ed = end_date.date()

    console.print(f"[bold]Running backtest: {sd.isoformat()} → {ed.isoformat()}[/bold]")
    console.print()

    conn = cache_mod.get_connection()
    try:
        results = run_backtest(conn, sd, ed)

        if not results:
            console.print("[yellow]No results — games may not be Final yet.[/yellow]")
            return

        metrics = compute_metrics(results)
        print_backtest_report(metrics, sd, ed)

    finally:
        conn.close()


@main.command()
@click.option(
    "--window", "windows",
    multiple=True,
    required=True,
    help="Date window as YYYY-MM-DD:YYYY-MM-DD. Repeat to pool multiple windows.",
)
@click.option(
    "--season-days", type=int, default=180,
    help="Season length for Monte Carlo simulation (default: 180).",
)
@click.option(
    "--target", "target_streak", type=int, default=57,
    help="Target streak length for P(target) estimate (default: 57).",
)
@click.option(
    "--iterations", "n_mc", type=int, default=10000,
    help="Number of Monte Carlo seasons to simulate (default: 10000).",
)
def strategy(
    windows: tuple[str, ...],
    season_days: int,
    target_streak: int,
    n_mc: int,
) -> None:
    """Compare day-by-day picking strategies over backtest windows."""
    parsed_windows: list[tuple[date, date]] = []
    for w in windows:
        try:
            s, e = w.split(":")
            parsed_windows.append((date.fromisoformat(s), date.fromisoformat(e)))
        except ValueError as exc:
            raise click.BadParameter(
                f"Invalid window '{w}'. Expected YYYY-MM-DD:YYYY-MM-DD."
            ) from exc

    conn = cache_mod.get_connection()
    try:
        all_results = []
        for sd, ed in parsed_windows:
            console.print(f"[bold]Backtesting {sd.isoformat()} → {ed.isoformat()}[/bold]")
            all_results.extend(run_backtest(conn, sd, ed))

        if not all_results:
            console.print("[yellow]No backtest results — cache may be empty.[/yellow]")
            return

        console.print(
            f"[dim]{len(all_results)} total predictions across "
            f"{len(parsed_windows)} window(s).[/dim]"
        )
        console.print()

        window_label = " + ".join(
            f"{sd.isoformat()}→{ed.isoformat()}" for sd, ed in parsed_windows
        )

        strategy_results = []
        for name, strat in strategy_mod.default_suite():
            sr = strategy_mod.simulate(
                all_results,
                strat,
                name=name,
                season_days=season_days,
                n_mc=n_mc,
                target_streak=target_streak,
            )
            strategy_results.append(sr)

        strategy_mod.print_strategy_report(
            strategy_results,
            season_days=season_days,
            target_streak=target_streak,
            window_label=window_label,
        )

    finally:
        conn.close()


@main.group()
def cache() -> None:
    """Manage the local data cache."""


@cache.command("warm")
@click.option("--season", type=int, required=True, help="Season year to pre-fetch.")
def cache_warm(season: int) -> None:
    """Pre-fetch all game logs, boxscores, and bios for a season."""
    conn = cache_mod.get_connection()
    try:
        warm_cache(conn, season)
    finally:
        conn.close()


@cache.command("status")
def cache_status() -> None:
    """Show cache statistics."""
    conn = cache_mod.get_connection()
    try:
        stats = cache_mod.cache_stats(conn)
        console.print("[bold]Cache Status[/bold]")
        console.print(f"  API responses: {stats['api_responses']}")
        console.print(f"  Player bios:   {stats['player_bios']}")
        console.print(f"  Pitcher bios:  {stats['pitcher_bios']}")
        console.print(f"  xBA entries:   {stats['xba_entries']}")
        console.print(f"  Database size: {stats['db_size_mb']} MB")
    finally:
        conn.close()


@cache.command("clear")
@click.confirmation_option(prompt="Delete all cached data?")
def cache_clear() -> None:
    """Delete all cached data."""
    conn = cache_mod.get_connection()
    try:
        cache_mod.clear_cache(conn)
        console.print("[green]Cache cleared.[/green]")
    finally:
        conn.close()
