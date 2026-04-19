"""Rich terminal output for BTS predictions."""

from __future__ import annotations

from datetime import date, datetime

from rich.console import Console
from rich.table import Table
from rich.text import Text

from bts.config import ConfidenceTier
from bts.models import Prediction


def render_predictions(
    predictions: list[Prediction],
    game_date: date,
    *,
    top_n: int | None = None,
    min_prob: float | None = None,
    console: Console | None = None,
) -> None:
    """Render the prediction leaderboard as a Rich table."""
    console = console or Console()

    # Filter
    filtered = predictions
    if min_prob is not None:
        filtered = [p for p in filtered if p.hit_probability >= min_prob]
    if top_n is not None:
        filtered = filtered[:top_n]

    if not filtered:
        console.print("[yellow]No predictions meet the criteria.[/yellow]")
        return

    # Header
    console.print()
    console.print(
        f"[bold]BTS Predictions for {game_date.isoformat()}[/bold]",
        justify="center",
    )
    console.print(
        f"[dim]Generated {datetime.now().strftime('%I:%M %p ET')} · "
        f"Model: P(≥1 hit) = 1-(1-p)^n[/dim]",
        justify="center",
    )
    console.print()

    # Legend
    console.print(
        f"  [bold]{ConfidenceTier.ESTABLISHED.symbol}[/bold] Established (200+ PA)  "
        f"  [bold]{ConfidenceTier.PARTTIME.symbol}[/bold] Part-Time (50-199 PA)  "
        f"  [dim]{ConfidenceTier.UNPROVEN.symbol} Unproven (<50 PA)[/dim]"
    )
    console.print()

    # Table
    table = Table(show_header=True, header_style="bold", border_style="dim")
    table.add_column("#", justify="right", width=3)
    table.add_column("", width=1)  # confidence tier symbol
    table.add_column("Player", width=20)
    table.add_column("Team", width=5)
    table.add_column("P(Hit)", justify="right", width=7)
    table.add_column("Matchup", justify="right", width=7)
    table.add_column("Platoon", justify="right", width=7)
    table.add_column("xBA", justify="right", width=6)
    table.add_column("Park", justify="right", width=6)
    table.add_column("vs SP", width=22)
    table.add_column("PAs", justify="right", width=4)
    table.add_column("PA", justify="right", width=4, style="dim")

    for rank, pred in enumerate(filtered, 1):
        f = pred.factors

        # Color-code the probability
        prob_str = f".{pred.hit_probability * 1000:.0f}"[:4]
        if pred.hit_probability >= 0.80:
            prob_style = "bold green"
        elif pred.hit_probability >= 0.70:
            prob_style = "green"
        elif pred.hit_probability >= 0.60:
            prob_style = "yellow"
        else:
            prob_style = "red"

        # Signed percentage strings for adjustments
        def _signed_pct(val: float) -> str:
            pct = (val - 1.0) * 100
            if abs(pct) < 0.5:
                return "[dim]—[/dim]"
            sign = "+" if pct > 0 else ""
            color = "green" if pct > 0 else "red"
            return f"[{color}]{sign}{pct:.0f}%[/{color}]"

        # Pitcher column — last name + hand + H/PA
        if f.pitcher_name != "TBD":
            last_name = f.pitcher_name.split()[-1][:10]
            pitcher_str = f"{last_name} ({f.pitcher_hand}) {f.pitcher_hpa_allowed:.3f}"
        else:
            pitcher_str = "[dim]TBD[/dim]"

        # Confidence tier
        tier_style = {
            ConfidenceTier.ESTABLISHED: "bold",
            ConfidenceTier.PARTTIME: "",
            ConfidenceTier.UNPROVEN: "dim",
        }[pred.confidence]

        table.add_row(
            str(rank),
            f"[{tier_style}]{pred.confidence.symbol}[/{tier_style}]",
            pred.player_name[:20],
            _team_abbrev(pred.team),
            f"[{prob_style}]{prob_str}[/{prob_style}]",
            f".{f.matchup_hpa:.3f}"[1:],
            _signed_pct(f.platoon_mult),
            _signed_pct(f.xba_adj),
            _signed_pct(f.park_factor),
            pitcher_str,
            f"{f.expected_pa:.1f}",
            str(pred.current_season_pa),
        )

    console.print(table)
    console.print()


def _team_abbrev(team_name: str) -> str:
    """Convert full team name to abbreviation."""
    abbrevs = {
        "Arizona Diamondbacks": "ARI",
        "Atlanta Braves": "ATL",
        "Baltimore Orioles": "BAL",
        "Boston Red Sox": "BOS",
        "Chicago Cubs": "CHC",
        "Chicago White Sox": "CWS",
        "Cincinnati Reds": "CIN",
        "Cleveland Guardians": "CLE",
        "Colorado Rockies": "COL",
        "Detroit Tigers": "DET",
        "Houston Astros": "HOU",
        "Kansas City Royals": "KC",
        "Los Angeles Angels": "LAA",
        "Los Angeles Dodgers": "LAD",
        "Miami Marlins": "MIA",
        "Milwaukee Brewers": "MIL",
        "Minnesota Twins": "MIN",
        "New York Mets": "NYM",
        "New York Yankees": "NYY",
        "Oakland Athletics": "OAK",
        "Philadelphia Phillies": "PHI",
        "Pittsburgh Pirates": "PIT",
        "San Diego Padres": "SD",
        "San Francisco Giants": "SF",
        "Seattle Mariners": "SEA",
        "St. Louis Cardinals": "STL",
        "Tampa Bay Rays": "TB",
        "Texas Rangers": "TEX",
        "Toronto Blue Jays": "TOR",
        "Washington Nationals": "WSH",
    }
    return abbrevs.get(team_name, team_name[:3].upper())
