"""Streak-strategy simulation for BTS.

Takes a list of BacktestResult objects (backtest output), groups by date, and
evaluates different day-by-day picking strategies:

  - top-1 always
  - threshold (only play when top-1 >= X)
  - confidence-tier filter
  - lineup-slot filter (top-of-order only)
  - probability-gap filter (play only when top-1 clearly beats top-2)
  - combined filters

For each strategy, reports observed streak metrics over the window and a
bootstrap Monte Carlo estimate of max-streak distribution in a full
season-length sequence. The Monte Carlo answers the central question:
given these daily outcomes, what's P(reaching a 57-game streak) in one season?

Skips don't break streaks (models the Beat the Streak "pass" / no-pick option).
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from rich.console import Console
from rich.table import Table

from bts.config import ConfidenceTier
from bts.models import BacktestResult

console = Console()


# A strategy is a function from "today's predictions" to "pick or skip".
Strategy = Callable[[list[BacktestResult]], "BacktestResult | None"]


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

def top1() -> Strategy:
    """Always pick the highest-probability prediction of the day."""
    def _pick(day: list[BacktestResult]) -> BacktestResult | None:
        if not day:
            return None
        return max(day, key=lambda r: r.prediction.hit_probability)
    return _pick


def threshold(min_prob: float) -> Strategy:
    """Pick top-1 only if its probability is at or above a threshold."""
    def _pick(day: list[BacktestResult]) -> BacktestResult | None:
        if not day:
            return None
        top = max(day, key=lambda r: r.prediction.hit_probability)
        return top if top.prediction.hit_probability >= min_prob else None
    return _pick


def tier_only(tier: ConfidenceTier, min_prob: float = 0.0) -> Strategy:
    """Pick the best prediction restricted to a confidence tier, with optional threshold."""
    def _pick(day: list[BacktestResult]) -> BacktestResult | None:
        pool = [r for r in day if r.prediction.confidence == tier]
        if not pool:
            return None
        top = max(pool, key=lambda r: r.prediction.hit_probability)
        return top if top.prediction.hit_probability >= min_prob else None
    return _pick


def top_of_order(max_slot: int, min_prob: float = 0.0) -> Strategy:
    """Restrict pool to top-of-order hitters (more PAs, lower variance)."""
    def _pick(day: list[BacktestResult]) -> BacktestResult | None:
        pool = [r for r in day if r.prediction.lineup_slot <= max_slot]
        if not pool:
            return None
        top = max(pool, key=lambda r: r.prediction.hit_probability)
        return top if top.prediction.hit_probability >= min_prob else None
    return _pick


def probability_gap(min_gap: float, min_prob: float = 0.0) -> Strategy:
    """Only play on days with a clear favorite (top-1 beats top-2 by min_gap)."""
    def _pick(day: list[BacktestResult]) -> BacktestResult | None:
        if len(day) < 2:
            return None
        ranked = sorted(day, key=lambda r: r.prediction.hit_probability, reverse=True)
        top, second = ranked[0], ranked[1]
        if top.prediction.hit_probability - second.prediction.hit_probability < min_gap:
            return None
        if top.prediction.hit_probability < min_prob:
            return None
        return top
    return _pick


def combined(min_prob: float, tier: ConfidenceTier, max_slot: int = 9) -> Strategy:
    """Threshold + tier + lineup-slot filters combined."""
    def _pick(day: list[BacktestResult]) -> BacktestResult | None:
        pool = [
            r for r in day
            if r.prediction.confidence == tier
            and r.prediction.lineup_slot <= max_slot
        ]
        if not pool:
            return None
        top = max(pool, key=lambda r: r.prediction.hit_probability)
        return top if top.prediction.hit_probability >= min_prob else None
    return _pick


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class DayOutcome:
    """A strategy's decision and outcome for one day."""

    date: date
    played: bool
    pick_prob: float = 0.0
    got_hit: bool = False
    pick_name: str = ""


@dataclass
class StrategyResult:
    """Aggregate result of running a strategy over a window."""

    name: str
    days_in_window: int
    days_played: int
    days_hit: int
    observed_longest: int
    observed_avg: float
    observed_streak_count: int

    # Monte Carlo over a season-length sequence
    mc_longest_mean: float = 0.0
    mc_longest_p50: int = 0
    mc_longest_p90: int = 0
    mc_longest_max: int = 0
    mc_p_target: float = 0.0

    outcomes: list[DayOutcome] = field(default_factory=list)

    @property
    def play_rate(self) -> float:
        return self.days_played / self.days_in_window if self.days_in_window else 0.0

    @property
    def hit_rate(self) -> float:
        return self.days_hit / self.days_played if self.days_played else 0.0


def _compute_streaks(outcomes: list[DayOutcome]) -> list[int]:
    """Compute consecutive-hit streak lengths. Skips don't break streaks."""
    streaks: list[int] = []
    current = 0
    for o in outcomes:
        if not o.played:
            continue
        if o.got_hit:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    return streaks


def simulate(
    results: list[BacktestResult],
    strategy: Strategy,
    *,
    name: str,
    season_days: int = 180,
    n_mc: int = 10000,
    target_streak: int = 57,
    seed: int = 42,
) -> StrategyResult:
    """Run a strategy over `results` and return aggregated metrics + MC estimate."""

    by_date: dict[date, list[BacktestResult]] = defaultdict(list)
    for r in results:
        by_date[r.prediction.game_date].append(r)

    outcomes: list[DayOutcome] = []
    for d in sorted(by_date.keys()):
        pick = strategy(by_date[d])
        if pick is None:
            outcomes.append(DayOutcome(date=d, played=False))
        else:
            outcomes.append(DayOutcome(
                date=d,
                played=True,
                pick_prob=pick.prediction.hit_probability,
                got_hit=pick.actual_got_hit,
                pick_name=pick.prediction.player_name,
            ))

    streak_lens = _compute_streaks(outcomes)
    days_played = sum(1 for o in outcomes if o.played)
    days_hit = sum(1 for o in outcomes if o.played and o.got_hit)

    # Monte Carlo: bootstrap resample the observed outcome sequence to build
    # `season_days`-length seasons, track max-streak distribution.
    rng = random.Random(seed)
    n_days = len(outcomes)
    max_streaks: list[int] = []
    hit_target = 0

    if n_days == 0:
        mc_mean = mc_p50 = mc_p90 = mc_max = 0
        mc_p = 0.0
    else:
        for _ in range(n_mc):
            current = 0
            max_s = 0
            for _ in range(season_days):
                o = outcomes[rng.randrange(n_days)]
                if not o.played:
                    continue
                if o.got_hit:
                    current += 1
                    if current > max_s:
                        max_s = current
                else:
                    current = 0
            max_streaks.append(max_s)
            if max_s >= target_streak:
                hit_target += 1

        max_streaks.sort()
        mc_mean = sum(max_streaks) / len(max_streaks)
        mc_p50 = max_streaks[len(max_streaks) // 2]
        mc_p90 = max_streaks[int(len(max_streaks) * 0.9)]
        mc_max = max_streaks[-1]
        mc_p = hit_target / n_mc

    return StrategyResult(
        name=name,
        days_in_window=len(outcomes),
        days_played=days_played,
        days_hit=days_hit,
        observed_longest=max(streak_lens) if streak_lens else 0,
        observed_avg=sum(streak_lens) / len(streak_lens) if streak_lens else 0.0,
        observed_streak_count=len(streak_lens),
        mc_longest_mean=mc_mean,
        mc_longest_p50=mc_p50,
        mc_longest_p90=mc_p90,
        mc_longest_max=mc_max,
        mc_p_target=mc_p,
        outcomes=outcomes,
    )


# ---------------------------------------------------------------------------
# Preset strategy suite
# ---------------------------------------------------------------------------

def default_suite() -> list[tuple[str, Strategy]]:
    """Canonical list of strategies to compare."""
    return [
        ("Baseline: always top-1",              top1()),
        ("Threshold >= 0.70",                   threshold(0.70)),
        ("Threshold >= 0.75",                   threshold(0.75)),
        ("Threshold >= 0.78",                   threshold(0.78)),
        ("Threshold >= 0.80",                   threshold(0.80)),
        ("Threshold >= 0.82",                   threshold(0.82)),
        ("Established only",                    tier_only(ConfidenceTier.ESTABLISHED)),
        ("Established + >= 0.78",               tier_only(ConfidenceTier.ESTABLISHED, 0.78)),
        ("Established + >= 0.80",               tier_only(ConfidenceTier.ESTABLISHED, 0.80)),
        ("Top-of-order (slots 1-5)",            top_of_order(5)),
        ("Top-of-order + >= 0.78",              top_of_order(5, 0.78)),
        ("Clear favorite (gap >= 0.03)",        probability_gap(0.03)),
        ("Clear favorite (gap >= 0.05)",        probability_gap(0.05)),
        ("Combined: Est + 1-5 + >= 0.78",       combined(0.78, ConfidenceTier.ESTABLISHED, max_slot=5)),
        ("Combined: Est + 1-5 + >= 0.80",       combined(0.80, ConfidenceTier.ESTABLISHED, max_slot=5)),
    ]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_strategy_report(
    results: list[StrategyResult],
    *,
    season_days: int,
    target_streak: int,
    window_label: str = "",
) -> None:
    """Print a comparison table of strategies."""
    console.print()
    title = "Strategy Comparison"
    if window_label:
        title += f"  ({window_label})"
    console.print(f"[bold]{title}[/bold]", justify="center")
    console.print(
        f"[dim]Monte Carlo: {season_days}-day season, target streak = {target_streak}[/dim]",
        justify="center",
    )
    console.print()

    table = Table(show_header=True, header_style="bold", border_style="dim")
    table.add_column("Strategy", width=34)
    table.add_column("Played", justify="right", width=7)
    table.add_column("Hit%", justify="right", width=7)
    table.add_column("Obs Max", justify="right", width=8)
    table.add_column("MC Mean", justify="right", width=8)
    table.add_column("MC p50", justify="right", width=7)
    table.add_column("MC p90", justify="right", width=7)
    table.add_column("MC Max", justify="right", width=7)
    table.add_column(f"P(≥{target_streak})", justify="right", width=10)

    # Highlight best on MC p90 and on P(target)
    if results:
        best_p90 = max(r.mc_longest_p90 for r in results)
        best_p_target = max(r.mc_p_target for r in results)
    else:
        best_p90 = 0
        best_p_target = 0.0

    for r in results:
        days_played = f"{r.days_played}/{r.days_in_window}"
        hit_rate = f"{r.hit_rate:.1%}" if r.days_played else "—"

        p90_str = f"{r.mc_longest_p90}"
        if r.mc_longest_p90 == best_p90 and best_p90 > 0:
            p90_str = f"[bold green]{p90_str}[/bold green]"

        p_target_str = f"{r.mc_p_target:.2%}" if r.mc_p_target > 0 else "<0.01%"
        if r.mc_p_target == best_p_target and best_p_target > 0:
            p_target_str = f"[bold green]{p_target_str}[/bold green]"

        table.add_row(
            r.name,
            days_played,
            hit_rate,
            str(r.observed_longest),
            f"{r.mc_longest_mean:.1f}",
            str(r.mc_longest_p50),
            p90_str,
            str(r.mc_longest_max),
            p_target_str,
        )

    console.print(table)
    console.print()
    console.print(
        "[dim]Observed Max = longest streak actually seen in the window.\n"
        "MC Mean/p50/p90/Max = distribution of longest-streak-per-season "
        "across bootstrap-resampled seasons.\n"
        f"P(≥{target_streak}) = fraction of simulated seasons that reached "
        f"the target streak.[/dim]"
    )
    console.print()
