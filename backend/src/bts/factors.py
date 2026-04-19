"""Factor computations for the BTS model.

Each function computes one adjustment factor used in the hit probability model.
All factors are multipliers (1.0 = no effect).
"""

from __future__ import annotations

import logging
from datetime import date

from bts.config import (
    DEFAULT_PARK_FACTOR,
    MIN_PA_TO_QUALIFY,
    PARK_FACTORS,
    PRIOR_SEASON_CUTOFF_PA,
    PRIOR_SEASON_K,
    RECENCY_WINDOWS,
    XBA_ALPHA,
    XBA_CLAMP,
    ConfidenceTier,
)
from bts.models import BattingStats, PitchingStats

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batter base rate — multi-window recency blend
# ---------------------------------------------------------------------------

def blended_batter_hpa(
    season_stats: BattingStats,
    recent_stats: dict[int, BattingStats],  # keyed by window days
    prior_season_hpa: float | None = None,
    *,
    recency_windows: list[dict] | None = None,
) -> float:
    """Compute the blended batter H/PA from multiple time windows.

    Args:
        season_stats: Full current-season batting stats.
        recent_stats: Dict mapping window_days → BattingStats for that window.
            e.g. {7: stats_7d, 14: stats_14d, 30: stats_30d}
        prior_season_hpa: Player's H/PA from last season (for stabilization).
            None if no prior-season data.
        recency_windows: Override for RECENCY_WINDOWS (used in ablation studies).

    Returns:
        Blended H/PA rate.
    """
    windows = recency_windows or RECENCY_WINDOWS

    # Step 1: Stabilize the season rate with prior-season data if needed
    season_hpa = _stabilized_season_rate(season_stats, prior_season_hpa)

    # Step 2: Compute weighted blend of recent windows + season
    total_weight = 0.0
    weighted_sum = 0.0

    for window in windows:
        days = window["days"]
        w = window["weight"]
        stats = recent_stats.get(days)

        if stats and stats.pa >= window["min_pa"]:
            weighted_sum += w * stats.hpa
            total_weight += w
        # If this window doesn't have enough PA, its weight redistributes
        # to the remaining windows (handled by normalizing at the end).

    # Add season rate with remaining weight
    season_weight = 1.0 - sum(w["weight"] for w in windows)
    weighted_sum += season_weight * season_hpa
    total_weight += season_weight

    if total_weight <= 0:
        return season_hpa

    return weighted_sum / total_weight


def _stabilized_season_rate(
    season_stats: BattingStats,
    prior_season_hpa: float | None,
) -> float:
    """Stabilize early-season rates using prior-season data.

    Uses a Bayesian-style blend:
        p = (H_current + k * prior_hpa) / (PA_current + k)

    Once the player has enough current-season PA, prior season drops off.
    """
    if season_stats.pa == 0:
        # No current-season data at all — use prior if available
        return prior_season_hpa if prior_season_hpa else 0.0

    if season_stats.pa >= PRIOR_SEASON_CUTOFF_PA or prior_season_hpa is None:
        # Enough current data, or no prior data — use raw current season
        return season_stats.hpa

    # Blend: weight prior season inversely with current PA
    # As PA grows toward PRIOR_SEASON_CUTOFF_PA, k shrinks toward 0
    effective_k = PRIOR_SEASON_K * (
        1.0 - season_stats.pa / PRIOR_SEASON_CUTOFF_PA
    )
    return (
        (season_stats.hits + effective_k * prior_season_hpa)
        / (season_stats.pa + effective_k)
    )


# ---------------------------------------------------------------------------
# Pitcher H/PA allowed — with prior-season stabilization
# ---------------------------------------------------------------------------

def stabilized_pitcher_hpa(
    current_stats: PitchingStats,
    prior_season_hpa: float | None = None,
) -> float:
    """Compute the pitcher's stabilized H/PA-allowed rate."""
    if current_stats.pa_faced == 0:
        return prior_season_hpa if prior_season_hpa else 0.250  # pitcher-specific fallback

    if current_stats.pa_faced >= PRIOR_SEASON_CUTOFF_PA or prior_season_hpa is None:
        return current_stats.hpa_allowed

    effective_k = PRIOR_SEASON_K * (
        1.0 - current_stats.pa_faced / PRIOR_SEASON_CUTOFF_PA
    )
    return (
        (current_stats.hits_allowed + effective_k * prior_season_hpa)
        / (current_stats.pa_faced + effective_k)
    )


# ---------------------------------------------------------------------------
# Platoon adjustment
# ---------------------------------------------------------------------------

def platoon_multiplier(
    overall_hpa: float,
    split_hpa: float | None,
    career_split_hpa: float | None,
    split_pa: int,
    career_split_pa: int,
    min_pa: int = 50,
) -> float:
    """Compute the platoon adjustment multiplier.

    Tries current-season split first; falls back to career split;
    falls back to no adjustment.

    Returns a multiplier (>1.0 = favorable platoon, <1.0 = unfavorable).
    """
    if overall_hpa <= 0:
        return 1.0

    # Prefer current-season split if we have enough PA
    if split_hpa is not None and split_pa >= min_pa:
        return split_hpa / overall_hpa

    # Fall back to career split
    if career_split_hpa is not None and career_split_pa >= min_pa:
        return career_split_hpa / overall_hpa

    # No reliable split data — skip adjustment
    return 1.0


# ---------------------------------------------------------------------------
# xBA adjustment
# ---------------------------------------------------------------------------

def xba_adjustment(xba: float | None, actual_ba: float | None) -> float:
    """Compute the xBA contact quality adjustment.

    If xBA > BA, the player is "unlucky" and due for regression upward.
    If xBA < BA, they're "lucky" and likely to regress down.

    Returns a multiplier capped at XBA_CLAMP.
    Currently disabled (XBA_ALPHA=0.0) based on backtest results.
    """
    if XBA_ALPHA == 0.0 or xba is None or actual_ba is None:
        return 1.0

    adj = 1.0 + XBA_ALPHA * (xba - actual_ba)
    return max(XBA_CLAMP[0], min(XBA_CLAMP[1], adj))


# ---------------------------------------------------------------------------
# Park factor
# ---------------------------------------------------------------------------

def park_factor(venue_name: str) -> float:
    """Look up the park factor for a venue."""
    return PARK_FACTORS.get(venue_name, DEFAULT_PARK_FACTOR)


# ---------------------------------------------------------------------------
# Confidence tier
# ---------------------------------------------------------------------------

def determine_confidence(
    current_season_pa: int,
    has_prior_season: bool,
) -> ConfidenceTier:
    """Determine the confidence tier for a player.

    - Established: 200+ PA current season, or 100+ PA with prior season
    - Part-time: 50-199 PA current season
    - Unproven: <50 PA with no prior season → excluded from predictions
    """
    if current_season_pa >= 200:
        return ConfidenceTier.ESTABLISHED
    if current_season_pa >= 100 and has_prior_season:
        return ConfidenceTier.ESTABLISHED
    if current_season_pa >= MIN_PA_TO_QUALIFY:
        return ConfidenceTier.PARTTIME
    if has_prior_season and current_season_pa >= 25:
        return ConfidenceTier.PARTTIME
    return ConfidenceTier.UNPROVEN
