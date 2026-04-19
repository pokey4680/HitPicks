"""Core probability calculations for BTS.

The fundamental model:
    P(≥1 hit) = 1 - (1 - p_final)^n

Where:
    p_final = per-PA hit probability (adjusted for all factors)
    n       = expected plate appearances
"""

from __future__ import annotations

import math

from bts.config import LINEUP_PA, HOME_PA_DISCOUNT, P_FINAL_CLAMP
from bts.models import FactorBreakdown


def hit_probability(p_per_pa: float, expected_pa: float) -> float:
    """Binomial probability of at least one hit.

    P(≥1 hit) = 1 - (1 - p)^n
    """
    if p_per_pa <= 0 or expected_pa <= 0:
        return 0.0
    if p_per_pa >= 1.0:
        return 1.0
    return 1.0 - (1.0 - p_per_pa) ** expected_pa


def matchup_rate(batter_hpa: float, pitcher_hpa_allowed: float) -> float:
    """Compute the batter-pitcher matchup H/PA using geometric mean.

    This directly combines the batter's hitting rate with the pitcher's
    hit-allowing rate. No league averages needed.

    Examples:
        .300 batter vs .300 pitcher → .300
        .300 batter vs .220 pitcher → .257
        .300 batter vs .350 pitcher → .324
    """
    if batter_hpa <= 0 or pitcher_hpa_allowed <= 0:
        return 0.0
    return math.sqrt(batter_hpa * pitcher_hpa_allowed)


def compute_p_final(
    matchup_hpa: float,
    platoon_mult: float,
    xba_adj: float,
    park_factor: float,
) -> float:
    """Combine all factors into the final per-PA hit probability.

    Each factor is a multiplier on the matchup rate.
    Result is clamped to P_FINAL_CLAMP.
    """
    p = matchup_hpa * platoon_mult * xba_adj * park_factor
    return max(P_FINAL_CLAMP[0], min(P_FINAL_CLAMP[1], p))


def expected_pa(lineup_slot: int, is_home: bool) -> float:
    """Estimate expected plate appearances for a lineup slot."""
    base = LINEUP_PA.get(lineup_slot, 4.0)
    if is_home:
        base *= HOME_PA_DISCOUNT
    return base


def full_prediction(
    batter_hpa: float,
    pitcher_hpa_allowed: float,
    platoon_mult: float,
    xba_adj: float,
    park_factor: float,
    lineup_slot: int,
    is_home: bool,
    pitcher_name: str = "",
    pitcher_hand: str = "",
) -> tuple[float, FactorBreakdown]:
    """Compute the complete prediction for a batter in a game.

    Returns (hit_probability, factor_breakdown).
    """
    m_rate = matchup_rate(batter_hpa, pitcher_hpa_allowed)
    p = compute_p_final(m_rate, platoon_mult, xba_adj, park_factor)
    n = expected_pa(lineup_slot, is_home)
    prob = hit_probability(p, n)

    breakdown = FactorBreakdown(
        batter_hpa=batter_hpa,
        pitcher_hpa_allowed=pitcher_hpa_allowed,
        matchup_hpa=m_rate,
        platoon_mult=platoon_mult,
        xba_adj=xba_adj,
        park_factor=park_factor,
        p_final=p,
        expected_pa=n,
        pitcher_name=pitcher_name,
        pitcher_hand=pitcher_hand,
    )
    return prob, breakdown
