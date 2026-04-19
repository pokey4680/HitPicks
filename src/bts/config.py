"""Constants, park factors, and configuration for BTS."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CACHE_DIR = Path.home() / ".bts"
CACHE_DB = CACHE_DIR / "cache.db"

# ---------------------------------------------------------------------------
# MLB Stats API
# ---------------------------------------------------------------------------
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# ---------------------------------------------------------------------------
# Baseball Savant
# ---------------------------------------------------------------------------
SAVANT_XBA_URL = (
    "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
    "?type=batter&year={year}&position=&team=&min=25&csv=true"
)

# ---------------------------------------------------------------------------
# HTTP behaviour
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT = 15  # seconds
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = [1.0, 2.0, 4.0]  # seconds between retries
RATE_LIMIT_DELAY = 0.25  # courtesy delay between MLB API calls

# ---------------------------------------------------------------------------
# Confidence tiers — based on current-season PA
# ---------------------------------------------------------------------------

class ConfidenceTier(str, Enum):
    """How much we trust a player's current stats."""

    ESTABLISHED = "established"  # 200+ PA current season OR 500+ career
    PARTTIME = "part-time"       # 50-199 PA current season
    UNPROVEN = "unproven"        # <50 PA current season, no meaningful prior data

    @property
    def symbol(self) -> str:
        return {"established": "★", "part-time": "◆", "unproven": "○"}[self.value]

    @property
    def label(self) -> str:
        return {"established": "Established", "part-time": "Part-Time", "unproven": "Unproven"}[
            self.value
        ]


# Minimum current-season PA for a player to appear in predictions at all.
# Players below this with no prior-season data are excluded.
MIN_PA_TO_QUALIFY = 50

# ---------------------------------------------------------------------------
# Recency weighting
#
# We don't know the optimal recency window a priori. Instead of hardcoding
# "14 days", we evaluate MULTIPLE windows and let backtesting reveal which
# is most predictive. The model computes H/PA for each window and blends them.
#
# Current/recent performance is king — prior season is ONLY used for early-
# season stabilization and drops off once the player has 100+ PA this year.
# ---------------------------------------------------------------------------

# Recency windows — tuned by backtest results (May–Aug 2025, 30K predictions).
# The 7-day window was pure noise. The 30-day window had the best Brier score.
# Remaining weight (1.0 - sum) goes to full-season rate (0.40 here).
RECENCY_WINDOWS: list[dict] = [
    {"days": 30, "min_pa": 30, "weight": 0.60, "label": "30d"},
]

# Prior-season stabilization: blending constant.
# p_stabilized = (H_current + k * prior_hpa) / (PA_current + k)
# As PA_current grows, this shrinks toward the player's current-season rate.
# k = 0 means "trust current season entirely" (used once PA >= cutoff).
PRIOR_SEASON_K = 60            # PA-equivalents of prior-season weight
PRIOR_SEASON_CUTOFF_PA = 100   # Once player reaches this many PA, ignore prior season

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------

# xBA adjustment — DISABLED based on backtest results (zero lift on Brier).
# Kept as config in case future analysis finds value with point-in-time xBA.
XBA_ALPHA = 0.0  # 0.0 = disabled
XBA_CLAMP = (0.90, 1.10)

# Platoon adjustment — DISABLED based on backtest results.
# Platoon splits improve calibration but hurt top-pick accuracy and streaks.
# For BTS (pick the #1 hitter), ranking accuracy matters more than calibration.
PLATOON_ENABLED = False

# Final per-PA hit probability clamp
P_FINAL_CLAMP = (0.10, 0.42)

# ---------------------------------------------------------------------------
# Expected plate appearances by lineup slot
# Empirically derived from 30,000+ boxscore entries (May–Aug 2025)
# ---------------------------------------------------------------------------
LINEUP_PA: dict[int, float] = {
    1: 4.28,
    2: 4.19,
    3: 4.10,
    4: 4.05,
    5: 3.84,
    6: 3.69,
    7: 3.50,
    8: 3.33,
    9: 3.03,
}

# Home team may not bat in the 9th if leading
HOME_PA_DISCOUNT = 0.97

# ---------------------------------------------------------------------------
# Park factors (hit-specific, multi-year averages from FanGraphs)
#
# 1.00 = neutral. >1.00 = hitter friendly. <1.00 = pitcher friendly.
# These are for HITS specifically, not runs.
# ---------------------------------------------------------------------------
PARK_FACTORS: dict[str, float] = {
    # AL East
    "Oriole Park at Camden Yards": 1.02,
    "Fenway Park": 1.05,
    "Rogers Centre": 1.00,
    "Tropicana Field": 0.97,
    "Yankee Stadium": 1.02,
    # AL Central
    "Guaranteed Rate Field": 1.01,
    "Progressive Field": 0.99,
    "Comerica Park": 0.98,
    "Kauffman Stadium": 1.01,
    "Target Field": 1.00,
    # AL West
    "Angel Stadium": 0.98,
    "Minute Maid Park": 1.02,
    "Oakland Coliseum": 0.94,
    "T-Mobile Park": 0.96,
    "Globe Life Field": 1.03,
    # NL East
    "Truist Park": 1.00,
    "loanDepot park": 0.97,
    "Citi Field": 0.97,
    "Citizens Bank Park": 1.04,
    "Nationals Park": 1.00,
    # NL Central
    "Wrigley Field": 1.03,
    "Great American Ball Park": 1.06,
    "American Family Field": 1.02,
    "PNC Park": 0.99,
    "Busch Stadium": 0.98,
    # NL West
    "Chase Field": 1.04,
    "Coors Field": 1.13,
    "Dodger Stadium": 0.97,
    "Petco Park": 0.95,
    "Oracle Park": 0.95,
}

# Fallback for unknown / new venues
DEFAULT_PARK_FACTOR = 1.00
