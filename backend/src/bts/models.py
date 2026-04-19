"""Data models for BTS."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from bts.config import ConfidenceTier


@dataclass
class PlayerBio:
    """Static player information (cached permanently)."""

    player_id: int
    full_name: str
    bat_side: str  # "L", "R", or "S" (switch)
    primary_position: str

    @property
    def is_switch(self) -> bool:
        return self.bat_side == "S"


@dataclass
class PitcherBio:
    """Static pitcher information."""

    player_id: int
    full_name: str
    pitch_hand: str  # "L" or "R"


@dataclass
class BattingStats:
    """Aggregated batting statistics for a period."""

    pa: int = 0
    ab: int = 0
    hits: int = 0
    bb: int = 0
    hbp: int = 0
    k: int = 0

    @property
    def hpa(self) -> float:
        """Hits per plate appearance — the core rate metric."""
        return self.hits / self.pa if self.pa > 0 else 0.0

    @property
    def ba(self) -> float:
        """Traditional batting average (H/AB)."""
        return self.hits / self.ab if self.ab > 0 else 0.0


@dataclass
class PitchingStats:
    """Aggregated pitching statistics."""

    pa_faced: int = 0     # Total batters faced (TBF)
    hits_allowed: int = 0
    bb_allowed: int = 0
    k: int = 0
    ip: float = 0.0

    @property
    def hpa_allowed(self) -> float:
        """Hits allowed per plate appearance faced."""
        return self.hits_allowed / self.pa_faced if self.pa_faced > 0 else 0.0


@dataclass
class GameInfo:
    """Information about a single scheduled game."""

    game_pk: int
    game_date: date
    away_team: str
    home_team: str
    venue_name: str
    status: str  # "Preview", "Live", "Final", etc.


@dataclass
class LineupEntry:
    """A single batter in a lineup."""

    player_id: int
    player_name: str
    lineup_slot: int  # 1-9
    is_home: bool
    team: str


@dataclass
class StartingPitcher:
    """Probable/actual starting pitcher for a game."""

    player_id: int
    player_name: str
    pitch_hand: str  # "L" or "R"
    is_home: bool
    team: str


@dataclass
class FactorBreakdown:
    """Decomposition of the prediction for the 'Why' column."""

    batter_hpa: float          # Blended batter H/PA (season + recent)
    pitcher_hpa_allowed: float  # Opposing SP H/PA allowed
    matchup_hpa: float          # Geometric mean of batter + pitcher
    platoon_mult: float         # Platoon adjustment multiplier
    xba_adj: float              # xBA adjustment multiplier
    park_factor: float          # Park factor multiplier
    p_final: float              # Final per-PA hit probability
    expected_pa: float          # Expected plate appearances
    pitcher_name: str = ""
    pitcher_hand: str = ""

    @property
    def platoon_pct(self) -> float:
        """Platoon effect as a signed percentage."""
        return (self.platoon_mult - 1.0) * 100

    @property
    def xba_pct(self) -> float:
        """xBA effect as a signed percentage."""
        return (self.xba_adj - 1.0) * 100

    @property
    def park_pct(self) -> float:
        """Park effect as a signed percentage."""
        return (self.park_factor - 1.0) * 100


@dataclass
class Prediction:
    """A complete prediction for one batter in one game."""

    player_id: int
    player_name: str
    team: str
    game_pk: int
    game_date: date
    hit_probability: float       # P(≥1 hit) — the headline number
    confidence: ConfidenceTier
    factors: FactorBreakdown
    current_season_pa: int       # For transparency
    lineup_slot: int
    is_home: bool


@dataclass
class BacktestResult:
    """One prediction matched against its actual outcome."""

    prediction: Prediction
    actual_got_hit: bool  # Did the player record ≥1 hit?
    actual_hits: int = 0
    actual_ab: int = 0
    actual_pa: int = 0
