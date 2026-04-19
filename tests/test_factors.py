"""Tests for factor computations."""

import pytest

from bts.config import ConfidenceTier
from bts.factors import (
    blended_batter_hpa,
    determine_confidence,
    park_factor,
    platoon_multiplier,
    stabilized_pitcher_hpa,
    xba_adjustment,
)
from bts.models import BattingStats, PitchingStats


class TestBlendedBatterHPA:
    def test_season_only_no_recent(self):
        """With no recent data meeting min_pa, should rely on season."""
        season = BattingStats(pa=400, ab=360, hits=100)
        # All recent windows have 0 PA
        recent = {7: BattingStats(), 14: BattingStats(), 30: BattingStats()}
        result = blended_batter_hpa(season, recent)
        assert abs(result - 0.250) < 0.01

    def test_recent_data_blends_in(self):
        """When recent windows have data, they should affect the result."""
        season = BattingStats(pa=400, ab=360, hits=100)  # .250 H/PA
        recent = {
            7: BattingStats(pa=20, ab=18, hits=8),     # .400 H/PA (hot!)
            14: BattingStats(pa=35, ab=32, hits=12),    # .343 H/PA
            30: BattingStats(pa=80, ab=72, hits=25),    # .313 H/PA
        }
        result = blended_batter_hpa(season, recent)
        # Should be above season rate due to hot recent stretch
        assert result > 0.250

    def test_prior_season_stabilization_early(self):
        """With few current PA, prior season should help stabilize."""
        season = BattingStats(pa=20, ab=18, hits=8)  # .400 — small sample noise
        recent = {7: BattingStats(), 14: BattingStats(), 30: BattingStats()}
        # Prior season of .270 should pull it down
        result = blended_batter_hpa(season, recent, prior_season_hpa=0.270)
        assert result < 0.400
        assert result > 0.270

    def test_prior_season_drops_off(self):
        """With 100+ PA, prior season should be ignored."""
        season = BattingStats(pa=150, ab=135, hits=45)  # .300 H/PA
        recent = {7: BattingStats(), 14: BattingStats(), 30: BattingStats()}
        result_with_prior = blended_batter_hpa(season, recent, prior_season_hpa=0.200)
        result_without_prior = blended_batter_hpa(season, recent)
        assert abs(result_with_prior - result_without_prior) < 0.001


class TestStabilizedPitcherHPA:
    def test_full_season_data(self):
        """With enough PA, just use the raw rate."""
        stats = PitchingStats(pa_faced=500, hits_allowed=125)  # .250
        result = stabilized_pitcher_hpa(stats)
        assert abs(result - 0.250) < 0.001

    def test_prior_season_blend(self):
        """Early season should blend with prior."""
        stats = PitchingStats(pa_faced=30, hits_allowed=12)  # .400 — hot start
        result = stabilized_pitcher_hpa(stats, prior_season_hpa=0.240)
        assert result < 0.400
        assert result > 0.240

    def test_no_data_uses_prior(self):
        stats = PitchingStats()
        result = stabilized_pitcher_hpa(stats, prior_season_hpa=0.260)
        assert abs(result - 0.260) < 0.001


class TestPlatoonMultiplier:
    def test_favorable_platoon(self):
        """Batter who hits better vs this handedness → multiplier > 1."""
        # .280 overall, .310 vs RHP
        mult = platoon_multiplier(
            overall_hpa=0.280, split_hpa=0.310, career_split_hpa=None,
            split_pa=100, career_split_pa=0,
        )
        assert mult > 1.0

    def test_unfavorable_platoon(self):
        """Batter who struggles vs this handedness → multiplier < 1."""
        mult = platoon_multiplier(
            overall_hpa=0.280, split_hpa=0.230, career_split_hpa=None,
            split_pa=100, career_split_pa=0,
        )
        assert mult < 1.0

    def test_small_sample_falls_back_to_career(self):
        """If current-season split has < 50 PA, use career split."""
        mult = platoon_multiplier(
            overall_hpa=0.270, split_hpa=0.500, career_split_hpa=0.290,
            split_pa=10, career_split_pa=300,
        )
        # Should use career .290 / .270 ≈ 1.074, not the noisy .500
        assert abs(mult - (0.290 / 0.270)) < 0.01

    def test_no_data_returns_neutral(self):
        mult = platoon_multiplier(
            overall_hpa=0.270, split_hpa=None, career_split_hpa=None,
            split_pa=0, career_split_pa=0,
        )
        assert mult == 1.0


class TestXBAAdj:
    def test_disabled_returns_neutral(self):
        """xBA is disabled (XBA_ALPHA=0.0) — always returns 1.0."""
        assert xba_adjustment(xba=0.300, actual_ba=0.260) == 1.0
        assert xba_adjustment(xba=0.240, actual_ba=0.300) == 1.0

    def test_no_data(self):
        assert xba_adjustment(xba=None, actual_ba=None) == 1.0


class TestParkFactor:
    def test_coors(self):
        pf = park_factor("Coors Field")
        assert pf == 1.13

    def test_petco(self):
        pf = park_factor("Petco Park")
        assert pf == 0.95

    def test_unknown_venue(self):
        pf = park_factor("Some New Stadium")
        assert pf == 1.0


class TestConfidenceTier:
    def test_established(self):
        assert determine_confidence(250, True) == ConfidenceTier.ESTABLISHED
        assert determine_confidence(200, False) == ConfidenceTier.ESTABLISHED

    def test_established_with_prior(self):
        assert determine_confidence(100, True) == ConfidenceTier.ESTABLISHED

    def test_part_time(self):
        assert determine_confidence(75, False) == ConfidenceTier.PARTTIME

    def test_part_time_with_prior(self):
        assert determine_confidence(30, True) == ConfidenceTier.PARTTIME

    def test_unproven(self):
        assert determine_confidence(15, False) == ConfidenceTier.UNPROVEN
        assert determine_confidence(20, False) == ConfidenceTier.UNPROVEN
