"""Tests for the core probability math."""

import math
import pytest

from bts.probability import (
    compute_p_final,
    expected_pa,
    full_prediction,
    hit_probability,
    matchup_rate,
)


class TestHitProbability:
    def test_basic_calculation(self):
        """P(≥1 hit) = 1 - (1-p)^n"""
        # .300 hitter, 4 PA → 1 - 0.7^4 = 0.7599
        result = hit_probability(0.300, 4)
        assert abs(result - 0.7599) < 0.001

    def test_zero_probability(self):
        assert hit_probability(0.0, 4) == 0.0

    def test_zero_pa(self):
        assert hit_probability(0.300, 0) == 0.0

    def test_certain_hit(self):
        assert hit_probability(1.0, 1) == 1.0

    def test_more_pa_increases_probability(self):
        """More plate appearances should increase P(≥1 hit)."""
        p3 = hit_probability(0.250, 3)
        p5 = hit_probability(0.250, 5)
        assert p5 > p3

    def test_higher_rate_increases_probability(self):
        """Higher per-PA rate should increase P(≥1 hit)."""
        p_low = hit_probability(0.200, 4)
        p_high = hit_probability(0.300, 4)
        assert p_high > p_low

    def test_typical_leadoff_hitter(self):
        """A good leadoff hitter should have ~80%+ probability."""
        # .280 H/PA, 4.85 PA
        result = hit_probability(0.280, 4.85)
        assert result > 0.78

    def test_typical_bottom_order(self):
        """A weaker hitter lower in the order should be lower."""
        # .220 H/PA, 4.05 PA
        result = hit_probability(0.220, 4.05)
        assert result < 0.70


class TestMatchupRate:
    def test_equal_rates(self):
        """Geometric mean of equal values is that value."""
        result = matchup_rate(0.300, 0.300)
        assert abs(result - 0.300) < 0.001

    def test_good_batter_vs_good_pitcher(self):
        """.300 batter vs .220 pitcher (dominant) → ~.257"""
        result = matchup_rate(0.300, 0.220)
        expected = math.sqrt(0.300 * 0.220)
        assert abs(result - expected) < 0.001
        assert result < 0.300  # Pitcher suppresses

    def test_good_batter_vs_bad_pitcher(self):
        """.300 batter vs .350 pitcher (hittable) → ~.324"""
        result = matchup_rate(0.300, 0.350)
        assert result > 0.300  # Pitcher inflates

    def test_zero_handling(self):
        assert matchup_rate(0.0, 0.300) == 0.0
        assert matchup_rate(0.300, 0.0) == 0.0


class TestComputePFinal:
    def test_neutral_factors(self):
        """All-neutral adjustments should pass through matchup rate."""
        result = compute_p_final(0.275, 1.0, 1.0, 1.0)
        assert abs(result - 0.275) < 0.001

    def test_favorable_platoon(self):
        """Favorable platoon should increase p."""
        neutral = compute_p_final(0.275, 1.0, 1.0, 1.0)
        favorable = compute_p_final(0.275, 1.06, 1.0, 1.0)
        assert favorable > neutral

    def test_coors_field_boost(self):
        """Coors Field park factor should boost p."""
        neutral = compute_p_final(0.275, 1.0, 1.0, 1.0)
        coors = compute_p_final(0.275, 1.0, 1.0, 1.13)
        assert coors > neutral

    def test_clamping_low(self):
        """Should not go below 0.10."""
        result = compute_p_final(0.05, 0.8, 0.9, 0.9)
        assert result >= 0.10

    def test_clamping_high(self):
        """Should not go above 0.42."""
        result = compute_p_final(0.50, 1.1, 1.1, 1.15)
        assert result <= 0.42


class TestExpectedPA:
    def test_leadoff(self):
        pa = expected_pa(1, is_home=False)
        assert abs(pa - 4.28) < 0.01

    def test_home_discount(self):
        away = expected_pa(1, is_home=False)
        home = expected_pa(1, is_home=True)
        assert home < away

    def test_ninth_hitter(self):
        pa = expected_pa(9, is_home=False)
        assert abs(pa - 3.03) < 0.01


class TestFullPrediction:
    def test_returns_probability_and_breakdown(self):
        prob, breakdown = full_prediction(
            batter_hpa=0.280,
            pitcher_hpa_allowed=0.260,
            platoon_mult=1.05,
            xba_adj=1.02,
            park_factor=1.0,
            lineup_slot=2,
            is_home=False,
            pitcher_name="Test Pitcher",
            pitcher_hand="R",
        )
        assert 0.0 < prob < 1.0
        assert breakdown.pitcher_name == "Test Pitcher"
        assert breakdown.pitcher_hand == "R"
        assert breakdown.expected_pa > 0
        assert breakdown.p_final > 0

    def test_all_factors_affect_probability(self):
        """Each factor should independently affect the probability."""
        base_prob, _ = full_prediction(0.270, 0.260, 1.0, 1.0, 1.0, 5, False)

        # Better platoon
        plat_prob, _ = full_prediction(0.270, 0.260, 1.10, 1.0, 1.0, 5, False)
        assert plat_prob > base_prob

        # Better xBA
        xba_prob, _ = full_prediction(0.270, 0.260, 1.0, 1.05, 1.0, 5, False)
        assert xba_prob > base_prob

        # Better park
        park_prob, _ = full_prediction(0.270, 0.260, 1.0, 1.0, 1.13, 5, False)
        assert park_prob > base_prob
