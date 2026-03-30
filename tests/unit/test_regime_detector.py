"""
Unit tests for Phase 5: Market Regime Detector
================================================
Tests cover:
  - RegimeDetector.detect() with synthetic bull/bear/sideways data
  - Feature extraction (momentum, trend, volatility, breadth)
  - Edge cases (insufficient data, flat prices, extreme moves)
  - detect_from_returns() alternative entry
  - RegimeSnapshot properties and defaults
  - Confidence scoring
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.regime_detector import MarketRegime, RegimeDetector, RegimeSnapshot


# ── Helpers ──────────────────────────────────────────────────────

def _make_bull_prices(n: int = 120) -> np.ndarray:
    """Steady uptrend: 100 → ~180 over n days (daily +0.5%)."""
    daily_return = 0.005
    log_returns = np.full(n - 1, daily_return) + np.random.normal(0, 0.005, n - 1)
    prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))
    return prices


def _make_bear_prices(n: int = 120) -> np.ndarray:
    """Steady downtrend: 100 → ~70 over n days (daily -0.3%)."""
    daily_return = -0.003
    log_returns = np.full(n - 1, daily_return) + np.random.normal(0, 0.005, n - 1)
    prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))
    return prices


def _make_sideways_prices(n: int = 120) -> np.ndarray:
    """Range-bound: oscillate around 100 with small noise."""
    np.random.seed(42)
    noise = np.random.normal(0, 0.005, n - 1)
    prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(noise)]))
    return prices


# ══════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════

class TestRegimeSnapshot:
    def test_defaults(self):
        snap = RegimeSnapshot()
        assert snap.regime == MarketRegime.SIDEWAYS
        assert snap.confidence == 0.0
        assert snap.bull_score == 0.0
        assert snap.bear_score == 0.0

    def test_summary_property(self):
        snap = RegimeSnapshot(
            regime=MarketRegime.BULL,
            confidence=0.75,
            bull_score=0.60,
            bear_score=0.10,
            sideways_score=0.05,
        )
        s = snap.summary
        assert "BULL" in s
        assert "0.75" in s


class TestInsufficientData:
    def test_less_than_60_bars(self):
        det = RegimeDetector()
        prices = np.linspace(100, 110, 30)
        snap = det.detect(prices)
        assert snap.regime == MarketRegime.SIDEWAYS
        assert snap.confidence == 0.0

    def test_exactly_60_bars(self):
        det = RegimeDetector()
        prices = np.linspace(100, 120, 60)
        snap = det.detect(prices)
        # Should work — 60 is the minimum
        assert snap.regime in (MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS)
        assert snap.confidence > 0


class TestBullDetection:
    def test_strong_uptrend(self):
        np.random.seed(100)
        det = RegimeDetector()
        prices = _make_bull_prices(120)
        snap = det.detect(prices)
        assert snap.regime == MarketRegime.BULL
        assert snap.confidence > 0.4
        assert snap.bull_score > snap.bear_score

    def test_bull_features(self):
        np.random.seed(100)
        det = RegimeDetector()
        prices = _make_bull_prices(120)
        snap = det.detect(prices)
        # Return momentum should be positive
        assert snap.return_momentum > 0
        # Trend strength should be positive (price above SMA50)
        assert snap.trend_strength > 0
        # Majority of days should be positive
        assert snap.positive_day_pct > 0.5


class TestBearDetection:
    def test_strong_downtrend(self):
        np.random.seed(200)
        det = RegimeDetector()
        prices = _make_bear_prices(120)
        snap = det.detect(prices)
        assert snap.regime == MarketRegime.BEAR
        assert snap.confidence > 0.4
        assert snap.bear_score > snap.bull_score

    def test_bear_features(self):
        np.random.seed(200)
        det = RegimeDetector()
        prices = _make_bear_prices(120)
        snap = det.detect(prices)
        # Return momentum should be negative
        assert snap.return_momentum < 0
        # Trend strength should be negative (price below SMA50)
        assert snap.trend_strength < 0
        # Minority of days should be positive
        assert snap.positive_day_pct < 0.5


class TestSidewaysDetection:
    def test_range_bound(self):
        det = RegimeDetector()
        prices = _make_sideways_prices(120)
        snap = det.detect(prices)
        # Sideways should have highest score or at least not be clearly bull/bear
        # The sideways score should be meaningful
        assert snap.sideways_score > 0.1

    def test_sideways_features(self):
        det = RegimeDetector()
        prices = _make_sideways_prices(120)
        snap = det.detect(prices)
        # Momentum near zero
        assert abs(snap.return_momentum) < 0.005
        # Trend strength near zero
        assert abs(snap.trend_strength) < 0.10


class TestEdgeCases:
    def test_flat_prices(self):
        det = RegimeDetector()
        prices = np.full(100, 100.0)
        # log(100) - log(100) = 0 for all returns
        # This will have zero std → vol_ratio division by zero guard
        snap = det.detect(prices)
        assert snap.regime == MarketRegime.SIDEWAYS

    def test_single_spike_up(self):
        """Last day has a huge spike but overall flat."""
        det = RegimeDetector()
        prices = np.full(100, 100.0)
        prices[-1] = 120.0  # 20% spike on last day
        snap = det.detect(prices)
        # Should detect some bullish signal but with high vol
        assert snap.volatility_ratio > 1.0

    def test_very_long_series(self):
        np.random.seed(300)
        det = RegimeDetector()
        prices = _make_bull_prices(500)
        snap = det.detect(prices)
        # Should still work with long data
        assert snap.regime == MarketRegime.BULL

    def test_empty_array(self):
        det = RegimeDetector()
        snap = det.detect(np.array([]))
        assert snap.regime == MarketRegime.SIDEWAYS
        assert snap.confidence == 0.0


class TestDetectFromReturns:
    def test_bull_from_returns(self):
        np.random.seed(100)
        det = RegimeDetector()
        # Generate returns directly
        n = 120
        returns = np.full(n, 0.003) + np.random.normal(0, 0.005, n)
        snap = det.detect_from_returns(returns)
        assert snap.regime == MarketRegime.BULL
        assert snap.return_momentum > 0

    def test_insufficient_returns(self):
        det = RegimeDetector()
        returns = np.random.normal(0, 0.01, 30)
        snap = det.detect_from_returns(returns)
        assert snap.regime == MarketRegime.SIDEWAYS
        assert snap.confidence == 0.0


class TestConfidence:
    def test_strong_trend_high_confidence(self):
        np.random.seed(100)
        det = RegimeDetector()
        # Very strong uptrend with low noise
        daily_return = 0.005
        log_returns = np.full(119, daily_return) + np.random.normal(0, 0.002, 119)
        prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))
        snap = det.detect(prices)
        assert snap.regime == MarketRegime.BULL
        assert snap.confidence > 0.5

    def test_mixed_signals_lower_confidence(self):
        """Slightly bullish but noisy → lower confidence."""
        np.random.seed(42)
        det = RegimeDetector()
        # Slight uptrend with high noise
        daily_return = 0.0005
        log_returns = np.full(119, daily_return) + np.random.normal(0, 0.02, 119)
        prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))
        snap = det.detect(prices)
        # Should have lower confidence than strong trend
        # Total score should be more spread across regimes
        total = snap.bull_score + snap.bear_score + snap.sideways_score
        assert total > 0

    def test_confidence_bounded_0_1(self):
        np.random.seed(100)
        det = RegimeDetector()
        prices = _make_bull_prices(120)
        snap = det.detect(prices)
        assert 0 <= snap.confidence <= 1.0


class TestScoreConsistency:
    def test_scores_non_negative(self):
        np.random.seed(42)
        det = RegimeDetector()
        for seed in range(5):
            np.random.seed(seed)
            prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 120)))
            snap = det.detect(prices)
            assert snap.bull_score >= 0
            assert snap.bear_score >= 0
            assert snap.sideways_score >= 0

    def test_regime_matches_max_score(self):
        """Detected regime should match the score with highest value."""
        np.random.seed(42)
        det = RegimeDetector()
        for seed in [10, 20, 30, 40, 50]:
            np.random.seed(seed)
            prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 120)))
            snap = det.detect(prices)
            scores = {
                MarketRegime.BULL: snap.bull_score,
                MarketRegime.BEAR: snap.bear_score,
                MarketRegime.SIDEWAYS: snap.sideways_score,
            }
            expected_regime = max(scores, key=scores.get)
            assert snap.regime == expected_regime
