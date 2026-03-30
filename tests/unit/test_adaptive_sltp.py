"""
Unit tests for Phase 3: Adaptive ATR-Based Stop-Loss / Take-Profit
===================================================================
Tests cover:
  - _adaptive_atr_multiples(): dynamic SL/TP multiplier computation
  - Volatility regime scaling (high vol → wider stops)
  - Confidence-based adjustments
  - MTF confluence bonus (all aligned → wider TP)
  - Safety bounds and minimum R:R enforcement
  - Integration with evaluate() envelope
"""
from __future__ import annotations

import numpy as np
import pytest

from src.agents.decision.risk import (
    PortfolioRiskState,
    RiskAgent,
    RiskVerdict,
)


@pytest.fixture
def agent():
    return RiskAgent(
        max_single_position_pct=0.05,
        max_var_pct_of_nav=0.10,
        atr_stop_multiplier=2.0,
        atr_tp_multiplier=3.0,
    )


@pytest.fixture
def portfolio():
    return PortfolioRiskState(
        total_nav=10_000.0,
        invested_value=3_000.0,
        cash=7_000.0,
        exposure_pct=0.30,
        positions={"AAPL": 1500, "MSFT": 1500},
    )


@pytest.fixture
def returns():
    rng = np.random.default_rng(42)
    return rng.normal(0.0006, 0.013, 250)


# ══════════════════════════════════════════════════════════════
# Volatility regime tests
# ══════════════════════════════════════════════════════════════

class TestVolatilityRegime:
    def test_high_vol_widens_stop_loss(self, agent):
        """High ATR% → wider SL to avoid shakeout."""
        sl_high, _ = agent._adaptive_atr_multiples(atr_pct=4.5, confidence=0.6)
        sl_low, _ = agent._adaptive_atr_multiples(atr_pct=1.0, confidence=0.6)
        assert sl_high > sl_low

    def test_high_vol_widens_take_profit(self, agent):
        """High ATR% → wider TP (bigger moves expected)."""
        _, tp_high = agent._adaptive_atr_multiples(atr_pct=4.5, confidence=0.6)
        _, tp_low = agent._adaptive_atr_multiples(atr_pct=1.0, confidence=0.6)
        assert tp_high > tp_low

    def test_mid_vol_interpolated(self, agent):
        """Mid-range ATR% → factors between extremes."""
        sl_mid, tp_mid = agent._adaptive_atr_multiples(atr_pct=2.5, confidence=0.6)
        sl_low, tp_low = agent._adaptive_atr_multiples(atr_pct=1.0, confidence=0.6)
        sl_high, tp_high = agent._adaptive_atr_multiples(atr_pct=4.5, confidence=0.6)
        assert sl_low <= sl_mid <= sl_high
        assert tp_low <= tp_mid <= tp_high


# ══════════════════════════════════════════════════════════════
# Confidence tests
# ══════════════════════════════════════════════════════════════

class TestConfidenceAdjustment:
    def test_high_confidence_tighter_sl(self, agent):
        """High confidence → tighter SL (more conviction)."""
        sl_high_conf, _ = agent._adaptive_atr_multiples(atr_pct=2.0, confidence=0.95)
        sl_low_conf, _ = agent._adaptive_atr_multiples(atr_pct=2.0, confidence=0.2)
        assert sl_high_conf < sl_low_conf

    def test_high_confidence_wider_tp(self, agent):
        """High confidence → wider TP (let profits run)."""
        _, tp_high_conf = agent._adaptive_atr_multiples(atr_pct=2.0, confidence=0.95)
        _, tp_low_conf = agent._adaptive_atr_multiples(atr_pct=2.0, confidence=0.2)
        assert tp_high_conf > tp_low_conf

    def test_high_confidence_better_rr_ratio(self, agent):
        """High confidence should give better risk/reward ratio."""
        sl_hi, tp_hi = agent._adaptive_atr_multiples(atr_pct=2.0, confidence=0.95)
        sl_lo, tp_lo = agent._adaptive_atr_multiples(atr_pct=2.0, confidence=0.2)
        rr_hi = tp_hi / sl_hi
        rr_lo = tp_lo / sl_lo
        assert rr_hi > rr_lo


# ══════════════════════════════════════════════════════════════
# MTF confluence tests
# ══════════════════════════════════════════════════════════════

class TestMTFConfluence:
    def test_full_alignment_wider_tp(self, agent):
        """Three timeframes aligned → wider TP (trend may extend)."""
        _, tp_aligned = agent._adaptive_atr_multiples(
            atr_pct=2.0, confidence=0.7, confluence_score=20,
        )
        _, tp_divergent = agent._adaptive_atr_multiples(
            atr_pct=2.0, confidence=0.7, confluence_score=-15,
        )
        assert tp_aligned > tp_divergent

    def test_divergence_tighter_tp(self, agent):
        """Timeframe divergence → tighter TP (uncertainty high)."""
        _, tp_div = agent._adaptive_atr_multiples(
            atr_pct=2.0, confidence=0.7, confluence_score=-15,
        )
        _, tp_neutral = agent._adaptive_atr_multiples(
            atr_pct=2.0, confidence=0.7, confluence_score=0,
        )
        assert tp_div < tp_neutral

    def test_confluence_doesnt_affect_sl(self, agent):
        """MTF confluence should NOT affect stop-loss (SL based on vol + confidence)."""
        sl_aligned, _ = agent._adaptive_atr_multiples(
            atr_pct=2.0, confidence=0.7, confluence_score=20,
        )
        sl_divergent, _ = agent._adaptive_atr_multiples(
            atr_pct=2.0, confidence=0.7, confluence_score=-15,
        )
        # mtf_sl_factor is always 1.0 in both cases
        assert sl_aligned == sl_divergent


# ══════════════════════════════════════════════════════════════
# Safety bounds tests
# ══════════════════════════════════════════════════════════════

class TestSafetyBounds:
    def test_sl_minimum_bound(self, agent):
        """SL multiplier never goes below 1.0×ATR."""
        sl, _ = agent._adaptive_atr_multiples(atr_pct=0.5, confidence=0.99)
        assert sl >= 1.0

    def test_sl_maximum_bound(self, agent):
        """SL multiplier never exceeds 4.0×ATR."""
        sl, _ = agent._adaptive_atr_multiples(atr_pct=10.0, confidence=0.1)
        assert sl <= 4.0

    def test_tp_minimum_bound(self, agent):
        """TP multiplier never goes below 1.5×ATR."""
        _, tp = agent._adaptive_atr_multiples(atr_pct=0.5, confidence=0.1, confluence_score=-15)
        assert tp >= 1.5

    def test_tp_maximum_bound(self, agent):
        """TP multiplier never exceeds 6.0×ATR."""
        _, tp = agent._adaptive_atr_multiples(atr_pct=5.0, confidence=0.99, confluence_score=20)
        assert tp <= 6.0

    def test_minimum_rr_ratio(self, agent):
        """Risk/reward ratio must be at least 1.5:1."""
        # Test across many combinations
        for atr_pct in [0.5, 1.5, 2.5, 3.5, 5.0]:
            for conf in [0.1, 0.4, 0.7, 0.95]:
                for mtf in [-15, 0, 20]:
                    sl, tp = agent._adaptive_atr_multiples(
                        atr_pct=atr_pct, confidence=conf, confluence_score=mtf,
                    )
                    rr = tp / sl
                    assert rr >= 1.5, (
                        f"R:R={rr:.2f} < 1.5 for atr={atr_pct}, "
                        f"conf={conf}, mtf={mtf}"
                    )


# ══════════════════════════════════════════════════════════════
# Integration with evaluate()
# ══════════════════════════════════════════════════════════════

class TestEvaluateIntegration:
    def test_envelope_has_adaptive_sltp(self, agent, portfolio, returns):
        envelope = agent.evaluate(
            direction=1, current_price=150.0, returns=returns,
            atr=3.0, confidence=0.8, portfolio=portfolio,
            confluence_score=20.0,
        )
        if envelope.verdict != RiskVerdict.REJECTED:
            assert envelope.stop_loss_price > 0
            assert envelope.take_profit_price > 0
            assert envelope.risk_reward_ratio >= 1.5

    def test_long_sl_below_price(self, agent, portfolio, returns):
        envelope = agent.evaluate(
            direction=1, current_price=150.0, returns=returns,
            atr=3.0, confidence=0.7, portfolio=portfolio,
        )
        if envelope.stop_loss_price > 0:
            assert envelope.stop_loss_price < 150.0
            assert envelope.take_profit_price > 150.0

    def test_sell_sl_above_price(self, agent, portfolio, returns):
        envelope = agent.evaluate(
            direction=-1, current_price=150.0, returns=returns,
            atr=3.0, confidence=0.7, portfolio=portfolio,
        )
        if envelope.stop_loss_price > 0:
            assert envelope.stop_loss_price > 150.0
            assert envelope.take_profit_price < 150.0

    def test_high_vol_wider_stops_in_envelope(self, agent, portfolio, returns):
        """With high ATR, stops should be further from current price."""
        env_normal = agent.evaluate(
            direction=1, current_price=150.0, returns=returns,
            atr=2.0, confidence=0.7, portfolio=portfolio,
        )
        env_high_vol = agent.evaluate(
            direction=1, current_price=150.0, returns=returns,
            atr=8.0, confidence=0.7, portfolio=portfolio,
        )
        if env_normal.stop_loss_price > 0 and env_high_vol.stop_loss_price > 0:
            sl_dist_normal = 150.0 - env_normal.stop_loss_price
            sl_dist_high = 150.0 - env_high_vol.stop_loss_price
            assert sl_dist_high > sl_dist_normal

    def test_no_atr_skips_sltp(self, agent, portfolio, returns):
        """When ATR is None, SL/TP should be 0."""
        envelope = agent.evaluate(
            direction=1, current_price=150.0, returns=returns,
            atr=None, confidence=0.7, portfolio=portfolio,
        )
        assert envelope.stop_loss_price == 0
        assert envelope.take_profit_price == 0
