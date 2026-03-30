"""
Unit tests for Phase 2: Adaptive Kelly Criterion Position Sizing
=================================================================
Tests cover:
  - _adaptive_kelly_scale(): dynamic scaling across 4 dimensions
  - Integration with evaluate(): adaptive_kelly_fraction in envelope
  - Edge cases: zero NAV, extreme volatility, full divergence
"""
from __future__ import annotations

import numpy as np
import pytest

from src.agents.decision.risk import (
    PortfolioRiskState,
    RiskAgent,
    RiskEnvelope,
    RiskVerdict,
    _KELLY_CEIL,
    _KELLY_FLOOR,
)


@pytest.fixture
def agent():
    return RiskAgent(
        max_single_position_pct=0.05,
        max_var_pct_of_nav=0.10,
    )


@pytest.fixture
def portfolio():
    return PortfolioRiskState(
        total_nav=10_000.0,
        invested_value=3_000.0,
        cash=7_000.0,
        exposure_pct=0.30,
        positions={"AAPL": 1500, "MSFT": 1500},
        max_single_position_pct=0.15,
    )


@pytest.fixture
def returns():
    """Realistic daily returns (250 days, ~15% annual, ~20% vol)."""
    rng = np.random.default_rng(42)
    return rng.normal(0.0006, 0.013, 250)


# ══════════════════════════════════════════════════════════════
# _adaptive_kelly_scale tests
# ══════════════════════════════════════════════════════════════

class TestAdaptiveKellyScale:
    def test_high_confidence_high_scale(self, agent, portfolio):
        scale = agent._adaptive_kelly_scale(
            confidence=0.95, atr_pct=1.0, portfolio=portfolio, confluence_score=20,
        )
        assert scale > 0.45, f"High confidence should give high scale, got {scale}"

    def test_low_confidence_low_scale(self, agent, portfolio):
        scale = agent._adaptive_kelly_scale(
            confidence=0.1, atr_pct=4.0, portfolio=portfolio, confluence_score=-15,
        )
        assert scale < 0.35, f"Low confidence should give low scale, got {scale}"

    def test_scale_within_bounds(self, agent, portfolio):
        """Scale must always be in [KELLY_FLOOR, KELLY_CEIL]."""
        test_cases = [
            (1.0, 0.5, 20),    # best case
            (0.0, 5.0, -15),   # worst case
            (0.5, 2.0, 0),     # average
        ]
        for conf, atr_pct, confluence in test_cases:
            scale = agent._adaptive_kelly_scale(
                confidence=conf, atr_pct=atr_pct,
                portfolio=portfolio, confluence_score=confluence,
            )
            assert _KELLY_FLOOR <= scale <= _KELLY_CEIL, (
                f"Scale {scale} out of bounds for conf={conf}, "
                f"atr={atr_pct}, confluence={confluence}"
            )

    def test_high_volatility_reduces_scale(self, agent, portfolio):
        low_vol = agent._adaptive_kelly_scale(
            confidence=0.7, atr_pct=1.0, portfolio=portfolio,
        )
        high_vol = agent._adaptive_kelly_scale(
            confidence=0.7, atr_pct=4.5, portfolio=portfolio,
        )
        assert low_vol > high_vol, (
            f"Low vol ({low_vol}) should give higher scale than high vol ({high_vol})"
        )

    def test_confluence_boosts_scale(self, agent, portfolio):
        aligned = agent._adaptive_kelly_scale(
            confidence=0.7, atr_pct=2.0, portfolio=portfolio, confluence_score=20,
        )
        divergent = agent._adaptive_kelly_scale(
            confidence=0.7, atr_pct=2.0, portfolio=portfolio, confluence_score=-15,
        )
        assert aligned > divergent, (
            f"Aligned ({aligned}) should give higher scale than divergent ({divergent})"
        )

    def test_high_concentration_reduces_scale(self, agent):
        """High exposure + high concentration → drawdown protection kicks in."""
        risky_portfolio = PortfolioRiskState(
            total_nav=10_000.0,
            invested_value=9_000.0,
            cash=1_000.0,
            exposure_pct=0.90,
            positions={"AAPL": 9000},  # all in one stock
            max_single_position_pct=0.90,
        )
        safe_portfolio = PortfolioRiskState(
            total_nav=10_000.0,
            invested_value=3_000.0,
            cash=7_000.0,
            exposure_pct=0.30,
            positions={"AAPL": 1000, "MSFT": 1000, "GOOG": 1000},
        )
        risky_scale = agent._adaptive_kelly_scale(
            confidence=0.7, atr_pct=2.0, portfolio=risky_portfolio,
        )
        safe_scale = agent._adaptive_kelly_scale(
            confidence=0.7, atr_pct=2.0, portfolio=safe_portfolio,
        )
        assert risky_scale < safe_scale, (
            f"Risky portfolio ({risky_scale}) should have lower scale "
            f"than safe portfolio ({safe_scale})"
        )


# ══════════════════════════════════════════════════════════════
# evaluate() integration tests
# ══════════════════════════════════════════════════════════════

class TestEvaluateWithAdaptiveKelly:
    def test_envelope_has_adaptive_fields(self, agent, portfolio, returns):
        envelope = agent.evaluate(
            direction=1, current_price=150.0, returns=returns,
            atr=3.0, confidence=0.8, portfolio=portfolio,
            confluence_score=20.0,
        )
        assert envelope.kelly_scale_factor > 0
        assert envelope.adaptive_kelly_fraction >= 0
        assert envelope.kelly_fraction >= 0
        # adaptive = raw * scale
        assert abs(
            envelope.adaptive_kelly_fraction
            - envelope.kelly_fraction * envelope.kelly_scale_factor
        ) < 1e-10

    def test_high_confidence_larger_position(self, agent, portfolio, returns):
        high_conf = agent.evaluate(
            direction=1, current_price=150.0, returns=returns,
            atr=3.0, confidence=0.95, portfolio=portfolio,
            confluence_score=20.0,
        )
        low_conf = agent.evaluate(
            direction=1, current_price=150.0, returns=returns,
            atr=3.0, confidence=0.3, portfolio=portfolio,
            confluence_score=-15.0,
        )
        if high_conf.verdict != RiskVerdict.REJECTED and low_conf.verdict != RiskVerdict.REJECTED:
            assert high_conf.suggested_quantity >= low_conf.suggested_quantity

    def test_backward_compat_half_kelly(self, agent, portfolio, returns):
        """half_kelly_fraction is still populated for backward compat."""
        envelope = agent.evaluate(
            direction=1, current_price=150.0, returns=returns,
            atr=3.0, confidence=0.7, portfolio=portfolio,
        )
        assert envelope.half_kelly_fraction == envelope.kelly_fraction / 2.0

    def test_confluence_in_reason_string(self, agent, portfolio, returns):
        envelope = agent.evaluate(
            direction=1, current_price=150.0, returns=returns,
            atr=3.0, confidence=0.7, portfolio=portfolio,
            confluence_score=20.0,
        )
        if envelope.verdict != RiskVerdict.REJECTED:
            assert "confluence" in envelope.reason.lower()

    def test_zero_confluence_default(self, agent, portfolio, returns):
        """When confluence_score not provided, defaults to 0."""
        envelope = agent.evaluate(
            direction=1, current_price=150.0, returns=returns,
            atr=3.0, confidence=0.7, portfolio=portfolio,
        )
        # Should still work (confluence defaults to 0.0)
        assert envelope.kelly_scale_factor > 0

    def test_sell_direction(self, agent, portfolio, returns):
        envelope = agent.evaluate(
            direction=-1, current_price=150.0, returns=returns,
            atr=3.0, confidence=0.8, portfolio=portfolio,
        )
        # Sell should still compute adaptive Kelly
        assert envelope.kelly_scale_factor > 0


# ══════════════════════════════════════════════════════════════
# Edge cases
# ══════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_zero_nav_portfolio(self, agent):
        empty = PortfolioRiskState(total_nav=0, cash=0)
        scale = agent._adaptive_kelly_scale(
            confidence=0.5, atr_pct=2.0, portfolio=empty,
        )
        assert _KELLY_FLOOR <= scale <= _KELLY_CEIL

    def test_extreme_atr(self, agent, portfolio):
        scale = agent._adaptive_kelly_scale(
            confidence=0.5, atr_pct=50.0, portfolio=portfolio,
        )
        assert scale <= _KELLY_CEIL

    def test_negative_confluence(self, agent, portfolio):
        scale = agent._adaptive_kelly_scale(
            confidence=0.5, atr_pct=2.0, portfolio=portfolio, confluence_score=-15,
        )
        assert scale >= _KELLY_FLOOR
