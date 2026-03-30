"""
Unit tests for Phase 6: Correlation Filter
============================================
Tests cover:
  - Correlation matrix computation
  - Candidate checking (PASS / PENALIZE / BLOCK)
  - Batch proposal filtering
  - Intra-cycle cross-checking (prevent buying two correlated stocks)
  - Cluster detection
  - Portfolio correlation report
  - Edge cases (insufficient data, single stock, constant series)
"""
from __future__ import annotations

import numpy as np
import pytest
from dataclasses import dataclass, field
from unittest.mock import MagicMock

from src.agents.decision.correlation_filter import (
    CorrelationFilter,
    CorrelationCheckResult,
    CorrelationMatrix,
)


# ── Helpers ──────────────────────────────────────────────────────

def _correlated_returns(
    n: int = 100,
    correlation: float = 0.9,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two correlated return series."""
    rng = np.random.RandomState(seed)
    base = rng.normal(0, 0.01, n)
    noise = rng.normal(0, 0.01, n)
    a = base
    b = correlation * base + np.sqrt(1 - correlation ** 2) * noise
    return a, b


def _uncorrelated_returns(n: int = 100, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate two uncorrelated return series."""
    rng = np.random.RandomState(seed)
    a = rng.normal(0, 0.01, n)
    b = rng.normal(0, 0.01, n)
    return a, b


def _make_proposal(ticker: str, action: str = "BUY", qty: float = 10.0, price: float = 100.0):
    """Create a mock TradeProposal-like object."""
    from src.agents.decision.decision_fusion import TradeAction, TradeProposal
    from src.agents.decision.risk import RiskEnvelope

    action_enum = {"BUY": TradeAction.BUY, "SELL": TradeAction.SELL, "HOLD": TradeAction.HOLD}[action]
    return TradeProposal(
        action=action_enum,
        ticker=ticker,
        quantity=qty,
        estimated_value=qty * price,
        current_price=price,
        risk=RiskEnvelope(risk_reward_ratio=2.0),
        reasoning="Test proposal",
    )


# ══════════════════════════════════════════════════════════════════
# Tests: CorrelationMatrix
# ══════════════════════════════════════════════════════════════════

class TestCorrelationMatrix:
    def test_compute_two_correlated(self):
        a, b = _correlated_returns(100, correlation=0.9)
        returns_map = {"AAPL": a, "MSFT": b}
        cf = CorrelationFilter()
        matrix = cf.compute_matrix(returns_map)

        assert len(matrix.tickers) == 2
        corr = matrix.get_correlation("AAPL", "MSFT")
        assert corr is not None
        assert corr > 0.80  # should be close to 0.9

    def test_compute_two_uncorrelated(self):
        a, b = _uncorrelated_returns(100)
        returns_map = {"AAPL": a, "XOM": b}
        cf = CorrelationFilter()
        matrix = cf.compute_matrix(returns_map)

        corr = matrix.get_correlation("AAPL", "XOM")
        assert corr is not None
        assert abs(corr) < 0.40  # should be near zero

    def test_self_correlation_is_one(self):
        a, b = _correlated_returns(100, 0.5)
        returns_map = {"AAPL": a, "MSFT": b}
        cf = CorrelationFilter()
        matrix = cf.compute_matrix(returns_map)

        assert matrix.get_correlation("AAPL", "AAPL") == pytest.approx(1.0)

    def test_insufficient_data(self):
        returns_map = {"AAPL": np.random.normal(0, 0.01, 10)}
        cf = CorrelationFilter()
        matrix = cf.compute_matrix(returns_map)
        assert len(matrix.tickers) == 0  # below min_overlap

    def test_single_ticker(self):
        returns_map = {"AAPL": np.random.normal(0, 0.01, 100)}
        cf = CorrelationFilter()
        matrix = cf.compute_matrix(returns_map)
        # Can't compute pairwise correlation with just one
        assert len(matrix.tickers) <= 1

    def test_missing_ticker_returns_none(self):
        a, b = _correlated_returns(100)
        returns_map = {"AAPL": a, "MSFT": b}
        cf = CorrelationFilter()
        matrix = cf.compute_matrix(returns_map)
        assert matrix.get_correlation("AAPL", "TSLA") is None

    def test_summary_property(self):
        a, b = _correlated_returns(100)
        returns_map = {"AAPL": a, "MSFT": b}
        cf = CorrelationFilter()
        matrix = cf.compute_matrix(returns_map)
        s = matrix.summary
        assert "2 tickers" in s
        assert "1 pairs" in s

    def test_three_tickers(self):
        rng = np.random.RandomState(42)
        base = rng.normal(0, 0.01, 100)
        returns_map = {
            "AAPL": base + rng.normal(0, 0.002, 100),
            "MSFT": base + rng.normal(0, 0.002, 100),
            "XOM": rng.normal(0, 0.01, 100),  # uncorrelated
        }
        cf = CorrelationFilter()
        matrix = cf.compute_matrix(returns_map)
        assert len(matrix.tickers) == 3
        # AAPL-MSFT should be high, AAPL-XOM should be low
        corr_tech = matrix.get_correlation("AAPL", "MSFT")
        corr_cross = matrix.get_correlation("AAPL", "XOM")
        assert corr_tech > corr_cross


# ══════════════════════════════════════════════════════════════════
# Tests: check_candidate
# ══════════════════════════════════════════════════════════════════

class TestCheckCandidate:
    def test_no_holdings_always_pass(self):
        cf = CorrelationFilter()
        result = cf.check_candidate("AAPL", [], {})
        assert result.action == "PASS"
        assert not result.is_blocked

    def test_no_return_data_pass(self):
        cf = CorrelationFilter()
        result = cf.check_candidate("AAPL", ["MSFT"], {})
        assert result.action == "PASS"

    def test_high_corr_blocked(self):
        a, b = _correlated_returns(100, correlation=0.95)
        returns_map = {"MSFT": a, "AAPL": b}
        cf = CorrelationFilter(block_threshold=0.85)
        result = cf.check_candidate("AAPL", ["MSFT"], returns_map)
        assert result.is_blocked
        assert result.correlated_with == "MSFT"
        assert result.max_pairwise_corr > 0.85

    def test_medium_corr_penalized(self):
        a, b = _correlated_returns(100, correlation=0.78)
        returns_map = {"MSFT": a, "AAPL": b}
        cf = CorrelationFilter(block_threshold=0.85, penalize_threshold=0.70)
        result = cf.check_candidate("AAPL", ["MSFT"], returns_map)
        assert result.is_penalized
        assert result.max_pairwise_corr > 0.70
        assert result.max_pairwise_corr < 0.85

    def test_low_corr_pass(self):
        a, b = _uncorrelated_returns(100)
        returns_map = {"XOM": a, "AAPL": b}
        cf = CorrelationFilter()
        result = cf.check_candidate("AAPL", ["XOM"], returns_map)
        assert result.action == "PASS"
        assert result.max_pairwise_corr < 0.70

    def test_multiple_holdings(self):
        """Check candidate against multiple holdings, report max."""
        rng = np.random.RandomState(42)
        base = rng.normal(0, 0.01, 100)
        returns_map = {
            "AAPL": base,
            "MSFT": base + rng.normal(0, 0.002, 100),  # high corr
            "XOM": rng.normal(0, 0.01, 100),            # low corr
        }
        cf = CorrelationFilter()
        result = cf.check_candidate("AAPL", ["MSFT", "XOM"], returns_map)
        assert result.correlated_with == "MSFT"  # max corr should be MSFT


# ══════════════════════════════════════════════════════════════════
# Tests: filter_proposals (batch)
# ══════════════════════════════════════════════════════════════════

class TestFilterProposals:
    def test_no_proposals(self):
        cf = CorrelationFilter()
        result = cf.filter_proposals([], [], {})
        assert result == []

    def test_sell_always_passes(self):
        """SELL proposals should never be blocked by correlation."""
        a, b = _correlated_returns(100, 0.95)
        returns_map = {"AAPL": a, "MSFT": b}
        proposals = [_make_proposal("AAPL", action="SELL")]
        cf = CorrelationFilter()
        cf.filter_proposals(proposals, ["MSFT"], returns_map)
        from src.agents.decision.decision_fusion import TradeAction
        assert proposals[0].action == TradeAction.SELL  # unchanged

    def test_blocked_buy_becomes_hold(self):
        a, b = _correlated_returns(100, 0.95)
        returns_map = {"AAPL": a, "MSFT": b}
        proposals = [_make_proposal("AAPL", action="BUY")]
        cf = CorrelationFilter(block_threshold=0.85)
        cf.filter_proposals(proposals, ["MSFT"], returns_map)

        from src.agents.decision.decision_fusion import TradeAction
        assert proposals[0].action == TradeAction.HOLD
        assert proposals[0].quantity == 0.0
        assert "CORRELATION BLOCKED" in proposals[0].reasoning

    def test_penalized_buy_reduced(self):
        a, b = _correlated_returns(100, 0.78)
        returns_map = {"AAPL": a, "MSFT": b}
        proposals = [_make_proposal("AAPL", action="BUY", qty=10.0)]
        cf = CorrelationFilter(
            block_threshold=0.85,
            penalize_threshold=0.70,
            penalty_factor=0.50,
        )
        cf.filter_proposals(proposals, ["MSFT"], returns_map)

        from src.agents.decision.decision_fusion import TradeAction
        assert proposals[0].action == TradeAction.BUY  # still buys
        assert proposals[0].quantity == pytest.approx(5.0)  # reduced by 50%
        assert "CORRELATION PENALIZED" in proposals[0].reasoning

    def test_uncorrelated_buy_passes(self):
        a, b = _uncorrelated_returns(100)
        returns_map = {"AAPL": a, "XOM": b}
        proposals = [_make_proposal("AAPL", action="BUY", qty=10.0)]
        cf = CorrelationFilter()
        cf.filter_proposals(proposals, ["XOM"], returns_map)

        from src.agents.decision.decision_fusion import TradeAction
        assert proposals[0].action == TradeAction.BUY
        assert proposals[0].quantity == 10.0

    def test_intra_cycle_cross_check(self):
        """Two highly correlated BUY proposals in same cycle: second is blocked."""
        rng = np.random.RandomState(42)
        base = rng.normal(0, 0.01, 100)
        returns_map = {
            "AAPL": base,
            "MSFT": base + rng.normal(0, 0.001, 100),  # ~0.98 corr
            "XOM": rng.normal(0, 0.01, 100),
        }
        proposals = [
            _make_proposal("AAPL", action="BUY"),
            _make_proposal("MSFT", action="BUY"),  # should be caught
        ]
        cf = CorrelationFilter(block_threshold=0.85)
        cf.filter_proposals(proposals, [], returns_map)  # no existing holdings

        from src.agents.decision.decision_fusion import TradeAction
        # First BUY should pass (no holdings to correlate with)
        assert proposals[0].action == TradeAction.BUY
        # Second BUY should be blocked (correlated with first)
        assert proposals[1].action == TradeAction.HOLD
        assert "CORRELATION BLOCKED" in proposals[1].reasoning

    def test_mixed_actions(self):
        """Mix of BUY, SELL, HOLD — only BUY gets checked."""
        a, b = _correlated_returns(100, 0.95)
        c = np.random.RandomState(99).normal(0, 0.01, 100)
        returns_map = {"AAPL": a, "MSFT": b, "XOM": c}

        proposals = [
            _make_proposal("AAPL", action="BUY"),
            _make_proposal("MSFT", action="SELL"),
            _make_proposal("XOM", action="BUY"),
        ]
        cf = CorrelationFilter(block_threshold=0.85)
        cf.filter_proposals(proposals, ["AAPL"], returns_map)

        from src.agents.decision.decision_fusion import TradeAction
        # AAPL BUY: buying same stock we hold → self-check skipped, PASS
        # MSFT SELL: never checked
        assert proposals[1].action == TradeAction.SELL
        # XOM BUY: uncorrelated with AAPL → PASS
        assert proposals[2].action == TradeAction.BUY


# ══════════════════════════════════════════════════════════════════
# Tests: Cluster detection
# ══════════════════════════════════════════════════════════════════

class TestClusterDetection:
    def test_no_clusters_uncorrelated(self):
        rng = np.random.RandomState(42)
        returns_map = {
            "AAPL": rng.normal(0, 0.01, 100),
            "XOM": rng.normal(0, 0.01, 100),
            "BA": rng.normal(0, 0.01, 100),
        }
        cf = CorrelationFilter()
        report = cf.portfolio_correlation_report(list(returns_map.keys()), returns_map)
        assert report["risk_level"] in ("LOW", "MEDIUM")
        assert len(report["clusters"]) == 0

    def test_tech_cluster(self):
        """Three highly correlated tech stocks form a cluster."""
        rng = np.random.RandomState(42)
        base = rng.normal(0, 0.01, 100)
        returns_map = {
            "AAPL": base + rng.normal(0, 0.002, 100),
            "MSFT": base + rng.normal(0, 0.002, 100),
            "GOOGL": base + rng.normal(0, 0.002, 100),
            "XOM": rng.normal(0, 0.01, 100),  # uncorrelated
        }
        cf = CorrelationFilter(penalize_threshold=0.70)
        report = cf.portfolio_correlation_report(list(returns_map.keys()), returns_map)

        # Should detect tech cluster
        clusters = report["clusters"]
        # At least one cluster with AAPL, MSFT, GOOGL
        has_tech_cluster = any(
            "AAPL" in c and "MSFT" in c
            for c in clusters
        )
        assert has_tech_cluster

    def test_report_empty_portfolio(self):
        cf = CorrelationFilter()
        report = cf.portfolio_correlation_report([], {})
        assert report["risk_level"] == "LOW"
        assert report["avg_correlation"] == 0.0

    def test_report_single_holding(self):
        returns_map = {"AAPL": np.random.normal(0, 0.01, 100)}
        cf = CorrelationFilter()
        report = cf.portfolio_correlation_report(["AAPL"], returns_map)
        assert report["risk_level"] == "LOW"


# ══════════════════════════════════════════════════════════════════
# Tests: Edge cases
# ══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_constant_series(self):
        """Constant return series → zero correlation."""
        returns_map = {
            "AAPL": np.zeros(100),
            "MSFT": np.random.normal(0, 0.01, 100),
        }
        cf = CorrelationFilter()
        result = cf.check_candidate("AAPL", ["MSFT"], returns_map)
        assert result.action == "PASS"

    def test_identical_series_blocked(self):
        """Exact same returns → correlation 1.0 → blocked."""
        returns = np.random.RandomState(42).normal(0, 0.01, 100)
        returns_map = {"AAPL": returns, "MSFT": returns.copy()}
        cf = CorrelationFilter(block_threshold=0.85)
        result = cf.check_candidate("AAPL", ["MSFT"], returns_map)
        assert result.is_blocked
        assert result.max_pairwise_corr == pytest.approx(1.0)

    def test_negatively_correlated_pass(self):
        """Negatively correlated → great for diversification, PASS."""
        a, _ = _correlated_returns(100, 0.0)
        returns_map = {"AAPL": a, "SQQQ": -a}
        cf = CorrelationFilter()
        result = cf.check_candidate("SQQQ", ["AAPL"], returns_map)
        assert result.action == "PASS"
        assert result.max_pairwise_corr < 0  # negative correlation

    def test_custom_thresholds(self):
        """Custom thresholds work correctly."""
        a, b = _correlated_returns(100, 0.80)
        returns_map = {"AAPL": a, "MSFT": b}
        # Very strict: penalize at 0.50
        cf = CorrelationFilter(penalize_threshold=0.50, block_threshold=0.90)
        result = cf.check_candidate("AAPL", ["MSFT"], returns_map)
        assert result.is_penalized or result.is_blocked

    def test_different_length_returns(self):
        """Different length return series should still work (aligned)."""
        rng = np.random.RandomState(42)
        returns_map = {
            "AAPL": rng.normal(0, 0.01, 200),   # 200 days
            "MSFT": rng.normal(0, 0.01, 80),    # 80 days
        }
        cf = CorrelationFilter()
        result = cf.check_candidate("AAPL", ["MSFT"], returns_map)
        assert result.action == "PASS"  # uncorrelated

    def test_penalty_factor_applied(self):
        """Verify penalty factor is correctly applied to quantity."""
        a, b = _correlated_returns(100, 0.78)
        returns_map = {"AAPL": a, "MSFT": b}
        proposals = [_make_proposal("AAPL", action="BUY", qty=20.0)]
        cf = CorrelationFilter(
            penalize_threshold=0.70,
            block_threshold=0.85,
            penalty_factor=0.30,
        )
        cf.filter_proposals(proposals, ["MSFT"], returns_map)
        assert proposals[0].quantity == pytest.approx(6.0)  # 20 * 0.30
