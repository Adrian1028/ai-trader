"""
Unit Tests for Portfolio Optimizer (投資組合優化)
=================================================
測試：
  1. BlackLittermanOptimizer — 均衡回報、觀點融合、權重正規化
  2. RiskParityAllocator — 風險平價、收斂性、單資產
  3. DynamicRebalancer — 偏離檢測、再平衡交易生成
"""
from __future__ import annotations

import numpy as np
import pytest

from src.agents.decision.portfolio_optimizer import (
    BlackLittermanOptimizer,
    DynamicRebalancer,
    RebalanceCheck,
    RiskParityAllocator,
)


# ══════════════════════════════════════════════════════════════════
# Black-Litterman
# ══════════════════════════════════════════════════════════════════

class TestBlackLitterman:
    def _simple_cov(self, n: int = 3) -> np.ndarray:
        """Create a simple positive-definite covariance matrix."""
        rng = np.random.default_rng(42)
        A = rng.normal(size=(n, n))
        return A @ A.T / n + np.eye(n) * 0.01

    def test_equilibrium_returns(self):
        bl = BlackLittermanOptimizer()
        cov = self._simple_cov(3)
        w = np.array([0.4, 0.3, 0.3])
        pi = bl.compute_equilibrium_returns(w, cov)
        assert pi.shape == (3,)
        assert not np.any(np.isnan(pi))

    def test_no_views_returns_market_weights(self):
        bl = BlackLittermanOptimizer()
        cov = self._simple_cov(3)
        w = np.array([0.5, 0.3, 0.2])
        result = bl.optimize(w, cov)
        np.testing.assert_array_almost_equal(result, w)

    def test_with_views_weights_sum_to_one(self):
        bl = BlackLittermanOptimizer()
        cov = self._simple_cov(3)
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        views = np.array([0.05, -0.02, 0.03])
        conf = np.array([0.8, 0.6, 0.7])
        result = bl.optimize(w, cov, views, conf)
        assert result.shape == (3,)
        assert abs(np.sum(result) - 1.0) < 1e-6
        assert np.all(result >= 0)  # long-only

    def test_strong_view_shifts_weight(self):
        bl = BlackLittermanOptimizer()
        # Use diagonal cov (uncorrelated) for predictable behavior
        cov = np.diag([0.04, 0.04, 0.04])
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        # Strong bullish view on asset 0, neutral on others
        views = np.array([0.10, 0.0, 0.0])
        conf = np.array([0.95, 0.5, 0.5])
        result = bl.optimize(w, cov, views, conf)
        # Asset 0 should get more weight than asset 2 (which has neutral view)
        assert result[0] > result[2]

    def test_empty_input(self):
        bl = BlackLittermanOptimizer()
        result = bl.optimize(np.array([]), np.array([]))
        assert len(result) == 0

    def test_single_asset(self):
        bl = BlackLittermanOptimizer()
        cov = np.array([[0.04]])
        w = np.array([1.0])
        result = bl.optimize(w, cov)
        np.testing.assert_array_almost_equal(result, [1.0])


# ══════════════════════════════════════════════════════════════════
# Risk Parity
# ══════════════════════════════════════════════════════════════════

class TestRiskParity:
    def _simple_cov(self, n: int = 3) -> np.ndarray:
        rng = np.random.default_rng(42)
        A = rng.normal(size=(n, n))
        return A @ A.T / n + np.eye(n) * 0.01

    def test_weights_sum_to_one(self):
        rp = RiskParityAllocator()
        cov = self._simple_cov(3)
        w = rp.optimize(cov)
        assert abs(np.sum(w) - 1.0) < 1e-6

    def test_all_positive(self):
        rp = RiskParityAllocator()
        cov = self._simple_cov(4)
        w = rp.optimize(cov)
        assert np.all(w > 0)

    def test_equal_vol_gives_equal_weights(self):
        """If all assets have same volatility and no correlation, weights should be equal."""
        rp = RiskParityAllocator()
        cov = np.eye(3) * 0.04  # identical, uncorrelated
        w = rp.optimize(cov)
        np.testing.assert_array_almost_equal(w, [1 / 3, 1 / 3, 1 / 3], decimal=4)

    def test_higher_vol_gets_lower_weight(self):
        rp = RiskParityAllocator()
        cov = np.diag([0.01, 0.04, 0.09])  # vol: 10%, 20%, 30%
        w = rp.optimize(cov)
        assert w[0] > w[1] > w[2]  # lower vol → higher weight

    def test_single_asset(self):
        rp = RiskParityAllocator()
        cov = np.array([[0.04]])
        w = rp.optimize(cov)
        np.testing.assert_array_almost_equal(w, [1.0])

    def test_empty(self):
        rp = RiskParityAllocator()
        w = rp.optimize(np.array([]).reshape(0, 0))
        assert len(w) == 0

    def test_risk_contributions_approximately_equal(self):
        rp = RiskParityAllocator()
        cov = self._simple_cov(3)
        w = rp.optimize(cov)
        rc = rp.compute_risk_contributions(w, cov)
        total_rc = np.sum(rc)
        if total_rc > 0:
            rc_pct = rc / total_rc
            # All risk contributions should be roughly equal
            assert np.max(np.abs(rc_pct - 1 / 3)) < 0.05


# ══════════════════════════════════════════════════════════════════
# Dynamic Rebalancer
# ══════════════════════════════════════════════════════════════════

class TestDynamicRebalancer:
    def test_no_drift_no_rebalance(self):
        rb = DynamicRebalancer(drift_threshold=0.05)
        current = {"AAPL": 0.33, "MSFT": 0.33, "GOOG": 0.34}
        target = {"AAPL": 0.33, "MSFT": 0.33, "GOOG": 0.34}
        check = rb.check_drift(current, target)
        assert not check.needs_rebalance

    def test_drift_triggers_rebalance(self):
        rb = DynamicRebalancer(drift_threshold=0.05)
        current = {"AAPL": 0.50, "MSFT": 0.25, "GOOG": 0.25}
        target = {"AAPL": 0.33, "MSFT": 0.33, "GOOG": 0.34}
        check = rb.check_drift(current, target)
        assert check.needs_rebalance
        assert abs(check.drifts["AAPL"]) > 0.05

    def test_compute_trades(self):
        rb = DynamicRebalancer(drift_threshold=0.05, min_trade_value=10.0)
        current = {"AAPL": 0.50, "MSFT": 0.25, "GOOG": 0.25}
        target = {"AAPL": 0.33, "MSFT": 0.33, "GOOG": 0.34}
        price_map = {"AAPL": 150.0, "MSFT": 300.0, "GOOG": 100.0}
        trades = rb.compute_rebalance_trades(current, target, 10000.0, price_map)
        assert len(trades) > 0
        # AAPL overweight → should have a SELL
        aapl_trades = [t for t in trades if t["ticker"] == "AAPL"]
        assert len(aapl_trades) == 1
        assert aapl_trades[0]["direction"] == "SELL"

    def test_sells_before_buys(self):
        rb = DynamicRebalancer(drift_threshold=0.05, min_trade_value=10.0)
        current = {"AAPL": 0.60, "MSFT": 0.10, "GOOG": 0.30}
        target = {"AAPL": 0.33, "MSFT": 0.33, "GOOG": 0.34}
        price_map = {"AAPL": 150.0, "MSFT": 300.0, "GOOG": 100.0}
        trades = rb.compute_rebalance_trades(current, target, 10000.0, price_map)
        if len(trades) >= 2:
            # SELLs should come before BUYs
            sell_indices = [i for i, t in enumerate(trades) if t["direction"] == "SELL"]
            buy_indices = [i for i, t in enumerate(trades) if t["direction"] == "BUY"]
            if sell_indices and buy_indices:
                assert max(sell_indices) < min(buy_indices)

    def test_skip_tiny_trades(self):
        rb = DynamicRebalancer(drift_threshold=0.01, min_trade_value=100.0)
        current = {"AAPL": 0.34}
        target = {"AAPL": 0.33}
        price_map = {"AAPL": 150.0}
        trades = rb.compute_rebalance_trades(current, target, 1000.0, price_map)
        # 1% of 1000 = 10, below min_trade_value 100
        assert len(trades) == 0

    def test_new_ticker_in_target(self):
        rb = DynamicRebalancer(drift_threshold=0.05, min_trade_value=10.0)
        current = {"AAPL": 1.0}
        target = {"AAPL": 0.5, "MSFT": 0.5}
        price_map = {"AAPL": 150.0, "MSFT": 300.0}
        trades = rb.compute_rebalance_trades(current, target, 10000.0, price_map)
        msft_trades = [t for t in trades if t["ticker"] == "MSFT"]
        assert len(msft_trades) == 1
        assert msft_trades[0]["direction"] == "BUY"

    def test_rebalance_check_summary(self):
        check = RebalanceCheck(needs_rebalance=False)
        assert "No rebalance" in check.summary
        check2 = RebalanceCheck(
            needs_rebalance=True,
            max_drift=0.12,
            max_drift_ticker="AAPL",
            trades=[{"ticker": "AAPL"}],
        )
        assert "12" in check2.summary
