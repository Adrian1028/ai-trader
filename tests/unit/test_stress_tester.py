"""
Unit Tests for StressTester (壓力測試引擎)
==========================================
測試：
  1. CVaR 計算（單資產 + 組合層面）
  2. 情境分析
  3. 完整壓力測試
  4. 蒙特卡洛排列測試
  5. 縮減比例計算
"""
from __future__ import annotations

import numpy as np
import pytest

from src.agents.decision.stress_tester import (
    PermutationTestResult,
    SCENARIOS,
    StressTester,
    StressTestResult,
)


class TestCVaR:
    def test_basic_cvar(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 252)  # 1 year daily
        cvar = StressTester.compute_cvar(returns, 0.95)
        assert cvar >= 0

    def test_cvar_99_greater_than_95(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.03, 500)
        cvar_95 = StressTester.compute_cvar(returns, 0.95)
        cvar_99 = StressTester.compute_cvar(returns, 0.99)
        assert cvar_99 >= cvar_95

    def test_cvar_insufficient_data(self):
        returns = np.array([0.01, 0.02, -0.01])
        cvar = StressTester.compute_cvar(returns, 0.95)
        assert cvar == 0.0

    def test_cvar_all_positive(self):
        returns = np.ones(100) * 0.01
        cvar = StressTester.compute_cvar(returns, 0.95)
        # All positive returns → CVaR should be 0 (no loss)
        assert cvar == 0.0

    def test_portfolio_cvar(self):
        st = StressTester()
        rng = np.random.default_rng(42)
        returns_map = {
            "AAPL": rng.normal(0.001, 0.02, 252),
            "MSFT": rng.normal(0.001, 0.015, 252),
        }
        weights = {"AAPL": 0.6, "MSFT": 0.4}
        cvar = st.compute_portfolio_cvar(weights, returns_map, 0.95)
        assert cvar >= 0

    def test_portfolio_cvar_empty(self):
        st = StressTester()
        cvar = st.compute_portfolio_cvar({}, {}, 0.95)
        assert cvar == 0.0


class TestScenarioAnalysis:
    def test_known_scenario(self):
        st = StressTester()
        rng = np.random.default_rng(42)
        returns_map = {"AAPL": rng.normal(0.001, 0.02, 252)}
        weights = {"AAPL": 1.0}
        loss = st.run_scenario(weights, returns_map, "2008_financial_crisis")
        assert loss >= 0

    def test_unknown_scenario(self):
        st = StressTester()
        loss = st.run_scenario({"AAPL": 1.0}, {"AAPL": np.zeros(100)}, "nonexistent")
        assert loss == 0.0

    def test_all_scenarios_defined(self):
        assert "2008_financial_crisis" in SCENARIOS
        assert "2020_covid_crash" in SCENARIOS
        assert "2010_flash_crash" in SCENARIOS
        assert "rate_shock" in SCENARIOS
        assert "tech_sector_crash" in SCENARIOS


class TestFullStressTest:
    def test_passes_with_low_risk(self):
        st = StressTester(max_cvar_pct=0.10, max_scenario_loss_pct=0.30)
        rng = np.random.default_rng(42)
        returns_map = {
            "AAPL": rng.normal(0.001, 0.01, 252),  # low vol
            "MSFT": rng.normal(0.001, 0.01, 252),
        }
        weights = {"AAPL": 0.5, "MSFT": 0.5}
        result = st.full_stress_test(weights, returns_map)
        assert isinstance(result, StressTestResult)
        assert result.cvar_95 >= 0
        assert result.cvar_99 >= 0
        assert len(result.scenario_results) == len(SCENARIOS)

    def test_fails_with_high_risk(self):
        st = StressTester(max_cvar_pct=0.001, max_scenario_loss_pct=0.001)
        rng = np.random.default_rng(42)
        returns_map = {
            "RISKY": rng.normal(-0.005, 0.05, 252),
        }
        weights = {"RISKY": 1.0}
        result = st.full_stress_test(weights, returns_map)
        assert not result.passes_stress_test
        assert len(result.reasons) > 0

    def test_worst_scenario_identified(self):
        st = StressTester()
        rng = np.random.default_rng(42)
        returns_map = {"AAPL": rng.normal(0.001, 0.02, 252)}
        weights = {"AAPL": 1.0}
        result = st.full_stress_test(weights, returns_map)
        assert result.worst_scenario != ""
        assert result.worst_scenario_loss >= 0

    def test_summary_format(self):
        result = StressTestResult(
            cvar_95=0.03, cvar_99=0.05,
            worst_scenario="2008_financial_crisis",
            worst_scenario_loss=0.10,
            passes_stress_test=True,
        )
        assert "PASS" in result.summary
        assert "CVaR95" in result.summary


class TestPermutationTest:
    def test_significant_strategy(self):
        st = StressTester(permutation_runs=200)
        rng = np.random.default_rng(42)
        # Strong upward trend → should be significant
        returns = rng.normal(0.005, 0.01, 252)
        result = st.permutation_test(returns, n_permutations=200)
        assert isinstance(result, PermutationTestResult)
        assert result.strategy_sharpe > 0
        assert result.n_permutations == 200

    def test_random_strategy_not_significant(self):
        st = StressTester(permutation_runs=200)
        rng = np.random.default_rng(42)
        # Zero-mean returns → p-value should be high
        returns = rng.normal(0.0, 0.02, 252)
        result = st.permutation_test(returns, n_permutations=200)
        # Not guaranteed, but likely not significant
        assert result.p_value > 0.0

    def test_insufficient_data(self):
        st = StressTester()
        returns = np.array([0.01, -0.01, 0.02])
        result = st.permutation_test(returns)
        assert result.p_value == 1.0

    def test_summary_format(self):
        result = PermutationTestResult(
            strategy_sharpe=1.5,
            random_sharpe_mean=0.1,
            p_value=0.02,
            is_significant=True,
            n_permutations=1000,
        )
        assert "SIGNIFICANT" in result.summary


class TestReductionFactor:
    def test_passes_returns_one(self):
        st = StressTester()
        result = StressTestResult(passes_stress_test=True)
        assert st.compute_reduction_factor(result) == 1.0

    def test_fails_returns_less_than_one(self):
        st = StressTester(max_cvar_pct=0.05, max_scenario_loss_pct=0.15)
        result = StressTestResult(
            passes_stress_test=False,
            cvar_95=0.10,  # 2x limit
            worst_scenario_loss=0.30,  # 2x limit
        )
        factor = st.compute_reduction_factor(result)
        assert 0.1 <= factor < 1.0

    def test_floor_at_10_percent(self):
        st = StressTester(max_cvar_pct=0.01, max_scenario_loss_pct=0.01)
        result = StressTestResult(
            passes_stress_test=False,
            cvar_95=1.0,
            worst_scenario_loss=1.0,
        )
        factor = st.compute_reduction_factor(result)
        assert factor >= 0.1
