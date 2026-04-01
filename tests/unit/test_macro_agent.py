"""
Unit Tests for MacroAgent (宏觀經濟分析代理)
=============================================
測試：
  1. 特徵提取（貨幣政策、通膨、經濟週期）
  2. OPRO 動態權重
  3. 方向映射
  4. End-to-End（mock FRED 數據）
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.agents.intelligence.macro_agent import MacroAgent
from src.core.base_agent import AnalysisSignal, SignalDirection
from src.data.providers.macro import MacroDataProvider


def _make_agent() -> MacroAgent:
    provider = MacroDataProvider(api_key="test")
    return MacroAgent(provider)


class TestExtractFeatures:
    def test_hiking_cycle_bearish(self):
        agent = _make_agent()
        fed = [{"date": "2026-03", "value": 5.5}, {"date": "2025-10", "value": 5.0}]
        metrics = agent._extract_features(fed, [], {}, None, [])
        assert metrics["monetary_score"] < 0

    def test_cutting_cycle_bullish(self):
        agent = _make_agent()
        fed = [{"date": "2026-03", "value": 4.0}, {"date": "2025-10", "value": 4.5}]
        metrics = agent._extract_features(fed, [], {}, None, [])
        assert metrics["monetary_score"] > 0

    def test_high_inflation_bearish(self):
        agent = _make_agent()
        cpi = [{"date": "2026-03", "value": 310}, {"date": "2025-10", "value": 300}]
        metrics = agent._extract_features([], cpi, {}, None, [])
        # ~3.3% over 6 months → ~6.6% annualized → high
        assert metrics["inflation_score"] < 0

    def test_inverted_yield_curve(self):
        agent = _make_agent()
        yields = {"10y": 3.5, "2y": 4.0, "3m": 4.2, "spread_10y_2y": -0.5}
        metrics = agent._extract_features([], [], yields, None, [])
        assert metrics["cycle_score"] < 0

    def test_high_vix_bearish(self):
        agent = _make_agent()
        metrics = agent._extract_features([], [], {}, 35.0, [])
        assert metrics["cycle_score"] < 0

    def test_no_data_neutral(self):
        agent = _make_agent()
        metrics = agent._extract_features([], [], {}, None, [])
        assert metrics.get("monetary_score", 0) == 0
        assert metrics.get("inflation_score", 0) == 0


class TestWeightedScore:
    def test_all_bullish(self):
        agent = _make_agent()
        metrics = {
            "monetary_score": 20.0,
            "inflation_score": 10.0,
            "cycle_score": 15.0,
        }
        score = agent._compute_weighted_score(metrics)
        assert score > 0

    def test_all_bearish(self):
        agent = _make_agent()
        metrics = {
            "monetary_score": -20.0,
            "inflation_score": -20.0,
            "cycle_score": -15.0,
        }
        score = agent._compute_weighted_score(metrics)
        assert score < 0


class TestOPRO:
    def test_weight_update(self):
        agent = _make_agent()
        agent.update_weights_from_opro({"monetary_policy": 0.6})
        assert agent.dynamic_weights["monetary_policy"] == pytest.approx(0.6)
        assert agent.dynamic_weights["inflation_pressure"] == pytest.approx(0.3)

    def test_ignores_unknown(self):
        agent = _make_agent()
        agent.update_weights_from_opro({"unknown": 0.99})
        assert "unknown" not in agent.dynamic_weights


class TestDirection:
    def test_strong_bullish(self):
        assert MacroAgent._score_to_direction(30) == SignalDirection.STRONG_BUY

    def test_bullish(self):
        assert MacroAgent._score_to_direction(12) == SignalDirection.BUY

    def test_neutral(self):
        assert MacroAgent._score_to_direction(3) == SignalDirection.NEUTRAL

    def test_bearish(self):
        assert MacroAgent._score_to_direction(-12) == SignalDirection.SELL


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_bullish_macro(self):
        agent = _make_agent()
        agent._macro.fed_funds_rate = AsyncMock(return_value=[
            {"date": "2026-03", "value": 4.0},
            {"date": "2025-10", "value": 4.5},
        ])
        agent._macro.cpi = AsyncMock(return_value=[
            {"date": "2026-03", "value": 303},
            {"date": "2025-10", "value": 300},
        ])
        agent._macro.treasury_yields = AsyncMock(return_value={
            "10y": 4.0, "2y": 3.5, "3m": 3.0, "spread_10y_2y": 0.5,
        })
        agent._macro.vix = AsyncMock(return_value=15.0)
        agent._macro.unemployment = AsyncMock(return_value=[
            {"date": "2026-03", "value": 3.5},
            {"date": "2025-10", "value": 3.8},
        ])

        result = await agent.analyse({"ticker": "AAPL"})
        assert isinstance(result, AnalysisSignal)
        assert result.source == "macro"
        assert result.direction in (SignalDirection.BUY, SignalDirection.STRONG_BUY)

    @pytest.mark.asyncio
    async def test_api_failure_neutral(self):
        agent = _make_agent()
        agent._macro.fed_funds_rate = AsyncMock(side_effect=Exception("API down"))
        agent._macro.cpi = AsyncMock(side_effect=Exception("API down"))
        agent._macro.treasury_yields = AsyncMock(side_effect=Exception("API down"))
        agent._macro.vix = AsyncMock(side_effect=Exception("API down"))
        agent._macro.unemployment = AsyncMock(side_effect=Exception("API down"))

        result = await agent.analyse({"ticker": "AAPL"})
        assert result.direction == SignalDirection.NEUTRAL
