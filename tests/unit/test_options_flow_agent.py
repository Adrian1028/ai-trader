"""
Unit Tests for OptionsFlowAgent (期權流量分析代理)
==================================================
測試：
  1. 特徵提取（異常活動、P/C Ratio、Smart Money）
  2. UK 股票跳過
  3. OPRO 介面
  4. End-to-End
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.agents.intelligence.options_flow_agent import OptionsFlowAgent
from src.core.base_agent import AnalysisSignal, SignalDirection
from src.data.providers.options_flow import OptionsFlowProvider


def _make_agent() -> OptionsFlowAgent:
    provider = OptionsFlowProvider(api_key="test")
    return OptionsFlowAgent(provider)


class TestExtractFeatures:
    def test_bullish_activity(self):
        agent = _make_agent()
        activity = [
            {"sentiment": "bullish", "premium": 50000},
            {"sentiment": "bullish", "premium": 30000},
            {"sentiment": "bearish", "premium": 10000},
        ]
        metrics = agent._extract_features(activity, {}, {})
        assert metrics["activity_score"] > 0

    def test_bearish_activity(self):
        agent = _make_agent()
        activity = [
            {"sentiment": "bearish", "premium": 50000},
            {"sentiment": "bearish", "premium": 30000},
            {"sentiment": "bullish", "premium": 10000},
        ]
        metrics = agent._extract_features(activity, {}, {})
        assert metrics["activity_score"] < 0

    def test_pcr_extreme_fear_contrarian_bullish(self):
        agent = _make_agent()
        pcr = {"ratio": 1.8, "put_volume": 180, "call_volume": 100}
        metrics = agent._extract_features([], pcr, {})
        assert metrics["pcr_score"] > 0  # Contrarian bullish

    def test_pcr_bullish_flow(self):
        agent = _make_agent()
        pcr = {"ratio": 0.6, "put_volume": 60, "call_volume": 100}
        metrics = agent._extract_features([], pcr, {})
        assert metrics["pcr_score"] > 0

    def test_smart_money_bullish(self):
        agent = _make_agent()
        smart = {
            "bullish_premium": 500000,
            "bearish_premium": 100000,
            "net_flow": 400000,
            "large_trades_count": 5,
        }
        metrics = agent._extract_features([], {}, smart)
        assert metrics["smart_score"] > 0

    def test_empty_data(self):
        agent = _make_agent()
        metrics = agent._extract_features([], {}, {})
        assert metrics["activity_score"] == 0
        assert metrics["pcr_score"] == 0
        assert metrics["smart_score"] == 0


class TestUKSkip:
    @pytest.mark.asyncio
    async def test_uk_stock_neutral(self):
        agent = _make_agent()
        result = await agent.analyse({"ticker": "BARC.L"})
        assert result.direction == SignalDirection.NEUTRAL


class TestOPRO:
    def test_weight_update(self):
        agent = _make_agent()
        agent.update_weights_from_opro({"unusual_activity": 0.5})
        assert agent.dynamic_weights["unusual_activity"] == pytest.approx(0.5)


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_bullish_flow(self):
        agent = _make_agent()
        agent._options.unusual_activity = AsyncMock(return_value=[
            {"sentiment": "bullish", "premium": 200000},
            {"sentiment": "bullish", "premium": 150000},
            {"sentiment": "neutral", "premium": 50000},
        ])
        agent._options.put_call_ratio = AsyncMock(return_value={
            "ratio": 0.6, "put_volume": 600, "call_volume": 1000,
        })
        agent._options.smart_money_flow = AsyncMock(return_value={
            "bullish_premium": 500000,
            "bearish_premium": 100000,
            "net_flow": 400000,
            "large_trades_count": 5,
        })

        result = await agent.analyse({"ticker": "AAPL"})
        assert isinstance(result, AnalysisSignal)
        assert result.source == "options_flow"
        assert result.direction in (SignalDirection.BUY, SignalDirection.STRONG_BUY)

    @pytest.mark.asyncio
    async def test_no_data_neutral(self):
        agent = _make_agent()
        agent._options.unusual_activity = AsyncMock(return_value=[])
        agent._options.put_call_ratio = AsyncMock(return_value={})
        agent._options.smart_money_flow = AsyncMock(return_value={})

        result = await agent.analyse({"ticker": "AAPL"})
        assert result.direction == SignalDirection.NEUTRAL
