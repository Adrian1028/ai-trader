"""
Unit Tests for InsiderTradingAgent (內部人交易分析代理)
======================================================
測試：
  1. 特徵提取
  2. UK 股票跳過
  3. OPRO 介面
  4. End-to-End
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.agents.intelligence.insider_agent import InsiderTradingAgent
from src.core.base_agent import AnalysisSignal, SignalDirection
from src.data.providers.sec_edgar import SECEdgarProvider


def _make_agent() -> InsiderTradingAgent:
    provider = SECEdgarProvider()
    return InsiderTradingAgent(provider)


class TestExtractFeatures:
    def test_high_activity(self):
        agent = _make_agent()
        summary = {"recent_form4_count": 15, "activity_level": "high"}
        transactions = [
            {"filing_date": "2026-03-01", "form": "4"},
            {"filing_date": "2026-03-01", "form": "4"},
            {"filing_date": "2026-03-01", "form": "4"},
        ]
        metrics = agent._extract_features(summary, transactions)
        assert metrics["intensity_score"] > 0
        assert metrics["cluster_score"] > 0

    def test_no_activity(self):
        agent = _make_agent()
        summary = {"recent_form4_count": 0, "activity_level": "none"}
        metrics = agent._extract_features(summary, [])
        assert metrics["intensity_score"] == 0
        assert metrics["cluster_score"] == 0


class TestUKSkip:
    @pytest.mark.asyncio
    async def test_uk_stock_neutral(self):
        agent = _make_agent()
        result = await agent.analyse({"ticker": "BARC.L"})
        assert result.direction == SignalDirection.NEUTRAL
        assert "not applicable" in result.reasoning.lower()


class TestOPRO:
    def test_weight_update(self):
        agent = _make_agent()
        agent.update_weights_from_opro({"activity_intensity": 0.6})
        assert agent.dynamic_weights["activity_intensity"] == pytest.approx(0.6)

    def test_ignores_unknown(self):
        agent = _make_agent()
        agent.update_weights_from_opro({"unknown": 0.5})
        assert "unknown" not in agent.dynamic_weights


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_high_insider_activity(self):
        agent = _make_agent()
        agent._sec.insider_summary = AsyncMock(return_value={
            "recent_form4_count": 12,
            "activity_level": "high",
            "ticker": "AAPL",
        })
        agent._sec.insider_transactions = AsyncMock(return_value=[
            {"filing_date": "2026-03-01", "form": "4"},
            {"filing_date": "2026-03-01", "form": "4"},
            {"filing_date": "2026-03-02", "form": "4"},
        ])

        result = await agent.analyse({"ticker": "AAPL"})
        assert isinstance(result, AnalysisSignal)
        assert result.source == "insider"
        assert result.direction in (SignalDirection.BUY, SignalDirection.STRONG_BUY)

    @pytest.mark.asyncio
    async def test_no_data_neutral(self):
        agent = _make_agent()
        agent._sec.insider_summary = AsyncMock(return_value={
            "recent_form4_count": 0,
            "activity_level": "none",
        })
        agent._sec.insider_transactions = AsyncMock(return_value=[])

        result = await agent.analyse({"ticker": "UNKNOWN"})
        assert result.direction == SignalDirection.NEUTRAL
