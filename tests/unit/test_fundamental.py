"""
Unit Tests for FundamentalAgent (基本面分析代理)
================================================
測試：
  1. 特徵提取 (metrics extraction)
  2. OPRO 動態權重計算 (weighted score)
  3. 補充維度分數 (bonus score)
  4. 權重更新介面 (OPRO interface)
  5. 安全值轉換 (safe_float)
  6. 分數到方向的映射 (score_to_direction)
  7. 完整分析流程 (end-to-end with mocked data)
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.intelligence.fundamental import FundamentalAgent
from src.core.base_agent import AnalysisSignal, SignalDirection
from src.data.providers.alpha_vantage import AlphaVantageProvider


def _make_agent() -> FundamentalAgent:
    av = AlphaVantageProvider(api_key="test")
    return FundamentalAgent(alpha_vantage=av)


# ═══════════════════════════════════════════════════════════════════
# 特徵提取
# ═══════════════════════════════════════════════════════════════════

class TestExtractMetrics:
    def test_parses_numeric_overview(self):
        agent = _make_agent()
        overview = {
            "PERatio": 22.5,
            "ForwardPE": 18.3,
            "PriceToBookRatio": 3.2,
            "EVToEBITDA": 14.5,
            "ReturnOnEquityTTM": 0.28,
            "ProfitMargin": 0.256,
            "EPS": 6.43,
            "QuarterlyRevenueGrowthYOY": 0.12,
            "DividendYield": 0.005,
        }
        m = agent._extract_metrics(overview)
        assert m["pe_ratio"] == pytest.approx(22.5)
        assert m["forward_pe"] == pytest.approx(18.3)
        assert m["roe"] == pytest.approx(0.28)
        assert m["profit_margin"] == pytest.approx(0.256)
        assert m["revenue_growth"] == pytest.approx(0.12)

    def test_handles_missing_fields(self):
        agent = _make_agent()
        m = agent._extract_metrics({})
        assert m["pe_ratio"] == 0.0
        assert m["roe"] == 0.0
        assert m["revenue_growth"] == 0.0

    def test_handles_string_none_values(self):
        agent = _make_agent()
        overview = {
            "PERatio": "None",
            "ForwardPE": "-",
            "EPS": "",
        }
        m = agent._extract_metrics(overview)
        assert m["pe_ratio"] == 0.0
        assert m["forward_pe"] == 0.0
        assert m["eps"] == 0.0


class TestExtractEarnings:
    def test_parses_quarterly_earnings(self):
        agent = _make_agent()
        data = {
            "quarterlyEarnings": [
                {
                    "reportedDate": "2024-01-25",
                    "reportedEPS": "2.18",
                    "estimatedEPS": "2.10",
                    "surprisePercentage": "3.81",
                },
                {
                    "reportedDate": "2023-10-25",
                    "reportedEPS": "1.46",
                    "estimatedEPS": "1.39",
                    "surprisePercentage": "5.04",
                },
            ],
        }
        result = agent._extract_earnings(data)
        assert len(result) == 2
        assert result[0]["reported_eps"] == pytest.approx(2.18)
        assert result[0]["surprise_pct"] == pytest.approx(3.81)

    def test_empty_earnings(self):
        agent = _make_agent()
        result = agent._extract_earnings({})
        assert result == []


# ═══════════════════════════════════════════════════════════════════
# OPRO 動態權重計算
# ═══════════════════════════════════════════════════════════════════

class TestWeightedScore:
    def test_cheap_profitable_growing(self):
        """低 P/E + 高 ROE + 高成長 → 強烈正分數"""
        agent = _make_agent()
        metrics = {
            "pe_ratio": 10.0,     # 便宜
            "pb_ratio": 0.8,      # 低於帳面
            "ev_to_ebitda": 8.0,  # 合理
            "roe": 0.20,          # 優秀
            "profit_margin": 0.25,  # 高毛利
            "revenue_growth": 0.25,  # 高成長
        }
        score = agent._compute_weighted_score(metrics)
        assert score > 0, "便宜 + 盈利 + 成長應產生正分數"

    def test_expensive_unprofitable_declining(self):
        """高 P/E + 低 ROE + 營收衰退 → 強烈負分數"""
        agent = _make_agent()
        metrics = {
            "pe_ratio": 50.0,       # 昂貴
            "pb_ratio": 8.0,        # 過高
            "ev_to_ebitda": 25.0,   # 昂貴
            "roe": -0.05,           # 虧損
            "profit_margin": -0.10, # 淨虧損
            "revenue_growth": -0.15,  # 營收衰退
        }
        score = agent._compute_weighted_score(metrics)
        assert score < 0, "昂貴 + 虧損 + 衰退應產生負分數"

    def test_mixed_signals_near_neutral(self):
        """混合信號 → 接近中性"""
        agent = _make_agent()
        metrics = {
            "pe_ratio": 20.0,      # 合理
            "pb_ratio": 2.0,       # 中性
            "ev_to_ebitda": 15.0,  # 中性
            "roe": 0.10,           # 尚可
            "profit_margin": 0.10, # 中等
            "revenue_growth": 0.03, # 低成長
        }
        score = agent._compute_weighted_score(metrics)
        assert -15 < score < 15, f"混合信號應接近中性，但得到 {score}"

    def test_loss_making_company(self):
        """虧損公司（P/E <= 0）→ 負分數"""
        agent = _make_agent()
        metrics = {
            "pe_ratio": -5.0,
            "pb_ratio": 3.0,
            "ev_to_ebitda": 0.0,
            "roe": -0.30,
            "profit_margin": -0.15,
            "revenue_growth": -0.20,
        }
        score = agent._compute_weighted_score(metrics)
        assert score < -10, "虧損公司應產生明顯負分數"

    def test_zero_weights_disable_all(self):
        """所有權重歸零 → 分數為 0"""
        agent = _make_agent()
        agent.update_weights_from_opro({
            "value": 0.0,
            "profitability": 0.0,
            "growth": 0.0,
        })
        metrics = {
            "pe_ratio": 5.0,     # 極度便宜
            "roe": 0.50,         # 極度賺錢
            "revenue_growth": 0.50,  # 極度成長
        }
        score = agent._compute_weighted_score(metrics)
        assert score == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════
# 補充維度 + 原因生成
# ═══════════════════════════════════════════════════════════════════

class TestBonusAndReasons:
    def test_forward_pe_improvement(self):
        """Forward P/E 明顯低於 trailing → 正 bonus"""
        agent = _make_agent()
        metrics = {
            "pe_ratio": 25.0,
            "forward_pe": 18.0,  # 比 trailing 低 28%
            "roe": 0.12,
            "profit_margin": 0.15,
            "ev_to_ebitda": 12.0,
            "revenue_growth": 0.08,
        }
        bonus, reasons = agent._compute_bonus_and_reasons(metrics, [], {})
        assert bonus > 0
        assert any("earnings growth" in r.lower() for r in reasons)

    def test_earnings_beat_streak(self):
        """連續 3 季盈餘超預期 → 正 bonus"""
        agent = _make_agent()
        metrics = {"pe_ratio": 20.0, "forward_pe": 0.0, "roe": 0.0,
                    "profit_margin": 0.0, "ev_to_ebitda": 0.0,
                    "revenue_growth": 0.0}
        earnings = [
            {"surprise_pct": 5.0},
            {"surprise_pct": 3.0},
            {"surprise_pct": 2.0},
            {"surprise_pct": 1.0},
        ]
        bonus, reasons = agent._compute_bonus_and_reasons(
            metrics, earnings, {},
        )
        assert bonus > 0
        assert any("beat streak" in r.lower() for r in reasons)

    def test_earnings_miss_streak(self):
        """連續 3 季盈餘不及預期 → 負 bonus"""
        agent = _make_agent()
        metrics = {"pe_ratio": 20.0, "forward_pe": 0.0, "roe": 0.0,
                    "profit_margin": 0.0, "ev_to_ebitda": 0.0,
                    "revenue_growth": 0.0}
        earnings = [
            {"surprise_pct": -3.0},
            {"surprise_pct": -5.0},
            {"surprise_pct": -2.0},
        ]
        bonus, reasons = agent._compute_bonus_and_reasons(
            metrics, earnings, {},
        )
        assert bonus < 0
        assert any("miss streak" in r.lower() for r in reasons)

    def test_reasons_generated_for_key_metrics(self):
        """確認每個重要指標都會產生人類可讀原因"""
        agent = _make_agent()
        metrics = {
            "pe_ratio": 10.0,
            "forward_pe": 0.0,
            "roe": 0.25,
            "profit_margin": 0.30,
            "ev_to_ebitda": 8.0,
            "revenue_growth": 0.25,
        }
        _, reasons = agent._compute_bonus_and_reasons(metrics, [], {})
        assert len(reasons) >= 3  # P/E + ROE + margin + EV + growth


# ═══════════════════════════════════════════════════════════════════
# OPRO 介面
# ═══════════════════════════════════════════════════════════════════

class TestOPROInterface:
    def test_default_weights(self):
        agent = _make_agent()
        assert agent.dynamic_weights["value"] == pytest.approx(0.4)
        assert agent.dynamic_weights["profitability"] == pytest.approx(0.4)
        assert agent.dynamic_weights["growth"] == pytest.approx(0.2)

    def test_full_weight_update(self):
        agent = _make_agent()
        agent.update_weights_from_opro({
            "value": 0.2,
            "profitability": 0.3,
            "growth": 0.5,
        })
        assert agent.dynamic_weights["value"] == pytest.approx(0.2)
        assert agent.dynamic_weights["profitability"] == pytest.approx(0.3)
        assert agent.dynamic_weights["growth"] == pytest.approx(0.5)

    def test_partial_weight_update(self):
        agent = _make_agent()
        agent.update_weights_from_opro({"growth": 0.6})
        assert agent.dynamic_weights["growth"] == pytest.approx(0.6)
        assert agent.dynamic_weights["value"] == pytest.approx(0.4)  # 未變

    def test_ignores_unknown_keys(self):
        agent = _make_agent()
        agent.update_weights_from_opro({"unknown_key": 0.99})
        assert "unknown_key" not in agent.dynamic_weights

    def test_weight_affects_score(self):
        """調高成長權重 → 高成長公司分數提升"""
        agent = _make_agent()
        metrics = {
            "pe_ratio": 35.0,      # 估值偏高（value 維度負分）
            "pb_ratio": 0.0,
            "ev_to_ebitda": 0.0,
            "roe": 0.05,           # 盈利一般
            "profit_margin": 0.05,
            "revenue_growth": 0.30,  # 高成長
        }

        # 預設權重
        score_default = agent._compute_weighted_score(metrics)

        # 調高成長權重
        agent.update_weights_from_opro({
            "value": 0.1,
            "profitability": 0.1,
            "growth": 0.8,
        })
        score_growth_heavy = agent._compute_weighted_score(metrics)

        assert score_growth_heavy > score_default, (
            "調高成長權重後，高成長公司的分數應提升"
        )


# ═══════════════════════════════════════════════════════════════════
# 輔助方法
# ═══════════════════════════════════════════════════════════════════

class TestHelpers:
    def test_safe_float_normal(self):
        assert FundamentalAgent._safe_float(28.5) == pytest.approx(28.5)
        assert FundamentalAgent._safe_float("28.5") == pytest.approx(28.5)

    def test_safe_float_edge_cases(self):
        assert FundamentalAgent._safe_float(None) == 0.0
        assert FundamentalAgent._safe_float("None") == 0.0
        assert FundamentalAgent._safe_float("-") == 0.0
        assert FundamentalAgent._safe_float("") == 0.0

    def test_score_to_direction(self):
        assert FundamentalAgent._score_to_direction(50) == SignalDirection.STRONG_BUY
        assert FundamentalAgent._score_to_direction(20) == SignalDirection.BUY
        assert FundamentalAgent._score_to_direction(5) == SignalDirection.NEUTRAL
        assert FundamentalAgent._score_to_direction(-20) == SignalDirection.SELL
        assert FundamentalAgent._score_to_direction(-50) == SignalDirection.STRONG_SELL


# ═══════════════════════════════════════════════════════════════════
# End-to-End (with mocked data)
# ═══════════════════════════════════════════════════════════════════

class TestAnalyseEndToEnd:
    @pytest.mark.asyncio
    async def test_bullish_company(self):
        """低 P/E + 高 ROE + 成長 → BUY 或 STRONG_BUY"""
        agent = _make_agent()
        agent._av.company_overview = AsyncMock(return_value={
            "Symbol": "AAPL",
            "PERatio": 12.0,
            "ForwardPE": 10.0,
            "PriceToBookRatio": 0.9,
            "EVToEBITDA": 8.0,
            "ReturnOnEquityTTM": 0.25,
            "ProfitMargin": 0.28,
            "EPS": 6.43,
            "QuarterlyRevenueGrowthYOY": 0.22,
            "DividendYield": 0.005,
        })
        agent._av.earnings = AsyncMock(return_value={
            "quarterlyEarnings": [
                {"reportedEPS": "2.18", "estimatedEPS": "2.10", "surprisePercentage": "3.81"},
                {"reportedEPS": "1.46", "estimatedEPS": "1.39", "surprisePercentage": "5.04"},
                {"reportedEPS": "1.52", "estimatedEPS": "1.43", "surprisePercentage": "6.29"},
            ],
        })

        result = await agent.analyse({"ticker": "AAPL"})

        assert isinstance(result, AnalysisSignal)
        assert result.source == "fundamental"
        assert result.direction in (SignalDirection.BUY, SignalDirection.STRONG_BUY)
        assert result.confidence > 0.3
        assert result.data["composite_score"] > 0
        assert result.data["dynamic_weights"]["value"] == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_bearish_company(self):
        """高 P/E + 虧損 + 衰退 → SELL 或 STRONG_SELL"""
        agent = _make_agent()
        agent._av.company_overview = AsyncMock(return_value={
            "Symbol": "BAD",
            "PERatio": -5.0,
            "ForwardPE": -3.0,
            "PriceToBookRatio": 6.0,
            "EVToEBITDA": 25.0,
            "ReturnOnEquityTTM": -0.15,
            "ProfitMargin": -0.10,
            "EPS": -2.0,
            "QuarterlyRevenueGrowthYOY": -0.20,
        })
        agent._av.earnings = AsyncMock(return_value={
            "quarterlyEarnings": [
                {"reportedEPS": "-0.50", "estimatedEPS": "0.10", "surprisePercentage": "-600"},
                {"reportedEPS": "-0.30", "estimatedEPS": "0.05", "surprisePercentage": "-700"},
                {"reportedEPS": "-0.20", "estimatedEPS": "0.15", "surprisePercentage": "-233"},
            ],
        })

        result = await agent.analyse({"ticker": "BAD"})

        assert result.direction in (SignalDirection.SELL, SignalDirection.STRONG_SELL)
        assert result.data["composite_score"] < 0

    @pytest.mark.asyncio
    async def test_no_data_returns_neutral(self):
        """數據取得失敗 → NEUTRAL"""
        agent = _make_agent()
        agent._av.company_overview = AsyncMock(return_value={})
        agent._av.earnings = AsyncMock(return_value={})

        result = await agent.analyse({"ticker": "UNKNOWN"})

        assert result.direction == SignalDirection.NEUTRAL
        assert result.confidence == 0.0
        assert "無法獲取" in result.reasoning

    @pytest.mark.asyncio
    async def test_data_fetch_exception_handled(self):
        """API 拋出例外 → 不崩潰，返回 NEUTRAL"""
        agent = _make_agent()
        agent._av.company_overview = AsyncMock(side_effect=Exception("API Error"))
        agent._av.earnings = AsyncMock(side_effect=Exception("API Error"))

        result = await agent.analyse({"ticker": "ERR"})

        assert result.direction == SignalDirection.NEUTRAL
