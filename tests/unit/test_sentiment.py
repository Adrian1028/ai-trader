"""
Unit Tests for SentimentAgent (情緒分析代理)
=============================================
測試：
  1. 關鍵字情緒分析 (keyword sentiment + stemming)
  2. API 情緒分數提取 (Finnhub sentiment extraction)
  3. 特徵提取流程 (feature extraction)
  4. OPRO 動態權重計算 (weighted score)
  5. 權重更新介面 (OPRO interface)
  6. 分數到方向映射 (score_to_direction)
  7. 完整分析流程 (end-to-end with mocked data)
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.agents.intelligence.sentiment import SentimentAgent
from src.core.base_agent import AnalysisSignal, SignalDirection
from src.data.providers.finnhub import FinnhubProvider


def _make_agent() -> SentimentAgent:
    fh = FinnhubProvider(api_key="test")
    return SentimentAgent(finnhub=fh)


# ═══════════════════════════════════════════════════════════════════
# 關鍵字情緒分析
# ═══════════════════════════════════════════════════════════════════

class TestKeywordSentiment:
    def test_basic_positive_negative(self):
        articles = [
            {"headline": "Company beats earnings, stock surges",
             "summary": "Strong growth reported"},
            {"headline": "Lawsuit filed, shares drop",
             "summary": "Negative outlook"},
        ]
        pos, neg = SentimentAgent._keyword_sentiment(articles)
        assert pos >= 3  # beat, surge, strong, growth
        assert neg >= 2  # lawsuit, drop, negative

    def test_stemming_strips_suffixes(self):
        """詞幹還原：beats→beat, surges→surge, gains→gain"""
        articles = [
            {"headline": "Stock surges after beats expectations",
             "summary": "Gains drive optimistic outlook"},
        ]
        pos, _ = SentimentAgent._keyword_sentiment(articles)
        # surges→surge✓, beats→beat✓, gains→gain✓, optimistic✓
        assert pos >= 3

    def test_empty_articles(self):
        pos, neg = SentimentAgent._keyword_sentiment([])
        assert pos == 0
        assert neg == 0

    def test_no_keywords_found(self):
        articles = [
            {"headline": "Weather forecast for tomorrow",
             "summary": "Partly cloudy skies expected"},
        ]
        pos, neg = SentimentAgent._keyword_sentiment(articles)
        assert pos == 0
        assert neg == 0

    def test_mixed_punctuation(self):
        """確認標點符號被正確去除"""
        articles = [
            {"headline": '"Profit!" says CEO, "growth!" echoes',
             "summary": "(crash) feared by [analysts]."},
        ]
        pos, neg = SentimentAgent._keyword_sentiment(articles)
        assert pos >= 2  # profit, growth
        assert neg >= 1  # crash


# ═══════════════════════════════════════════════════════════════════
# API 情緒分數提取
# ═══════════════════════════════════════════════════════════════════

class TestExtractSentimentScore:
    def test_bullish_percent(self):
        raw = {"sentiment": {"bullishPercent": 0.72}}
        score = SentimentAgent._extract_sentiment_score(raw)
        assert score == pytest.approx(0.72)

    def test_company_news_hq(self):
        raw = {"companyNewsHQSentiment": {"score": 0.65}}
        score = SentimentAgent._extract_sentiment_score(raw)
        assert score == pytest.approx(0.65)

    def test_empty_returns_none(self):
        assert SentimentAgent._extract_sentiment_score({}) is None
        assert SentimentAgent._extract_sentiment_score(None) is None


# ═══════════════════════════════════════════════════════════════════
# 特徵提取
# ═══════════════════════════════════════════════════════════════════

class TestExtractFeatures:
    def test_full_features(self):
        agent = _make_agent()
        news = [
            {"headline": "Stock surges on strong earnings", "summary": ""},
            {"headline": "Growth outlook positive", "summary": ""},
            {"headline": "Profit beats estimates", "summary": ""},
        ]
        sentiment_raw = {"sentiment": {"bullishPercent": 0.75}}
        recommendations = [
            {"buy": 10, "strongBuy": 5, "hold": 3, "sell": 1, "strongSell": 0},
        ]
        earnings = [
            {"surprisePercent": 5.0},
            {"surprisePercent": 3.0},
        ]

        metrics = agent._extract_features(
            news, sentiment_raw, recommendations, earnings,
        )

        assert metrics["api_sentiment"] == pytest.approx(0.75)
        assert metrics["positive_hits"] > 0
        assert metrics["total_articles"] == 3.0
        assert metrics["analyst_score"] > 0  # 15 buy vs 1 sell
        assert metrics["earnings_surprise_score"] > 0  # positive surprises
        assert len(metrics["_reasons"]) >= 3

    def test_few_articles_dampened(self):
        """不足 3 篇新聞時，news_score 應被減半"""
        agent = _make_agent()
        news = [
            {"headline": "Strong profit growth", "summary": ""},
        ]
        metrics = agent._extract_features(news, {}, [], [])
        # 只有 1 篇新聞 → news_score 應被 * 0.5
        raw_pos, raw_neg = SentimentAgent._keyword_sentiment(news)
        if raw_pos + raw_neg > 0:
            expected_full = (raw_pos - raw_neg) / (raw_pos + raw_neg)
            assert metrics["news_score"] == pytest.approx(
                expected_full * 0.5,
            )

    def test_no_data_graceful(self):
        """全部數據為空時不崩潰"""
        agent = _make_agent()
        metrics = agent._extract_features([], {}, [], [])
        assert metrics["news_score"] == 0.0
        assert metrics["analyst_score"] == 0.0
        assert metrics["earnings_surprise_score"] == 0.0


# ═══════════════════════════════════════════════════════════════════
# OPRO 動態權重計算
# ═══════════════════════════════════════════════════════════════════

class TestWeightedScore:
    def test_bullish_all_dimensions(self):
        """三維度全看多 → 正分數"""
        agent = _make_agent()
        metrics = {
            "api_sentiment_centered": 0.5,  # bullish
            "news_score": 0.8,              # 正面新聞
            "analyst_score": 0.7,           # 升評居多
            "earnings_surprise_score": 0.6, # 盈餘超預期
        }
        score = agent._compute_weighted_score(metrics)
        assert score > 0

    def test_bearish_all_dimensions(self):
        """三維度全看空 → 負分數"""
        agent = _make_agent()
        metrics = {
            "api_sentiment_centered": -0.5,
            "news_score": -0.8,
            "analyst_score": -0.6,
            "earnings_surprise_score": -0.5,
        }
        score = agent._compute_weighted_score(metrics)
        assert score < 0

    def test_zero_weights_disable_all(self):
        agent = _make_agent()
        agent.update_weights_from_opro({
            "news_sentiment": 0.0,
            "analyst_consensus": 0.0,
            "social_momentum": 0.0,
        })
        metrics = {
            "api_sentiment_centered": 0.9,
            "news_score": 1.0,
            "analyst_score": 1.0,
            "earnings_surprise_score": 1.0,
        }
        score = agent._compute_weighted_score(metrics)
        assert score == pytest.approx(0.0)

    def test_no_api_sentiment_uses_keywords_only(self):
        """無 API 情緒時，新聞維度只靠關鍵字"""
        agent = _make_agent()
        metrics = {
            "api_sentiment_centered": None,
            "news_score": 0.8,
            "analyst_score": 0.0,
            "earnings_surprise_score": 0.0,
        }
        score = agent._compute_weighted_score(metrics)
        assert score > 0  # 靠 news_score 支撐


# ═══════════════════════════════════════════════════════════════════
# OPRO 介面
# ═══════════════════════════════════════════════════════════════════

class TestOPROInterface:
    def test_default_weights(self):
        agent = _make_agent()
        assert agent.dynamic_weights["news_sentiment"] == pytest.approx(0.5)
        assert agent.dynamic_weights["analyst_consensus"] == pytest.approx(0.3)
        assert agent.dynamic_weights["social_momentum"] == pytest.approx(0.2)

    def test_full_update(self):
        agent = _make_agent()
        agent.update_weights_from_opro({
            "news_sentiment": 0.7,
            "analyst_consensus": 0.2,
            "social_momentum": 0.1,
        })
        assert agent.dynamic_weights["news_sentiment"] == pytest.approx(0.7)
        assert agent.dynamic_weights["analyst_consensus"] == pytest.approx(0.2)

    def test_partial_update(self):
        agent = _make_agent()
        agent.update_weights_from_opro({"social_momentum": 0.5})
        assert agent.dynamic_weights["social_momentum"] == pytest.approx(0.5)
        assert agent.dynamic_weights["news_sentiment"] == pytest.approx(0.5)  # 未變

    def test_ignores_unknown_keys(self):
        agent = _make_agent()
        agent.update_weights_from_opro({"unknown": 0.99})
        assert "unknown" not in agent.dynamic_weights

    def test_weight_shift_affects_score(self):
        """調高分析師權重 → 分析師看多時分數提升"""
        agent = _make_agent()
        metrics = {
            "api_sentiment_centered": -0.3,  # 新聞偏空
            "news_score": -0.5,
            "analyst_score": 0.9,             # 分析師強烈看多
            "earnings_surprise_score": 0.0,
        }

        score_default = agent._compute_weighted_score(metrics)

        agent.update_weights_from_opro({
            "news_sentiment": 0.1,
            "analyst_consensus": 0.8,
            "social_momentum": 0.1,
        })
        score_analyst_heavy = agent._compute_weighted_score(metrics)

        assert score_analyst_heavy > score_default


# ═══════════════════════════════════════════════════════════════════
# 輔助方法
# ═══════════════════════════════════════════════════════════════════

class TestHelpers:
    def test_score_to_direction(self):
        assert SentimentAgent._score_to_direction(40) == SignalDirection.STRONG_BUY
        assert SentimentAgent._score_to_direction(15) == SignalDirection.BUY
        assert SentimentAgent._score_to_direction(5) == SignalDirection.NEUTRAL
        assert SentimentAgent._score_to_direction(-15) == SignalDirection.SELL
        assert SentimentAgent._score_to_direction(-40) == SignalDirection.STRONG_SELL


# ═══════════════════════════════════════════════════════════════════
# End-to-End (with mocked Finnhub)
# ═══════════════════════════════════════════════════════════════════

class TestAnalyseEndToEnd:
    @pytest.mark.asyncio
    async def test_bullish_news_produces_buy(self):
        """大量正面新聞 + 分析師看多 → BUY 或 STRONG_BUY"""
        agent = _make_agent()
        agent._finnhub.company_news = AsyncMock(return_value=[
            {"headline": "Stock surges on record profit",
             "summary": "Strong growth beats estimates"},
            {"headline": "Upgrade to outperform, bullish outlook",
             "summary": "Optimistic forecast"},
            {"headline": "Dividend increase and buyback announced",
             "summary": "Positive shareholder returns"},
        ])
        agent._finnhub.news_sentiment = AsyncMock(return_value={
            "sentiment": {"bullishPercent": 0.80},
        })
        agent._finnhub.recommendation_trends = AsyncMock(return_value=[
            {"buy": 12, "strongBuy": 5, "hold": 2, "sell": 0, "strongSell": 0},
        ])
        agent._finnhub.earnings_surprises = AsyncMock(return_value=[
            {"surprisePercent": 8.0},
            {"surprisePercent": 5.0},
        ])

        result = await agent.analyse({"ticker": "AAPL"})

        assert isinstance(result, AnalysisSignal)
        assert result.source == "sentiment"
        assert result.direction in (SignalDirection.BUY, SignalDirection.STRONG_BUY)
        assert result.confidence > 0.3
        assert result.data["composite_score"] > 0

    @pytest.mark.asyncio
    async def test_bearish_news_produces_sell(self):
        """大量負面新聞 + 分析師看空 → SELL"""
        agent = _make_agent()
        agent._finnhub.company_news = AsyncMock(return_value=[
            {"headline": "Stock crashes after fraud investigation",
             "summary": "Massive loss and decline expected"},
            {"headline": "Downgrade to underperform",
             "summary": "Bearish pessimistic outlook"},
            {"headline": "Lawsuit filed, bankruptcy feared",
             "summary": "Negative sentiment prevails"},
        ])
        agent._finnhub.news_sentiment = AsyncMock(return_value={
            "sentiment": {"bullishPercent": 0.15},
        })
        agent._finnhub.recommendation_trends = AsyncMock(return_value=[
            {"buy": 1, "strongBuy": 0, "hold": 3, "sell": 8, "strongSell": 5},
        ])
        agent._finnhub.earnings_surprises = AsyncMock(return_value=[
            {"surprisePercent": -10.0},
            {"surprisePercent": -5.0},
        ])

        result = await agent.analyse({"ticker": "BAD"})

        assert result.direction in (SignalDirection.SELL, SignalDirection.STRONG_SELL)
        assert result.data["composite_score"] < 0

    @pytest.mark.asyncio
    async def test_no_news_returns_neutral(self):
        """無新聞數據 → NEUTRAL"""
        agent = _make_agent()
        agent._finnhub.company_news = AsyncMock(return_value=[])
        agent._finnhub.news_sentiment = AsyncMock(return_value={})
        agent._finnhub.recommendation_trends = AsyncMock(return_value=[])
        agent._finnhub.earnings_surprises = AsyncMock(return_value=[])

        result = await agent.analyse({"ticker": "UNKNOWN"})

        assert result.direction == SignalDirection.NEUTRAL

    @pytest.mark.asyncio
    async def test_api_failure_graceful(self):
        """API 全部失敗 → 不崩潰，返回 NEUTRAL"""
        agent = _make_agent()
        agent._finnhub.company_news = AsyncMock(
            side_effect=Exception("API down"),
        )
        agent._finnhub.news_sentiment = AsyncMock(
            side_effect=Exception("API down"),
        )
        agent._finnhub.recommendation_trends = AsyncMock(
            side_effect=Exception("API down"),
        )
        agent._finnhub.earnings_surprises = AsyncMock(
            side_effect=Exception("API down"),
        )

        result = await agent.analyse({"ticker": "ERR"})

        assert result.direction == SignalDirection.NEUTRAL
        assert "Insufficient" in result.reasoning
