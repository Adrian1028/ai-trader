"""
Unit Tests for SocialSentimentAgent (社群情緒分析代理)
======================================================
測試：
  1. 特徵提取（提及速度、情緒極性、互動深度）
  2. OPRO 介面
  3. 低數據信心衰減
  4. End-to-End
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.agents.intelligence.social_sentiment_agent import SocialSentimentAgent
from src.core.base_agent import AnalysisSignal, SignalDirection
from src.data.providers.social_sentiment import SocialSentimentProvider


def _make_agent() -> SocialSentimentAgent:
    provider = SocialSentimentProvider()
    return SocialSentimentAgent(provider)


class TestExtractFeatures:
    def test_high_bullish_velocity(self):
        agent = _make_agent()
        data = {
            "score": 0.8,
            "total_mentions": 30,
            "bullish_count": 24,
            "bearish_count": 3,
            "neutral_count": 3,
            "avg_engagement": 150,
            "mention_velocity": 15,
        }
        metrics = agent._extract_features(data, "AAPL")
        assert metrics["velocity_score"] > 0
        assert metrics["polarity_score"] > 0
        assert metrics["engagement_score"] > 0

    def test_bearish_sentiment(self):
        agent = _make_agent()
        data = {
            "score": -0.6,
            "total_mentions": 20,
            "bullish_count": 4,
            "bearish_count": 14,
            "neutral_count": 2,
            "avg_engagement": 80,
            "mention_velocity": 8,
        }
        metrics = agent._extract_features(data, "BAD")
        assert metrics["polarity_score"] < 0

    def test_no_mentions(self):
        agent = _make_agent()
        data = {
            "score": 0.0,
            "total_mentions": 0,
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "avg_engagement": 0.0,
            "mention_velocity": 0.0,
        }
        metrics = agent._extract_features(data, "UNKNOWN")
        assert metrics["velocity_score"] == 0
        assert metrics["polarity_score"] == 0
        assert metrics["engagement_score"] == 0


class TestOPRO:
    def test_weight_update(self):
        agent = _make_agent()
        agent.update_weights_from_opro({"mention_velocity": 0.5})
        assert agent.dynamic_weights["mention_velocity"] == pytest.approx(0.5)

    def test_weight_shift_affects_score(self):
        agent = _make_agent()
        metrics = {
            "velocity_score": 20.0,
            "polarity_score": -5.0,
            "engagement_score": 10.0,
        }
        score_default = agent._compute_weighted_score(metrics)

        agent.update_weights_from_opro({
            "mention_velocity": 0.8,
            "sentiment_polarity": 0.1,
            "engagement_depth": 0.1,
        })
        score_velocity = agent._compute_weighted_score(metrics)
        assert score_velocity > score_default


class TestDirection:
    def test_strong_bullish(self):
        assert SocialSentimentAgent._score_to_direction(25) == SignalDirection.STRONG_BUY

    def test_neutral(self):
        assert SocialSentimentAgent._score_to_direction(3) == SignalDirection.NEUTRAL


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_bullish_social(self):
        agent = _make_agent()
        agent._social.reddit_sentiment_score = AsyncMock(return_value={
            "score": 0.7,
            "total_mentions": 25,
            "bullish_count": 18,
            "bearish_count": 3,
            "neutral_count": 4,
            "avg_engagement": 120,
            "mention_velocity": 12,
        })

        result = await agent.analyse({"ticker": "GME"})
        assert isinstance(result, AnalysisSignal)
        assert result.source == "social_sentiment"
        assert result.direction in (SignalDirection.BUY, SignalDirection.STRONG_BUY)
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_low_data_reduced_confidence(self):
        agent = _make_agent()
        agent._social.reddit_sentiment_score = AsyncMock(return_value={
            "score": 0.5,
            "total_mentions": 2,
            "bullish_count": 2,
            "bearish_count": 0,
            "neutral_count": 0,
            "avg_engagement": 10,
            "mention_velocity": 1,
        })

        result = await agent.analyse({"ticker": "RARE"})
        # Low data → confidence should be dampened
        assert result.confidence < 0.5

    @pytest.mark.asyncio
    async def test_api_failure_neutral(self):
        agent = _make_agent()
        agent._social.reddit_sentiment_score = AsyncMock(
            side_effect=Exception("API down"),
        )

        result = await agent.analyse({"ticker": "ERR"})
        assert result.direction == SignalDirection.NEUTRAL
