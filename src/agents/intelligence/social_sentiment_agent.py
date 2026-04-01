"""
Social Sentiment Agent — 社群情緒分析代理
=========================================
分析 Reddit 投資社群的討論熱度與情緒走向，
捕捉散戶動量信號。

三大策略維度（由 OPRO 動態調整權重）：
  1. 提及速度 (mention_velocity) — 提及量的變化率
  2. 情緒極性 (sentiment_polarity) — 看多/看空帖子比例
  3. 互動深度 (engagement_depth) — 高讚數帖子的信號權重更大

核心邏輯：
  - 短期內提及量暴增 + 看多情緒 → 散戶動量（但也需留意過熱）
  - 持續負面情緒 + 高互動 → 恐慌蔓延
  - 無社群數據 → NEUTRAL

使用者：
  - IntelligenceOrchestrator：多代理信號融合
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.core.base_agent import AnalysisSignal, BaseAgent, SignalDirection
from src.data.providers.social_sentiment import SocialSentimentProvider

logger = logging.getLogger(__name__)


class SocialSentimentAgent(BaseAgent):
    """
    社群情緒分析代理。

    從 Reddit 投資社群（WSB、stocks、investing）收集數據，
    分析散戶情緒的方向與強度。
    """

    _DEFAULT_STRATEGY_WEIGHTS = {
        "mention_velocity": 0.30,
        "sentiment_polarity": 0.40,
        "engagement_depth": 0.30,
    }

    def __init__(self, social_provider: SocialSentimentProvider) -> None:
        super().__init__("social_sentiment")
        self._social = social_provider
        self.dynamic_weights = dict(self._DEFAULT_STRATEGY_WEIGHTS)

    async def analyse(self, context: dict[str, Any]) -> AnalysisSignal:
        """執行社群情緒分析。"""
        ticker = context["ticker"]

        try:
            sentiment_data = await self._social.reddit_sentiment_score(ticker)
        except Exception:
            self.logger.warning("Failed to fetch social data for %s", ticker)
            return AnalysisSignal(
                source=self.name,
                direction=SignalDirection.NEUTRAL,
                confidence=0.0,
                reasoning=f"Failed to fetch social data for {ticker}",
            )

        # Extract features
        metrics = self._extract_features(sentiment_data, ticker)
        reasons: list[str] = metrics.pop("_reasons", [])

        # Weighted score
        weighted_score = self._compute_weighted_score(metrics)

        direction = self._score_to_direction(weighted_score)
        confidence = min(abs(weighted_score) / 50.0, 1.0)

        # Low data → reduce confidence
        total_mentions = sentiment_data.get("total_mentions", 0)
        if total_mentions < 3:
            confidence *= 0.5

        if not reasons:
            reasons.append(f"Insufficient social data for {ticker}")

        logger.info(
            "[SocialSentimentAgent] %s 分析完成: 訊號=%s, 信心度=%.2f, "
            "mentions=%d",
            ticker, direction.name, confidence, total_mentions,
        )

        return AnalysisSignal(
            source=self.name,
            direction=direction,
            confidence=confidence,
            reasoning=" | ".join(reasons),
            data={
                "composite_score": weighted_score,
                "total_mentions": total_mentions,
                "sentiment_score": sentiment_data.get("score", 0.0),
                "dynamic_weights": dict(self.dynamic_weights),
                **{k: v for k, v in metrics.items() if not k.startswith("_")},
            },
        )

    def _extract_features(
        self,
        data: dict[str, Any],
        ticker: str,
    ) -> dict[str, Any]:
        """將社群數據轉為量化指標。"""
        metrics: dict[str, Any] = {}
        reasons: list[str] = []

        total = data.get("total_mentions", 0)
        bullish = data.get("bullish_count", 0)
        bearish = data.get("bearish_count", 0)
        score = data.get("score", 0.0)
        avg_engagement = data.get("avg_engagement", 0.0)
        velocity = data.get("mention_velocity", 0.0)

        # 1. Mention velocity — how much is the stock being discussed
        velocity_score = 0.0
        if velocity > 10:
            velocity_score = 20.0
            reasons.append(f"High mention velocity ({velocity:.0f}/sub) — trending")
        elif velocity > 5:
            velocity_score = 10.0
            reasons.append(f"Moderate mentions ({velocity:.0f}/sub)")
        elif velocity > 1:
            velocity_score = 3.0
            reasons.append(f"Low mentions ({velocity:.0f}/sub)")
        else:
            reasons.append(f"Minimal social discussion for {ticker}")

        # High velocity amplifies the direction, not inherently bullish
        metrics["velocity_score"] = velocity_score * (1 if score >= 0 else -1)

        # 2. Sentiment polarity — bullish vs bearish ratio
        polarity_score = 0.0
        if total > 0:
            polarity_score = score * 25.0  # score is -1 to +1
            if bullish > bearish:
                reasons.append(
                    f"Social sentiment: {bullish} bullish, {bearish} bearish "
                    f"(score={score:+.2f})"
                )
            elif bearish > bullish:
                reasons.append(
                    f"Social sentiment: {bearish} bearish, {bullish} bullish "
                    f"(score={score:+.2f})"
                )
            else:
                reasons.append(f"Social sentiment: balanced ({total} mentions)")

        metrics["polarity_score"] = polarity_score

        # 3. Engagement depth — high upvotes = stronger conviction
        engagement_score = 0.0
        if avg_engagement > 100:
            engagement_score = 15.0 * (1 if score >= 0 else -1)
            reasons.append(f"High engagement (avg {avg_engagement:.0f} upvotes)")
        elif avg_engagement > 20:
            engagement_score = 5.0 * (1 if score >= 0 else -1)
            reasons.append(f"Moderate engagement (avg {avg_engagement:.0f} upvotes)")

        metrics["engagement_score"] = engagement_score

        metrics["_reasons"] = reasons
        return metrics

    def _compute_weighted_score(self, metrics: dict[str, Any]) -> float:
        """OPRO 動態權重融合。"""
        return (
            metrics.get("velocity_score", 0.0) * self.dynamic_weights["mention_velocity"]
            + metrics.get("polarity_score", 0.0) * self.dynamic_weights["sentiment_polarity"]
            + metrics.get("engagement_score", 0.0) * self.dynamic_weights["engagement_depth"]
        )

    def update_weights_from_opro(self, new_weights: dict[str, float]) -> None:
        """OPRO 介面。"""
        for key in self._DEFAULT_STRATEGY_WEIGHTS:
            if key in new_weights:
                self.dynamic_weights[key] = new_weights[key]
        logger.info("[SocialSentimentAgent] OPRO 權重更新: %s", self.dynamic_weights)

    @staticmethod
    def _score_to_direction(score: float) -> SignalDirection:
        if score >= 20:
            return SignalDirection.STRONG_BUY
        if score >= 7:
            return SignalDirection.BUY
        if score <= -20:
            return SignalDirection.STRONG_SELL
        if score <= -7:
            return SignalDirection.SELL
        return SignalDirection.NEUTRAL
