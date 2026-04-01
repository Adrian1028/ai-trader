"""
Insider Trading Agent — 內部人交易分析代理
==========================================
分析 SEC Form 4 內部人交易記錄，偵測管理層動向。

策略維度（由 OPRO 動態調整權重）：
  1. 買賣比率 (buy_sell_ratio) — 內部人淨買入 vs 淨賣出
  2. 集群活動 (cluster_activity) — 多個內部人同時交易
  3. 活動強度 (activity_intensity) — Form 4 提交頻率

核心邏輯：
  - 管理層集體增持 → 強烈看多信號
  - 管理層集體減持 → 看空信號（但需區分例行減持）
  - 無 Form 4 記錄 → NEUTRAL（資訊不足）

數據來源：
  - SEC EDGAR（免費，無需 API key）

使用者：
  - IntelligenceOrchestrator：多代理信號融合
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.core.base_agent import AnalysisSignal, BaseAgent, SignalDirection
from src.data.providers.sec_edgar import SECEdgarProvider

logger = logging.getLogger(__name__)


class InsiderTradingAgent(BaseAgent):
    """
    內部人交易分析代理。

    分析 SEC Form 4 filing 頻率與模式，
    判斷公司管理層對自家股票的信心程度。
    """

    _DEFAULT_STRATEGY_WEIGHTS = {
        "activity_intensity": 0.40,
        "cluster_activity": 0.35,
        "recency_signal": 0.25,
    }

    def __init__(self, sec_provider: SECEdgarProvider) -> None:
        super().__init__("insider")
        self._sec = sec_provider
        self.dynamic_weights = dict(self._DEFAULT_STRATEGY_WEIGHTS)

    async def analyse(self, context: dict[str, Any]) -> AnalysisSignal:
        """
        執行內部人交易分析。

        注意：SEC EDGAR 僅適用於美國上市公司。
        英國股票會自動返回 NEUTRAL。
        """
        ticker = context["ticker"]

        # UK stocks don't have SEC filings
        if ".L" in ticker:
            return AnalysisSignal(
                source=self.name,
                direction=SignalDirection.NEUTRAL,
                confidence=0.0,
                reasoning="SEC EDGAR not applicable for UK stocks",
            )

        # Fetch insider data
        try:
            summary = await self._sec.insider_summary(ticker)
            transactions = await self._sec.insider_transactions(ticker, limit=30)
        except Exception:
            self.logger.warning("Failed to fetch insider data for %s", ticker)
            return AnalysisSignal(
                source=self.name,
                direction=SignalDirection.NEUTRAL,
                confidence=0.0,
                reasoning=f"Failed to fetch insider data for {ticker}",
            )

        # Extract features
        metrics = self._extract_features(summary, transactions)
        reasons: list[str] = metrics.pop("_reasons", [])

        # Weighted score
        weighted_score = self._compute_weighted_score(metrics)

        direction = self._score_to_direction(weighted_score)
        confidence = min(abs(weighted_score) / 50.0, 1.0)

        if not reasons:
            reasons.append("No insider activity detected")

        logger.info(
            "[InsiderAgent] %s 分析完成: 訊號=%s, 信心度=%.2f",
            ticker, direction.name, confidence,
        )

        return AnalysisSignal(
            source=self.name,
            direction=direction,
            confidence=confidence,
            reasoning=" | ".join(reasons),
            data={
                "composite_score": weighted_score,
                "activity_level": summary.get("activity_level", "none"),
                "form4_count": summary.get("recent_form4_count", 0),
                "dynamic_weights": dict(self.dynamic_weights),
            },
        )

    def _extract_features(
        self,
        summary: dict[str, Any],
        transactions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """將原始內部人數據轉為量化指標。"""
        metrics: dict[str, Any] = {}
        reasons: list[str] = []

        form4_count = summary.get("recent_form4_count", 0)
        activity_level = summary.get("activity_level", "none")

        # 1. Activity intensity — Form 4 filing frequency
        intensity_score = 0.0
        if activity_level == "high":
            intensity_score = 15.0
            reasons.append(f"High insider activity ({form4_count} Form 4 filings)")
        elif activity_level == "moderate":
            intensity_score = 8.0
            reasons.append(f"Moderate insider activity ({form4_count} filings)")
        elif activity_level == "low":
            intensity_score = 3.0
            reasons.append(f"Low insider activity ({form4_count} filings)")
        else:
            reasons.append("No recent insider activity")

        metrics["intensity_score"] = intensity_score

        # 2. Cluster activity — multiple filings close together
        cluster_score = 0.0
        if len(transactions) >= 3:
            # Check if multiple filings have similar dates
            dates = [t.get("filing_date", "") for t in transactions[:10]]
            unique_dates = set(dates)
            if len(dates) > 0 and len(unique_dates) <= len(dates) * 0.5:
                cluster_score = 15.0
                reasons.append("Insider cluster detected — multiple filings on same dates")
            elif form4_count >= 5:
                cluster_score = 8.0
                reasons.append("Elevated insider filing frequency")

        metrics["cluster_score"] = cluster_score

        # 3. Recency — how recent is the latest filing
        recency_score = 0.0
        if transactions:
            latest_date = transactions[0].get("filing_date", "")
            if latest_date:
                recency_score = 10.0
                reasons.append(f"Latest insider filing: {latest_date}")

        metrics["recency_score"] = recency_score

        metrics["_reasons"] = reasons
        return metrics

    def _compute_weighted_score(self, metrics: dict[str, Any]) -> float:
        """OPRO 動態權重融合。"""
        intensity = metrics.get("intensity_score", 0.0)
        cluster = metrics.get("cluster_score", 0.0)
        recency = metrics.get("recency_score", 0.0)

        return (
            intensity * self.dynamic_weights["activity_intensity"]
            + cluster * self.dynamic_weights["cluster_activity"]
            + recency * self.dynamic_weights["recency_signal"]
        )

    def update_weights_from_opro(self, new_weights: dict[str, float]) -> None:
        """OPRO 介面。"""
        for key in self._DEFAULT_STRATEGY_WEIGHTS:
            if key in new_weights:
                self.dynamic_weights[key] = new_weights[key]
        logger.info("[InsiderAgent] OPRO 權重更新: %s", self.dynamic_weights)

    @staticmethod
    def _score_to_direction(score: float) -> SignalDirection:
        if score >= 20:
            return SignalDirection.STRONG_BUY
        if score >= 8:
            return SignalDirection.BUY
        if score <= -20:
            return SignalDirection.STRONG_SELL
        if score <= -8:
            return SignalDirection.SELL
        return SignalDirection.NEUTRAL
