"""
Options Flow Agent — 期權流量分析代理
=====================================
分析期權市場的異常活動，追蹤大戶（Smart Money）動向。

三大策略維度（由 OPRO 動態調整權重）：
  1. 異常活動 (unusual_activity) — 大額異常交易量
  2. Put/Call 情緒 (put_call_sentiment) — P/C 比偏離歷史均值
  3. 聰明錢流向 (smart_money_flow) — 高額期權金方向性流動

核心邏輯：
  - 大量看漲期權異常交易 → 機構看多
  - P/C Ratio 偏高 → 恐慌（逆向指標，可能反轉）
  - Smart Money 淨流入看漲 → 偏多信號

使用者：
  - IntelligenceOrchestrator：多代理信號融合
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.core.base_agent import AnalysisSignal, BaseAgent, SignalDirection
from src.data.providers.options_flow import OptionsFlowProvider

logger = logging.getLogger(__name__)


class OptionsFlowAgent(BaseAgent):
    """
    期權流量分析代理。

    分析期權市場的異常活動、Put/Call Ratio 和大額交易流向，
    推斷機構投資者（Smart Money）的方向性押注。
    """

    _DEFAULT_STRATEGY_WEIGHTS = {
        "unusual_activity": 0.35,
        "put_call_sentiment": 0.30,
        "smart_money_flow": 0.35,
    }

    def __init__(self, options_provider: OptionsFlowProvider) -> None:
        super().__init__("options_flow")
        self._options = options_provider
        self.dynamic_weights = dict(self._DEFAULT_STRATEGY_WEIGHTS)

    async def analyse(self, context: dict[str, Any]) -> AnalysisSignal:
        """執行期權流量分析。"""
        ticker = context["ticker"]

        # UK stocks typically don't have US options data
        if ".L" in ticker:
            return AnalysisSignal(
                source=self.name,
                direction=SignalDirection.NEUTRAL,
                confidence=0.0,
                reasoning="Options flow not available for UK stocks",
            )

        # Parallel fetch
        activity_task = asyncio.create_task(
            self._safe_fetch(self._options.unusual_activity(ticker), []),
        )
        pcr_task = asyncio.create_task(
            self._safe_fetch(self._options.put_call_ratio(ticker), {}),
        )
        smart_task = asyncio.create_task(
            self._safe_fetch(self._options.smart_money_flow(ticker), {}),
        )

        activity = await activity_task
        pcr = await pcr_task
        smart_flow = await smart_task

        # Extract features
        metrics = self._extract_features(activity, pcr, smart_flow)
        reasons: list[str] = metrics.pop("_reasons", [])

        # Weighted score
        weighted_score = self._compute_weighted_score(metrics)

        direction = self._score_to_direction(weighted_score)
        confidence = min(abs(weighted_score) / 60.0, 1.0)

        if not reasons:
            reasons.append("No options flow data available")

        logger.info(
            "[OptionsFlowAgent] %s 分析完成: 訊號=%s, 信心度=%.2f",
            ticker, direction.name, confidence,
        )

        return AnalysisSignal(
            source=self.name,
            direction=direction,
            confidence=confidence,
            reasoning=" | ".join(reasons),
            data={
                "composite_score": weighted_score,
                "dynamic_weights": dict(self.dynamic_weights),
                **{k: v for k, v in metrics.items() if not k.startswith("_")},
            },
        )

    def _extract_features(
        self,
        activity: list[dict],
        pcr: dict,
        smart_flow: dict,
    ) -> dict[str, Any]:
        """將期權數據轉為量化指標。"""
        metrics: dict[str, Any] = {}
        reasons: list[str] = []

        # 1. Unusual activity — count bullish vs bearish
        activity_score = 0.0
        if activity:
            bullish = sum(1 for a in activity if a.get("sentiment") == "bullish")
            bearish = sum(1 for a in activity if a.get("sentiment") == "bearish")
            total = len(activity)

            metrics["unusual_bullish"] = bullish
            metrics["unusual_bearish"] = bearish
            metrics["unusual_total"] = total

            if total > 0:
                ratio = (bullish - bearish) / total
                activity_score = ratio * 25.0

                if bullish > bearish:
                    reasons.append(
                        f"Unusual options: {bullish} bullish vs {bearish} bearish "
                        f"({total} total) — call-heavy"
                    )
                elif bearish > bullish:
                    reasons.append(
                        f"Unusual options: {bearish} bearish vs {bullish} bullish "
                        f"({total} total) — put-heavy"
                    )
                else:
                    reasons.append(f"Unusual options: balanced ({total} total)")

        metrics["activity_score"] = activity_score

        # 2. Put/Call Ratio
        pcr_score = 0.0
        pcr_ratio = pcr.get("ratio", 0.0)
        if pcr_ratio > 0:
            metrics["pcr_ratio"] = pcr_ratio

            # P/C Ratio interpretation:
            # < 0.7 → bullish (more calls)
            # 0.7-1.0 → neutral
            # > 1.0 → bearish (more puts) BUT can be contrarian bullish if extreme
            # > 1.5 → extreme fear → contrarian bullish signal
            if pcr_ratio > 1.5:
                pcr_score = 15.0  # Contrarian: extreme fear → buy opportunity
                reasons.append(
                    f"P/C ratio {pcr_ratio:.2f} — extreme fear (contrarian bullish)"
                )
            elif pcr_ratio > 1.0:
                pcr_score = -10.0
                reasons.append(f"P/C ratio {pcr_ratio:.2f} — elevated put activity")
            elif pcr_ratio < 0.5:
                pcr_score = -5.0  # Extreme complacency → contrarian bearish
                reasons.append(
                    f"P/C ratio {pcr_ratio:.2f} — extreme complacency (contrarian risk)"
                )
            elif pcr_ratio < 0.7:
                pcr_score = 10.0
                reasons.append(f"P/C ratio {pcr_ratio:.2f} — bullish call flow")
            else:
                reasons.append(f"P/C ratio {pcr_ratio:.2f} — neutral")

        metrics["pcr_score"] = pcr_score

        # 3. Smart Money flow
        smart_score = 0.0
        net_flow = smart_flow.get("net_flow", 0.0)
        large_count = smart_flow.get("large_trades_count", 0)

        if large_count > 0:
            metrics["smart_net_flow"] = net_flow
            metrics["large_trades"] = large_count

            # Normalize net flow
            bullish_prem = smart_flow.get("bullish_premium", 0.0)
            bearish_prem = smart_flow.get("bearish_premium", 0.0)
            total_prem = bullish_prem + bearish_prem

            if total_prem > 0:
                flow_ratio = net_flow / total_prem
                smart_score = flow_ratio * 25.0

                if net_flow > 0:
                    reasons.append(
                        f"Smart money: ${net_flow:,.0f} net bullish "
                        f"({large_count} large trades)"
                    )
                else:
                    reasons.append(
                        f"Smart money: ${abs(net_flow):,.0f} net bearish "
                        f"({large_count} large trades)"
                    )

        metrics["smart_score"] = smart_score

        metrics["_reasons"] = reasons
        return metrics

    def _compute_weighted_score(self, metrics: dict[str, Any]) -> float:
        """OPRO 動態權重融合。"""
        return (
            metrics.get("activity_score", 0.0) * self.dynamic_weights["unusual_activity"]
            + metrics.get("pcr_score", 0.0) * self.dynamic_weights["put_call_sentiment"]
            + metrics.get("smart_score", 0.0) * self.dynamic_weights["smart_money_flow"]
        )

    def update_weights_from_opro(self, new_weights: dict[str, float]) -> None:
        """OPRO 介面。"""
        for key in self._DEFAULT_STRATEGY_WEIGHTS:
            if key in new_weights:
                self.dynamic_weights[key] = new_weights[key]
        logger.info("[OptionsFlowAgent] OPRO 權重更新: %s", self.dynamic_weights)

    async def _safe_fetch(self, coro: Any, default: Any) -> Any:
        try:
            return await coro
        except Exception:
            self.logger.warning("期權數據取得失敗", exc_info=True)
            return default

    @staticmethod
    def _score_to_direction(score: float) -> SignalDirection:
        if score >= 25:
            return SignalDirection.STRONG_BUY
        if score >= 8:
            return SignalDirection.BUY
        if score <= -25:
            return SignalDirection.STRONG_SELL
        if score <= -8:
            return SignalDirection.SELL
        return SignalDirection.NEUTRAL
