"""
Macro Analysis Agent — 宏觀經濟分析代理
========================================
分析宏觀經濟指標，提供市場整體方向判斷。

三大策略維度（由 OPRO 動態調整權重）：
  1. 貨幣政策 (monetary_policy) — Fed 利率方向（升息=偏空，降息=偏多）
  2. 通膨壓力 (inflation_pressure) — CPI 趨勢 vs 預期
  3. 經濟週期 (economic_cycle) — 失業率、VIX、殖利率曲線

特性：
  - 此代理提供「全市場」信號，不因個股而異
  - 宏觀數據更新頻率低（月/季），適合長期方向判斷
  - FRED API 免費，120 req/min

使用者：
  - IntelligenceOrchestrator：多代理信號融合
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.core.base_agent import AnalysisSignal, BaseAgent, SignalDirection
from src.data.providers.macro import MacroDataProvider

logger = logging.getLogger(__name__)


class MacroAgent(BaseAgent):
    """
    宏觀經濟分析代理。

    分析 Fed 利率、CPI、殖利率曲線等宏觀指標，
    判斷整體市場環境是有利還是不利於股票投資。

    此代理輸出的信號為「全市場」等級，不因個股而異。
    """

    _DEFAULT_STRATEGY_WEIGHTS = {
        "monetary_policy": 0.40,
        "inflation_pressure": 0.30,
        "economic_cycle": 0.30,
    }

    def __init__(self, macro_provider: MacroDataProvider) -> None:
        super().__init__("macro")
        self._macro = macro_provider
        self.dynamic_weights = dict(self._DEFAULT_STRATEGY_WEIGHTS)

    async def analyse(self, context: dict[str, Any]) -> AnalysisSignal:
        """
        執行宏觀經濟分析。

        此代理忽略 ticker/isin，返回市場整體信號。
        """
        ticker = context.get("ticker", "MARKET")

        # Parallel fetch all macro data
        fed_task = asyncio.create_task(self._safe_fetch(self._macro.fed_funds_rate(6), []))
        cpi_task = asyncio.create_task(self._safe_fetch(self._macro.cpi(6), []))
        yields_task = asyncio.create_task(self._safe_fetch(self._macro.treasury_yields(), {}))
        vix_task = asyncio.create_task(self._safe_fetch(self._macro.vix(), None))
        unemp_task = asyncio.create_task(self._safe_fetch(self._macro.unemployment(6), []))

        fed_data = await fed_task
        cpi_data = await cpi_task
        yields = await yields_task
        vix = await vix_task
        unemployment = await unemp_task

        # Extract features
        metrics = self._extract_features(fed_data, cpi_data, yields, vix, unemployment)
        reasons: list[str] = metrics.pop("_reasons", [])

        # Weighted score
        weighted_score = self._compute_weighted_score(metrics)

        # Map to direction
        direction = self._score_to_direction(weighted_score)
        confidence = min(abs(weighted_score) / 60.0, 1.0)

        if not reasons:
            reasons.append("Insufficient macro data")

        logger.info(
            "[MacroAgent] 分析完成: 訊號=%s, 信心度=%.2f, 分數=%+.1f",
            direction.name, confidence, weighted_score,
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
        fed_data: list[dict],
        cpi_data: list[dict],
        yields: dict,
        vix: float | None,
        unemployment: list[dict],
    ) -> dict[str, Any]:
        """將原始宏觀數據轉為量化指標。"""
        metrics: dict[str, Any] = {}
        reasons: list[str] = []

        # 1. Monetary policy — Fed rate direction
        monetary_score = 0.0
        if len(fed_data) >= 2:
            current_rate = fed_data[0]["value"]
            prev_rate = fed_data[-1]["value"]
            rate_change = current_rate - prev_rate

            metrics["fed_rate"] = current_rate
            metrics["fed_rate_change"] = rate_change

            if rate_change > 0.25:
                monetary_score = -20.0  # Hiking → bearish for stocks
                reasons.append(f"Fed hiking cycle (+{rate_change:.2f}%) — headwind")
            elif rate_change < -0.25:
                monetary_score = 20.0   # Cutting → bullish for stocks
                reasons.append(f"Fed cutting cycle ({rate_change:.2f}%) — tailwind")
            else:
                monetary_score = 5.0    # Stable → mildly bullish
                reasons.append(f"Fed rate stable ({current_rate:.2f}%)")

        metrics["monetary_score"] = monetary_score

        # 2. Inflation pressure — CPI trend
        inflation_score = 0.0
        if len(cpi_data) >= 2:
            latest_cpi = cpi_data[0]["value"]
            prev_cpi = cpi_data[-1]["value"]
            cpi_change_pct = ((latest_cpi - prev_cpi) / prev_cpi) * 100 if prev_cpi > 0 else 0

            metrics["cpi_latest"] = latest_cpi
            metrics["cpi_change_pct"] = cpi_change_pct

            # Annualize (rough): 6-month change × 2
            annual_rate = cpi_change_pct * 2

            if annual_rate > 4.0:
                inflation_score = -20.0
                reasons.append(f"High inflation pressure ({annual_rate:.1f}% annualized)")
            elif annual_rate > 2.5:
                inflation_score = -5.0
                reasons.append(f"Moderate inflation ({annual_rate:.1f}% annualized)")
            elif annual_rate < 1.0:
                inflation_score = -10.0  # Deflation risk
                reasons.append(f"Low inflation risk ({annual_rate:.1f}% annualized)")
            else:
                inflation_score = 10.0
                reasons.append(f"Goldilocks inflation ({annual_rate:.1f}% annualized)")

        metrics["inflation_score"] = inflation_score

        # 3. Economic cycle — yield curve + VIX + unemployment
        cycle_score = 0.0

        # Yield curve spread
        spread = yields.get("spread_10y_2y") if yields else None
        if spread is not None:
            metrics["yield_spread"] = spread
            if spread < 0:
                cycle_score -= 15.0  # Inverted → recession signal
                reasons.append(f"Inverted yield curve ({spread:.2f}%) — recession risk")
            elif spread < 0.5:
                cycle_score -= 5.0
                reasons.append(f"Flat yield curve ({spread:.2f}%) — caution")
            else:
                cycle_score += 10.0
                reasons.append(f"Normal yield curve ({spread:.2f}%) — expansion")

        # VIX
        if vix is not None:
            metrics["vix"] = vix
            if vix > 30:
                cycle_score -= 10.0
                reasons.append(f"High VIX ({vix:.1f}) — fear elevated")
            elif vix > 20:
                cycle_score -= 3.0
                reasons.append(f"Moderate VIX ({vix:.1f})")
            else:
                cycle_score += 5.0
                reasons.append(f"Low VIX ({vix:.1f}) — complacency/calm")

        # Unemployment trend
        if len(unemployment) >= 2:
            current_unemp = unemployment[0]["value"]
            prev_unemp = unemployment[-1]["value"]
            unemp_change = current_unemp - prev_unemp
            metrics["unemployment"] = current_unemp

            if unemp_change > 0.5:
                cycle_score -= 10.0
                reasons.append(f"Rising unemployment ({current_unemp:.1f}%, +{unemp_change:.1f}pp)")
            elif unemp_change < -0.3:
                cycle_score += 10.0
                reasons.append(f"Falling unemployment ({current_unemp:.1f}%)")
            else:
                reasons.append(f"Stable unemployment ({current_unemp:.1f}%)")

        metrics["cycle_score"] = cycle_score

        metrics["_reasons"] = reasons
        return metrics

    def _compute_weighted_score(self, metrics: dict[str, Any]) -> float:
        """OPRO 動態權重融合。"""
        monetary = metrics.get("monetary_score", 0.0)
        inflation = metrics.get("inflation_score", 0.0)
        cycle = metrics.get("cycle_score", 0.0)

        weighted = (
            monetary * self.dynamic_weights["monetary_policy"]
            + inflation * self.dynamic_weights["inflation_pressure"]
            + cycle * self.dynamic_weights["economic_cycle"]
        )
        return weighted

    def update_weights_from_opro(self, new_weights: dict[str, float]) -> None:
        """OPRO 介面。"""
        for key in self._DEFAULT_STRATEGY_WEIGHTS:
            if key in new_weights:
                self.dynamic_weights[key] = new_weights[key]
        logger.info("[MacroAgent] OPRO 權重更新: %s", self.dynamic_weights)

    async def _safe_fetch(self, coro: Any, default: Any) -> Any:
        """安全包裝：失敗時返回 default。"""
        try:
            return await coro
        except Exception:
            self.logger.warning("宏觀數據取得失敗", exc_info=True)
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
