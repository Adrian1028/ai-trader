"""
Intelligence Orchestrator — 情報統籌大腦
==========================================
平行呼叫所有情報代理 (Fan-out)，收集 AnalysisSignal，
進行加權融合與分歧懲罰 (Fan-in)，輸出統一的 MarketView。

核心職責：
  1. 平行運算：asyncio.gather 同時呼叫 Fundamental / Technical / Sentiment
  2. 加權融合：根據 OPRO 動態權重計算綜合分數
  3. 分歧懲罰：代理間意見相左時，自動扣減信心度
  4. 安全降級：單一代理崩潰不影響其他代理的結果

資料流：
  IntelligenceOrchestrator.evaluate(context)
  │
  ├── Fan-out: agent.safe_analyse(context) × N  (並行)
  │   ├── FundamentalAgent → AnalysisSignal
  │   ├── TechnicalAgent   → AnalysisSignal
  │   └── SentimentAgent   → AnalysisSignal
  │
  └── Fan-in: _fuse_signals() → MarketView
      ├── 加權融合 (weighted score)
      ├── 分歧懲罰 (disagreement penalty)
      └── 方向判定 (score → SignalDirection)

使用者：
  - SystemOrchestrator (src/core/orchestrator.py)
  - OPRO (update_weights → 動態調整代理權重)
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src.core.base_agent import AnalysisSignal, BaseAgent, SignalDirection
from src.core.regime_detector import RegimeSnapshot

logger = logging.getLogger(__name__)


@dataclass
class MarketView:
    """
    Fused output from all intelligence agents for one instrument.
    這是情報統籌大腦對一檔標的的最終判決。
    """
    isin: str
    ticker: str                   # standard market ticker (AAPL, AMD)
    t212_ticker: str = ""         # Trading 212 internal code (AAPL_US_EQ)
    signals: list[AnalysisSignal] = field(default_factory=list)
    fused_score: float = 0.0
    fused_direction: SignalDirection = SignalDirection.NEUTRAL
    fused_confidence: float = 0.0
    divergence_detected: bool = False
    regime: RegimeSnapshot | None = None  # Phase 5: injected by system orchestrator
    timestamp: float = field(default_factory=time.time)

    @property
    def summary(self) -> str:
        parts = [
            f"[{s.source}] {s.direction.name} (conf={s.confidence:.2f})"
            for s in self.signals
        ]
        divergence_tag = " [DIVERGENCE]" if self.divergence_detected else ""
        return (
            f"{self.ticker} → {self.fused_direction.name} "
            f"(score={self.fused_score:+.2f}, "
            f"conf={self.fused_confidence:.2f}){divergence_tag} | "
            + " | ".join(parts)
        )

    @property
    def reasons(self) -> list[str]:
        """Collect reasoning from all contributing agents."""
        return [
            f"[{s.source}: {s.direction.name}] {s.reasoning}"
            for s in self.signals
            if s.reasoning
        ]


# ── 預設融合權重 ──────────────────────────────────────────────
# 後續可由 Adaptive-OPRO 動態調整，適應不同市場體制
_DEFAULT_WEIGHTS = {
    "fundamental": 0.20,           # 長期內在價值
    "technical": 0.20,             # 短期價格趨勢與動能
    "sentiment": 0.15,             # 市場情緒與新聞解讀
    "gemini_strategist": 0.15,     # LLM 語義推理（跨因子綜合判斷）
    "macro": 0.10,                 # 宏觀經濟環境（Phase 6）
    "options_flow": 0.08,          # 期權異常流量（Phase 6）
    "insider": 0.05,               # 內部人交易信號（Phase 6）
    "social_sentiment": 0.07,      # 社群情緒動量（Phase 6）
}

# ── 分歧懲罰參數 ──────────────────────────────────────────────
_DIVERGENCE_PENALTY = 0.15   # 偵測到多空分歧時，扣減 15% 信心分數


class IntelligenceOrchestrator:
    """
    情報統籌大腦 (Intelligence Orchestrator)

    負責平行呼叫所有情報代理，收集 AnalysisSignal，
    並進行加權融合與分歧懲罰。
    最終輸出統一的 MarketView 給 RiskAgent。

    設計原則：
      - 單一代理崩潰不會導致整個大腦當機（safe_analyse 包裝）
      - 代理間意見分歧時自動降低信心度（防止過度自信）
      - 所有權重都可由 OPRO 動態調整（AI 自我進化）
    """

    def __init__(
        self,
        agents: list[BaseAgent],
        weights: dict[str, float] | None = None,
    ) -> None:
        self._agents = agents
        self._weights = dict(weights) if weights else dict(_DEFAULT_WEIGHTS)

    # ══════════════════════════════════════════════════════════════
    # 主要介面
    # ══════════════════════════════════════════════════════════════

    async def evaluate(self, context: dict[str, Any]) -> MarketView:
        """
        平行扇出 (Fan-out) 收集所有代理的訊號並融合。

        Parameters
        ----------
        context : dict
            Must contain "isin" and "ticker" at minimum.

        Returns
        -------
        MarketView : 融合後的市場觀點
        """
        ticker = context.get("ticker", "")
        isin = context.get("isin", "")
        t212_ticker = context.get("t212_ticker", ticker)

        logger.info(
            "[Orchestrator] 開始對 %s 進行多代理聯合分析 (共 %d 個代理)...",
            ticker, len(self._agents),
        )

        # ── Step 1: Fan-out — 平行執行規則型代理 ─────────────────
        # Separate rule-based agents from LLM agent so we can feed
        # rule-based signals to the LLM as context.
        rule_agents = [a for a in self._agents if a.name != "gemini_strategist"]
        llm_agents = [a for a in self._agents if a.name == "gemini_strategist"]

        tasks = [
            asyncio.create_task(agent.safe_analyse(context))
            for agent in rule_agents
        ]
        rule_signals: list[AnalysisSignal] = await asyncio.gather(*tasks)

        # ── Step 1b: Run LLM agent with prior signals injected ────
        llm_signals: list[AnalysisSignal] = []
        if llm_agents:
            enriched = dict(context)
            enriched["prior_signals"] = [
                {
                    "source": s.source,
                    "direction": s.direction.name,
                    "confidence": s.confidence,
                }
                for s in rule_signals
            ]
            llm_tasks = [
                asyncio.create_task(agent.safe_analyse(enriched))
                for agent in llm_agents
            ]
            llm_signals = await asyncio.gather(*llm_tasks)

        signals = rule_signals + llm_signals

        # ── Step 2: Fan-in — 加權融合 + 分歧懲罰 ──────────────
        view = self._fuse_signals(isin, ticker, signals, t212_ticker=t212_ticker)

        logger.info("[Orchestrator] MarketView: %s", view.summary)
        return view

    async def evaluate_batch(
        self,
        contexts: list[dict[str, Any]],
        max_concurrency: int = 5,
    ) -> list[MarketView]:
        """
        批次評估多檔標的，限制並行數避免 API 過載。

        Parameters
        ----------
        contexts : list[dict]
            每個 dict 都包含 "isin" 和 "ticker"
        max_concurrency : int
            最大同時評估數量（預設 5）
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded(ctx: dict[str, Any]) -> MarketView:
            async with semaphore:
                return await self.evaluate(ctx)

        return await asyncio.gather(
            *[_bounded(ctx) for ctx in contexts],
        )

    # ══════════════════════════════════════════════════════════════
    # 加權融合引擎
    # ══════════════════════════════════════════════════════════════

    def _fuse_signals(
        self,
        isin: str,
        ticker: str,
        signals: list[AnalysisSignal],
        t212_ticker: str = "",
    ) -> MarketView:
        """
        將多個 AnalysisSignal 融合成單一 MarketView。

        流程：
          1. 加權融合 — 各代理的 weighted_score × 代理權重
          2. 分歧偵測 — 檢查是否有多空分歧
          3. 分歧懲罰 — 有分歧時扣減信心度
          4. 方向判定 — 融合分數 → SignalDirection
        """
        # ── 過濾有效信號 ──────────────────────────────────────
        active_signals = [
            s for s in signals if s.confidence > 0
        ]

        if not active_signals:
            logger.warning(
                "[Orchestrator] %s 所有代理均未能回傳有效訊號 "
                "(API 可能全部限流)", ticker,
            )
            return MarketView(
                isin=isin, ticker=ticker, t212_ticker=t212_ticker,
                signals=signals,
            )

        # ── 1. 加權融合 ──────────────────────────────────────
        fused_score = 0.0
        total_weight = 0.0

        for signal in signals:
            w = self._weights.get(
                signal.source,
                1.0 / len(self._agents) if self._agents else 0.0,
            )
            fused_score += signal.weighted_score * w
            total_weight += w

        # 正規化（避免部分代理缺失導致分數失真）
        if total_weight > 0:
            fused_score /= total_weight

        # ── 2. 信心度計算 ─────────────────────────────────────
        confidences = [s.confidence for s in signals if s.confidence > 0]
        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        # ── 3. 分歧偵測與懲罰 ────────────────────────────────
        # 計算代理間方向分歧度
        directions = [
            s.direction.value for s in signals if s.confidence > 0.1
        ]
        divergence_detected = False

        if directions:
            direction_spread = max(directions) - min(directions)
            # spread=0 → 完全一致, spread=4 → 強買vs強賣（最大分歧）

            if direction_spread >= 2:
                # 存在多空分歧（至少一個 BUY 和一個 SELL）
                divergence_detected = True

            # 連續性衰減：spread 越大 → agreement_factor 越小
            agreement_factor = max(
                1.0 - _DIVERGENCE_PENALTY,
                1.0 - direction_spread / 4.0,
            )
        else:
            agreement_factor = 0.5

        fused_confidence = min(avg_confidence * agreement_factor, 1.0)

        if divergence_detected:
            logger.info(
                "[Orchestrator] %s 偵測到代理間多空分歧，"
                "信心由 %.2f 下調至 %.2f (懲罰係數=%.2f)",
                ticker, avg_confidence, fused_confidence, agreement_factor,
            )

        # ── 4. 方向判定 ──────────────────────────────────────
        fused_direction = self._score_to_direction(fused_score)

        return MarketView(
            isin=isin,
            ticker=ticker,
            t212_ticker=t212_ticker,
            signals=signals,
            fused_score=fused_score,
            fused_direction=fused_direction,
            fused_confidence=fused_confidence,
            divergence_detected=divergence_detected,
        )

    # ══════════════════════════════════════════════════════════════
    # OPRO 介面 — AI 自我進化入口
    # ══════════════════════════════════════════════════════════════

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """
        【機器學習介面】
        由 Adaptive-OPRO 模組呼叫，動態調整各代理的融合權重。

        在不同市場體制下，代理的重要性不同：
          - 牛市狂飆期：technical 權重 ↑ (趨勢追蹤更有效)
          - 恐慌崩盤期：sentiment 權重 ↑ (情緒主導行情)
          - 震盪整理期：fundamental 權重 ↑ (回歸價值)

        Parameters
        ----------
        new_weights : dict
            e.g. {"fundamental": 0.2, "technical": 0.5, "sentiment": 0.3}
        """
        self._weights.update(new_weights)
        logger.info(
            "[Orchestrator] 代理融合權重已更新: %s", self._weights,
        )

    # ══════════════════════════════════════════════════════════════
    # 輔助方法
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _score_to_direction(score: float) -> SignalDirection:
        """
        將融合分數映射到信號方向。

        注意：融合分數範圍約 -2.0 到 +2.0
        （因為 weighted_score = direction.value × confidence，
        且 direction 可達 ±2 for STRONG_BUY/SELL）
        """
        if score >= 1.2:
            return SignalDirection.STRONG_BUY
        if score >= 0.4:
            return SignalDirection.BUY
        if score <= -1.2:
            return SignalDirection.STRONG_SELL
        if score <= -0.4:
            return SignalDirection.SELL
        return SignalDirection.NEUTRAL
