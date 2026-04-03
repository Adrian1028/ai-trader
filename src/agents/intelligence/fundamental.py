"""
Fundamental Analysis Agent — AI 基本面分析代理
================================================
抓取企業概況與財報，將生硬的 P/E、ROE 與營收成長率
轉化為可量化的 AI 特徵，並利用 OPRO 動態權重算出最終的投資價值訊號。

三大策略維度（由 OPRO 動態調整權重）：
  1. 估值 (Value)        — P/E、Forward P/E、P/B、EV/EBITDA
  2. 盈利能力 (Profitability) — ROE、淨利率、EPS
  3. 成長潛力 (Growth)    — 營收成長率、EPS 成長趨勢、盈餘驚喜

補充維度（不受權重影響，作為額外信號）：
  - 盈餘驚喜連勝/連敗 (Earnings Surprise Streak)
  - Forward P/E vs Trailing P/E 預期變化

使用者：
  - IntelligenceOrchestrator：多代理信號融合
  - CognitiveLoop：績效反饋 → 權重調整

數據來源：
  - Alpha Vantage：公司概覽 + 盈餘數據
  - Intrinio：深度財務報表 (降級/補充)

API 限制：
  - AV 免費版 25 次/日 → 公司概覽 + 盈餘共用 2 次 quota
  - Intrinio 可選，抓取失敗不影響主流程
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.core.base_agent import AnalysisSignal, BaseAgent, SignalDirection
from src.data.providers.alpha_vantage import AlphaVantageProvider

logger = logging.getLogger(__name__)


class FundamentalAgent(BaseAgent):
    """
    基本面分析代理 (Fundamental Agent)
    負責抓取企業財報與概況數據，計算估值、盈利能力與成長潛力，
    判斷公司長期內在價值。

    就像系統裡內建的一位「價值投資經理人」。

    指標分三大策略維度：
      1. 估值 (value)           — P/E、P/B、EV/EBITDA
      2. 盈利能力 (profitability) — ROE、淨利率
      3. 成長潛力 (growth)       — 營收成長率、EPS 成長趨勢
    """

    # ── 預設 OPRO 動態權重 ──────────────────────────────────────
    _DEFAULT_STRATEGY_WEIGHTS = {
        "value": 0.40,           # 估值權重 (P/E, P/B, EV/EBITDA)
        "profitability": 0.40,   # 盈利能力權重 (ROE, 淨利率)
        "growth": 0.20,          # 成長潛力權重 (營收成長率)
    }

    def __init__(
        self,
        alpha_vantage: AlphaVantageProvider,
        intrinio: Any | None = None,
    ) -> None:
        """
        Parameters
        ----------
        alpha_vantage : AlphaVantageProvider
            主要數據源（公司概覽 + 盈餘）
        intrinio : IntrinioProvider | None
            可選的深度財務數據源（降級/補充）
        """
        super().__init__("fundamental")
        self._av = alpha_vantage
        self._intrinio = intrinio
        # OPRO 動態權重 — 初始為平衡值，後續由機器學習迴圈調整
        self.dynamic_weights = dict(self._DEFAULT_STRATEGY_WEIGHTS)

    # ══════════════════════════════════════════════════════════════
    # 主分析流程
    # ══════════════════════════════════════════════════════════════

    async def analyse(self, context: dict[str, Any]) -> AnalysisSignal:
        """
        執行完整的基本面分析與 AI 推理流程。

        Parameters
        ----------
        context : dict
            Must contain "ticker" (str). Optional: "isin", "current_price".

        Returns
        -------
        AnalysisSignal : 標準化信號 (direction, confidence, reasoning, data)
        """
        ticker = context["ticker"]

        # ── Step 1: 抓取基本面數據 ─────────────────────────────
        overview, earnings_data = await self._fetch_data(ticker)

        if not overview:
            return AnalysisSignal(
                source=self.name,
                reasoning="無法獲取企業基本面資料",
            )

        # ── Step 2: 特徵提取 ───────────────────────────────────
        metrics = self._extract_metrics(overview)
        earnings_history = self._extract_earnings(earnings_data)

        # ── Step 3: OPRO 動態權重融合 ──────────────────────────
        weighted_score = self._compute_weighted_score(metrics)

        # ── Step 4: 補充維度 (非權重) ──────────────────────────
        bonus_score, reasons = self._compute_bonus_and_reasons(
            metrics, earnings_history, overview,
        )

        # ── Step 5: 最終合成 ───────────────────────────────────
        composite_score = weighted_score + bonus_score
        composite_score = max(-100.0, min(100.0, composite_score))

        direction = self._score_to_direction(composite_score)
        confidence = min(abs(composite_score) / 30.0, 1.0)
        # Floor: if we got data and computed a score, confidence >= 0.20
        if abs(composite_score) > 1.0:
            confidence = max(confidence, 0.20)

        if not reasons:
            reasons.append("Insufficient fundamental data for scoring")

        logger.info(
            "[FundamentalAgent] %s 分析完成: 訊號=%s, 信心度=%.2f, "
            "加權分=%+.1f (估值/盈利/成長 動態融合)",
            ticker, direction.name, confidence, composite_score,
        )

        return AnalysisSignal(
            source=self.name,
            direction=direction,
            confidence=confidence,
            reasoning=" | ".join(reasons),
            data={
                "composite_score": composite_score,
                "weighted_score": weighted_score,
                "bonus_score": bonus_score,
                "dynamic_weights": dict(self.dynamic_weights),
                "pe_ratio": metrics.get("pe_ratio"),
                "forward_pe": metrics.get("forward_pe"),
                "pb_ratio": metrics.get("pb_ratio"),
                "ev_to_ebitda": metrics.get("ev_to_ebitda"),
                "roe": metrics.get("roe"),
                "profit_margin": metrics.get("profit_margin"),
                "eps": metrics.get("eps"),
                "revenue_growth": metrics.get("revenue_growth"),
                "earnings_surprise_pct": (
                    earnings_history[0].get("surprise_pct")
                    if earnings_history else None
                ),
            },
        )

    # ══════════════════════════════════════════════════════════════
    # 資料取得
    # ══════════════════════════════════════════════════════════════

    async def _fetch_data(
        self, ticker: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        並行抓取公司概覽 + 盈餘數據。
        Intrinio 作為可選補充（失敗不影響主流程）。
        """
        overview_task = asyncio.create_task(
            self._safe_fetch(self._av.company_overview(ticker)),
        )
        earnings_task = asyncio.create_task(
            self._safe_fetch(self._av.earnings(ticker)),
        )

        overview = await overview_task
        earnings = await earnings_task

        return overview or {}, earnings or {}

    async def _safe_fetch(self, coro: Any) -> dict[str, Any] | None:
        """安全包裝：失敗時返回 None 而非拋出例外"""
        try:
            return await coro
        except Exception:
            self.logger.warning("數據取得失敗", exc_info=True)
            return None

    # ══════════════════════════════════════════════════════════════
    # 特徵提取
    # ══════════════════════════════════════════════════════════════

    def _extract_metrics(self, overview: dict[str, Any]) -> dict[str, float]:
        """
        將 Alpha Vantage 公司概覽的數據安全轉換為浮點數指標。

        注意：company_overview() 已經做過 _parse_overview()，
        所以 PERatio 等欄位已經是 float 而非字串。
        """
        return {
            "pe_ratio": self._safe_float(overview.get("PERatio")),
            "forward_pe": self._safe_float(overview.get("ForwardPE")),
            "pb_ratio": self._safe_float(overview.get("PriceToBookRatio")),
            "ev_to_ebitda": self._safe_float(overview.get("EVToEBITDA")),
            "roe": self._safe_float(overview.get("ReturnOnEquityTTM")),
            "profit_margin": self._safe_float(overview.get("ProfitMargin")),
            "eps": self._safe_float(overview.get("EPS")),
            "revenue_growth": self._safe_float(
                overview.get("QuarterlyRevenueGrowthYOY"),
            ),
            "dividend_yield": self._safe_float(
                overview.get("DividendYield"),
            ),
        }

    def _extract_earnings(
        self, earnings_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """提取季度盈餘歷史"""
        quarterly = earnings_data.get("quarterlyEarnings", [])
        result = []
        for q in quarterly:
            result.append({
                "date": q.get("reportedDate", ""),
                "reported_eps": self._safe_float(q.get("reportedEPS")),
                "estimated_eps": self._safe_float(q.get("estimatedEPS")),
                "surprise_pct": self._safe_float(
                    q.get("surprisePercentage"),
                ),
            })
        return result

    # ══════════════════════════════════════════════════════════════
    # OPRO 動態權重計算
    # ══════════════════════════════════════════════════════════════

    def _compute_weighted_score(self, metrics: dict[str, float]) -> float:
        """
        利用 OPRO 動態權重進行策略維度評分。

        三大策略維度：
          1. 估值 (value)           — P/E、P/B、EV/EBITDA
          2. 盈利能力 (profitability) — ROE、淨利率
          3. 成長潛力 (growth)       — 營收成長率

        如果系統發現近期買「低 P/E (價值)」的股票一直賠錢，
        但買「高營收成長 (成長)」的股票大賺，
        OPRO 就會自動調高 growth 的權重，並降低 value 的權重。

        Returns
        -------
        float : 加權分數，大約在 -50 到 +50 之間
        """
        # ── 估值策略分數 (Value) ──────────────────────────────
        val_score = 0.0
        pe = metrics.get("pe_ratio", 0.0)

        if 0 < pe <= 15:
            val_score += 20.0    # 估值便宜 (低本益比)
        elif 15 < pe <= 25:
            val_score += 5.0     # 估值合理
        elif pe > 30:
            val_score -= 15.0    # 估值過高 (泡沫風險)
        elif pe <= 0:
            val_score -= 20.0    # 公司處於虧損狀態

        # P/B 補充
        pb = metrics.get("pb_ratio", 0.0)
        if 0 < pb < 1.0:
            val_score += 10.0    # 股價低於帳面價值
        elif pb > 5.0:
            val_score -= 5.0     # 可能過高

        # EV/EBITDA 補充
        ev_ebitda = metrics.get("ev_to_ebitda", 0.0)
        if 0 < ev_ebitda < 10:
            val_score += 10.0
        elif ev_ebitda > 20:
            val_score -= 10.0

        # 正規化到 ±25 範圍
        val_score = max(-25.0, min(25.0, val_score))

        # ── 盈利能力策略分數 (Profitability) ──────────────────
        prof_score = 0.0
        roe = metrics.get("roe", 0.0)

        if roe >= 0.15:
            prof_score += 20.0   # ROE >= 15% 是優秀公司的指標
        elif roe >= 0.08:
            prof_score += 8.0    # 尚可接受
        elif roe < 0:
            prof_score -= 20.0   # 嚴重虧損

        # 淨利率補充
        margin = metrics.get("profit_margin", 0.0)
        if margin > 0.20:
            prof_score += 10.0   # 高毛利護城河
        elif margin < 0.05 and margin >= 0:
            prof_score -= 5.0    # 薄利
        elif margin < 0:
            prof_score -= 10.0   # 淨虧損

        # 正規化到 ±25 範圍
        prof_score = max(-25.0, min(25.0, prof_score))

        # ── 成長潛力策略分數 (Growth) ─────────────────────────
        growth_score = 0.0
        rev_growth = metrics.get("revenue_growth", 0.0)

        if rev_growth >= 0.20:
            growth_score += 25.0   # 營收高成長 (>=20%)
        elif rev_growth >= 0.05:
            growth_score += 10.0   # 營收穩定成長
        elif rev_growth < -0.10:
            growth_score -= 25.0   # 營收嚴重衰退
        elif rev_growth < 0:
            growth_score -= 10.0   # 營收小幅衰退

        # 正規化到 ±25 範圍
        growth_score = max(-25.0, min(25.0, growth_score))

        # ── 加權融合 ──────────────────────────────────────────
        weighted = (
            val_score * self.dynamic_weights["value"]
            + prof_score * self.dynamic_weights["profitability"]
            + growth_score * self.dynamic_weights["growth"]
        )

        return weighted

    # ══════════════════════════════════════════════════════════════
    # 補充維度 (非權重) + 原因生成
    # ══════════════════════════════════════════════════════════════

    def _compute_bonus_and_reasons(
        self,
        metrics: dict[str, float],
        earnings_history: list[dict[str, Any]],
        overview: dict[str, Any],
    ) -> tuple[float, list[str]]:
        """
        計算非策略維度的補充分數 + 生成人類可讀原因。

        Returns
        -------
        (bonus_score, reasons)
        """
        bonus = 0.0
        reasons: list[str] = []

        pe = metrics.get("pe_ratio", 0.0)
        forward_pe = metrics.get("forward_pe", 0.0)
        roe = metrics.get("roe", 0.0)
        margin = metrics.get("profit_margin", 0.0)
        ev_ebitda = metrics.get("ev_to_ebitda", 0.0)
        rev_growth = metrics.get("revenue_growth", 0.0)

        # ── Forward P/E vs Trailing P/E ───────────────────────
        if pe > 0 and forward_pe > 0:
            if forward_pe < pe * 0.85:
                bonus += 10
                reasons.append(
                    f"Forward P/E ({forward_pe:.1f}) < trailing ({pe:.1f}): "
                    f"earnings growth expected"
                )
            elif forward_pe > pe * 1.15:
                bonus -= 10
                reasons.append(
                    f"Forward P/E ({forward_pe:.1f}) > trailing ({pe:.1f}): "
                    f"earnings contraction"
                )

        # ── 各維度原因 ────────────────────────────────────────
        if pe > 0:
            if pe <= 15:
                reasons.append(f"Low P/E ({pe:.1f}): attractively valued")
            elif pe > 30:
                reasons.append(f"High P/E ({pe:.1f}): may be overvalued")

        if roe >= 0.15:
            reasons.append(f"Strong ROE ({roe:.0%})")
        elif roe < 0.05 and roe != 0.0:
            reasons.append(f"Weak ROE ({roe:.0%})")

        if margin > 0.20:
            reasons.append(f"High profit margin ({margin:.0%})")
        elif margin < 0.05 and margin != 0.0:
            reasons.append(f"Thin profit margin ({margin:.0%})")

        if ev_ebitda > 20:
            reasons.append(
                f"High EV/EBITDA ({ev_ebitda:.1f}): expensive or leveraged"
            )
        elif 0 < ev_ebitda < 10:
            reasons.append(f"Reasonable EV/EBITDA ({ev_ebitda:.1f})")

        if rev_growth >= 0.20:
            reasons.append(f"Strong revenue growth ({rev_growth:.0%} YoY)")
        elif rev_growth < -0.10:
            reasons.append(f"Revenue declining ({rev_growth:.0%} YoY)")

        # ── 盈餘驚喜連勝/連敗 ────────────────────────────────
        if len(earnings_history) >= 3:
            recent_surprises = [
                e["surprise_pct"] for e in earnings_history[:4]
                if e.get("surprise_pct") is not None
            ]
            if recent_surprises:
                beats = sum(1 for s in recent_surprises if s > 0)
                misses = sum(1 for s in recent_surprises if s < 0)
                if beats >= 3:
                    bonus += 10
                    reasons.append(
                        f"Earnings beat streak ({beats}/{len(recent_surprises)} quarters)"
                    )
                elif misses >= 3:
                    bonus -= 10
                    reasons.append(
                        f"Earnings miss streak ({misses}/{len(recent_surprises)} quarters)"
                    )

                # 最近一次驚喜
                last = recent_surprises[0]
                if last > 5:
                    bonus += 5
                    reasons.append(
                        f"Last earnings beat by {last:.1f}%"
                    )
                elif last < -5:
                    bonus -= 5
                    reasons.append(
                        f"Last earnings missed by {abs(last):.1f}%"
                    )

        # ── EPS 連續成長/衰退 ─────────────────────────────────
        annual = overview.get("annualEarnings") or []
        if not annual:
            # 從 earnings_data 的 annualEarnings 也可能在 overview 之外
            pass  # 已在 _fetch_data 中取得

        return bonus, reasons

    # ══════════════════════════════════════════════════════════════
    # OPRO 介面 — AI 自我進化入口
    # ══════════════════════════════════════════════════════════════

    def update_weights_from_opro(self, new_weights: dict[str, float]) -> None:
        """
        【機器學習介面】
        由 Adaptive-OPRO 模組呼叫。

        如果系統發現近期買「低 P/E (價值)」的股票一直賠錢，
        但買「高營收成長 (成長)」的股票大賺，
        OPRO 就會自動調高 growth 的權重，並降低 value 的權重。

        Parameters
        ----------
        new_weights : dict
            e.g. {"value": 0.2, "profitability": 0.3, "growth": 0.5}
            只更新提供的 key，未提供的 key 保持不變。
        """
        for key in ("value", "profitability", "growth"):
            if key in new_weights:
                self.dynamic_weights[key] = new_weights[key]

        logger.info(
            "[FundamentalAgent] 接收到 OPRO 的機器學習權重更新: %s",
            self.dynamic_weights,
        )

    # ══════════════════════════════════════════════════════════════
    # 輔助方法
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _safe_float(value: Any) -> float:
        """安全轉換為 float，處理 'None', '-', '' 等異常值"""
        if value is None or value == "None" or value == "-" or value == "":
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _score_to_direction(score: float) -> SignalDirection:
        """將數值分數映射到信號方向。"""
        if score >= 25:
            return SignalDirection.STRONG_BUY
        if score >= 5:
            return SignalDirection.BUY
        if score <= -25:
            return SignalDirection.STRONG_SELL
        if score <= -5:
            return SignalDirection.SELL
        return SignalDirection.NEUTRAL
