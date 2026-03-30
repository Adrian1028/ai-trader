"""
Technical Analysis Agent — AI 技術分析代理
==========================================
**多時間框架 (Multi-Timeframe)** 技術評分 + OPRO 動態權重融合。

三層時間框架分析：
  - 週線 (Weekly)：定義大趨勢方向 — 順勢交易的「大方向」
  - 日線 (Daily)：主要交易決策層 — 完整指標計算
  - 小時線 (Hourly)：進場確認層 — 短期動能驗證

多時間框架共振 (Confluence)：
  - 三框架同向 → 高信心 (confluence_bonus = +20)
  - 二框架同向 → 標準信心 (無 bonus)
  - 框架分歧 → 降信心 (confluence_penalty = -15)

指標體系：
  - 順勢交易 (Trend-Following)：SMA 交叉 (50/200)、價格 vs SMA-200
  - 均值回歸 (Mean-Reversion)：RSI(14)
  - 動量 (Momentum)：MACD 柱狀圖方向
  - 波動率 (Volatility)：布林通道寬度/位置、ATR 體制
  - 成交量 (Volume)：量能突增偵測

OPRO 動態權重：
  - 順勢/均值回歸/波動率三大策略維度各有獨立權重
  - 若近期「均值回歸」策略一直虧錢，OPRO 會自動降低其權重
  - update_weights_from_opro() 是 AI 自我進化的入口

使用者：
  - IntelligenceOrchestrator：多代理信號融合
  - RiskAgent：ATR 用於止損/倉位計算
  - CognitiveLoop：績效反饋 → 權重調整
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.core.base_agent import AnalysisSignal, BaseAgent, SignalDirection
from src.data.providers.alpha_vantage import AlphaVantageProvider
from src.data.providers.polygon import PolygonProvider
from src.data.providers.base_provider import OHLCVBar

logger = logging.getLogger(__name__)


class TechnicalAgent(BaseAgent):
    """
    技術分析代理 (Technical Agent)
    將歷史 K 線轉化為數學指標，並透過 OPRO 動態權重進行 AI 語義推理。

    指標分三大策略維度：
      1. 順勢交易 (trend_following)：SMA 交叉、價格位置
      2. 均值回歸 (mean_reversion)：RSI 超買超賣
      3. 波動率 (volatility)：布林通道位置、ATR 體制

    額外計算（不受權重影響，作為補充資訊）：
      - MACD 柱狀圖動量
      - 成交量異常偵測
    """

    # ── 預設 OPRO 動態權重 ──────────────────────────────────────
    _DEFAULT_STRATEGY_WEIGHTS = {
        "trend_following": 0.40,   # 順勢交易 (均線)
        "mean_reversion": 0.40,    # 均值回歸 (RSI)
        "volatility": 0.20,        # 波動率 (布林通道/ATR)
    }

    def __init__(
        self,
        alpha_vantage: AlphaVantageProvider,
        polygon: PolygonProvider,
    ) -> None:
        super().__init__("technical")
        self._av = alpha_vantage
        self._polygon = polygon
        # OPRO 動態權重 — 初始為平衡值，後續由機器學習迴圈調整
        self.dynamic_weights = dict(self._DEFAULT_STRATEGY_WEIGHTS)

    # ══════════════════════════════════════════════════════════════
    # 主分析流程
    # ══════════════════════════════════════════════════════════════

    async def analyse(self, context: dict[str, Any]) -> AnalysisSignal:
        """
        執行多時間框架技術分析與 AI 推理流程。

        分析順序：
          1. 週線 — 判斷大趨勢方向
          2. 日線 — 主要交易決策（完整指標）
          3. 小時線 — 短期動能確認
          4. 多時間框架共振檢查

        Parameters
        ----------
        context : dict
            Must contain "ticker" (str). Optional: "isin", "current_price".

        Returns
        -------
        AnalysisSignal : 標準化信號 (direction, confidence, reasoning, data)
        """
        ticker = context["ticker"]

        # ══════════════════════════════════════════════════════════
        # Step 1: 並行抓取三個時間框架的 K 線資料
        # ══════════════════════════════════════════════════════════
        import asyncio
        daily_task = asyncio.create_task(self._fetch_daily(ticker))
        weekly_task = asyncio.create_task(self._fetch_weekly(ticker))
        hourly_task = asyncio.create_task(self._fetch_hourly(ticker))

        daily_bars, weekly_bars, hourly_bars = await asyncio.gather(
            daily_task, weekly_task, hourly_task,
        )

        if not daily_bars:
            return AnalysisSignal(
                source=self.name,
                reasoning="無法獲取日線 K 線資料",
            )

        # ══════════════════════════════════════════════════════════
        # Step 2: 各時間框架獨立指標計算
        # ══════════════════════════════════════════════════════════
        daily_ind = self._bars_to_indicators(daily_bars)
        weekly_ind = self._bars_to_indicators(weekly_bars) if weekly_bars else {}
        hourly_ind = self._bars_to_indicators(hourly_bars) if hourly_bars else {}

        # 日線是主要決策層 — 完整計算
        reasons: list[str] = daily_ind.pop("_reasons", [])
        bonus_score = daily_ind.get("_bonus_score", 0.0)
        weighted_score = self._compute_weighted_score(daily_ind)

        # ══════════════════════════════════════════════════════════
        # Step 3: 多時間框架共振分析 (Multi-Timeframe Confluence)
        # ══════════════════════════════════════════════════════════
        daily_bias = self._timeframe_bias(daily_ind)
        weekly_bias = self._timeframe_bias(weekly_ind) if weekly_ind else 0.0
        hourly_bias = self._timeframe_bias(hourly_ind) if hourly_ind else 0.0

        confluence_score, confluence_reasons = self._multi_timeframe_confluence(
            weekly_bias=weekly_bias,
            daily_bias=daily_bias,
            hourly_bias=hourly_bias,
        )
        reasons.extend(confluence_reasons)

        # ══════════════════════════════════════════════════════════
        # Step 4: 綜合計算 — 日線分數 + bonus + 共振分數
        # ══════════════════════════════════════════════════════════
        composite_score = weighted_score + bonus_score + confluence_score
        composite_score = max(-100.0, min(100.0, composite_score))

        direction = self._score_to_direction(composite_score)
        confidence = min(abs(composite_score) / 80.0, 1.0)

        logger.info(
            "[TechnicalAgent] %s MTF分析完成: 訊號=%s, 信心度=%.2f, "
            "加權分=%+.1f (日=%+.1f, 共振=%+.1f) "
            "| 週線偏向=%+.1f 日線=%+.1f 時線=%+.1f",
            ticker, direction.name, confidence, composite_score,
            weighted_score + bonus_score, confluence_score,
            weekly_bias, daily_bias, hourly_bias,
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
                "confluence_score": confluence_score,
                "dynamic_weights": dict(self.dynamic_weights),
                # 多時間框架偏向
                "weekly_bias": weekly_bias,
                "daily_bias": daily_bias,
                "hourly_bias": hourly_bias,
                # 日線指標
                "rsi": daily_ind.get("rsi_14"),
                "atr": daily_ind.get("atr"),
                "atr_pct": daily_ind.get("atr_pct"),
                "bb_width": daily_ind.get("bb_width"),
                "bb_position": daily_ind.get("bb_position"),
                "price_to_sma20_pct": daily_ind.get("price_to_sma20_pct"),
                "sma_50": daily_ind.get("sma_50"),
                "sma_200": daily_ind.get("sma_200"),
                "current_price": daily_ind.get("current_price"),
                # 週線 / 小時線摘要
                "weekly_rsi": weekly_ind.get("rsi_14"),
                "hourly_rsi": hourly_ind.get("rsi_14"),
            },
        )

    # ══════════════════════════════════════════════════════════════
    # 多時間框架共振分析
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _timeframe_bias(ind: dict[str, Any]) -> float:
        """
        從指標字典中提取該時間框架的多空偏向 (-1.0 到 +1.0)。

        綜合考量：
          - SMA 交叉方向
          - 價格相對 SMA 位置
          - RSI 偏向

        Returns
        -------
        float : 正值=看多, 負值=看空, 接近0=中性
        """
        if not ind:
            return 0.0

        score = 0.0
        count = 0

        # SMA 交叉
        if ind.get("sma_50_above_200") is not None:
            score += 1.0 if ind["sma_50_above_200"] else -1.0
            count += 1

        # 價格 vs SMA200
        if ind.get("price_above_sma200") is not None:
            score += 1.0 if ind["price_above_sma200"] else -1.0
            count += 1
        elif ind.get("price_to_sma20_pct") is not None:
            score += 1.0 if ind["price_to_sma20_pct"] > 0 else -1.0
            count += 1

        # RSI 偏向
        rsi = ind.get("rsi_14")
        if rsi is not None:
            if rsi > 60:
                score += 0.5
            elif rsi < 40:
                score -= 0.5
            count += 1

        return score / max(count, 1)

    @staticmethod
    def _multi_timeframe_confluence(
        weekly_bias: float,
        daily_bias: float,
        hourly_bias: float,
    ) -> tuple[float, list[str]]:
        """
        多時間框架共振分析 — 檢查三個時間框架是否同方向。

        規則：
          - 三框架同向 (all > 0 or all < 0) → bonus +20
          - 二框架同向 → 標準 (no bonus)
          - 三框架分歧 (有多有空) → penalty -15
          - 任何框架無資料 → 不懲罰，只用可用框架

        Parameters
        ----------
        weekly_bias : 週線偏向 (-1 to +1)
        daily_bias : 日線偏向 (-1 to +1)
        hourly_bias : 小時線偏向 (-1 to +1)

        Returns
        -------
        (confluence_score, reasons) : 共振分數 + 人類可讀原因
        """
        reasons: list[str] = []

        # 收集可用的時間框架偏向
        biases: list[tuple[str, float]] = []
        if abs(weekly_bias) > 0.1:
            biases.append(("Weekly", weekly_bias))
        if abs(daily_bias) > 0.1:
            biases.append(("Daily", daily_bias))
        if abs(hourly_bias) > 0.1:
            biases.append(("Hourly", hourly_bias))

        if len(biases) < 2:
            # 不足兩個框架有信號 — 無法判斷共振
            reasons.append("MTF: insufficient timeframes for confluence check")
            return 0.0, reasons

        bullish = sum(1 for _, b in biases if b > 0)
        bearish = sum(1 for _, b in biases if b < 0)

        label = ", ".join(f"{name}={'↑' if b > 0 else '↓'}" for name, b in biases)

        if bullish == len(biases):
            # 全部看多
            reasons.append(f"MTF CONFLUENCE ↑↑↑ all bullish ({label}) — high conviction")
            return 20.0, reasons
        elif bearish == len(biases):
            # 全部看空
            reasons.append(f"MTF CONFLUENCE ↓↓↓ all bearish ({label}) — high conviction")
            return -20.0, reasons
        elif bullish > 0 and bearish > 0:
            # 有多有空 — 分歧
            reasons.append(f"MTF DIVERGENCE ({label}) — reducing confidence")
            return -15.0, reasons
        else:
            # 部分中性
            reasons.append(f"MTF partial alignment ({label})")
            return 0.0, reasons

    # ══════════════════════════════════════════════════════════════
    # 特徵工程 — 計算所有技術指標
    # ══════════════════════════════════════════════════════════════

    def _bars_to_indicators(self, bars: list[OHLCVBar]) -> dict[str, Any]:
        """將 OHLCVBar 清單轉為指標字典 (任何時間框架通用)。"""
        closes = np.array([b.close for b in bars], dtype=np.float64)
        highs = np.array([b.high for b in bars], dtype=np.float64)
        lows = np.array([b.low for b in bars], dtype=np.float64)
        volumes = np.array([b.volume for b in bars], dtype=np.float64)
        return self._calculate_indicators(closes, highs, lows, volumes)

    def _calculate_indicators(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
    ) -> dict[str, Any]:
        """
        計算全部技術指標，返回指標字典。

        特殊 key:
          _reasons: list[str]  — 人類可讀的分析原因
          _bonus_score: float  — MACD/Volume 等非策略維度的額外分數
        """
        indicators: dict[str, Any] = {}
        reasons: list[str] = []
        bonus_score = 0.0
        price = closes[-1] if len(closes) > 0 else 0.0
        indicators["current_price"] = price

        # ── 順勢特徵：SMA 交叉 + 價格位置 ──────────────────
        if len(closes) >= 200:
            sma_50 = float(np.mean(closes[-50:]))
            sma_200 = float(np.mean(closes[-200:]))
            indicators["sma_50"] = sma_50
            indicators["sma_200"] = sma_200
            indicators["sma_50_above_200"] = sma_50 > sma_200
            indicators["price_above_sma200"] = price > sma_200

            if sma_50 > sma_200:
                reasons.append(
                    f"Golden cross: SMA50 ({sma_50:.2f}) > SMA200 ({sma_200:.2f})"
                )
            else:
                reasons.append(
                    f"Death cross: SMA50 ({sma_50:.2f}) < SMA200 ({sma_200:.2f})"
                )

            if price > sma_200:
                reasons.append("Price above SMA200 (long-term uptrend)")
            else:
                reasons.append("Price below SMA200 (long-term downtrend)")

        # SMA(20) 偏離度 — 用於順勢策略
        if len(closes) >= 20:
            sma_20 = float(np.mean(closes[-20:]))
            indicators["sma_20"] = sma_20
            indicators["price_to_sma20_pct"] = (
                (price - sma_20) / sma_20 if sma_20 > 0 else 0.0
            )

        # ── 均值回歸特徵：RSI(14) ──────────────────────────
        rsi = self._compute_rsi(closes, 14)
        indicators["rsi_14"] = rsi
        if rsi is not None:
            if rsi < 30:
                reasons.append(
                    f"RSI({rsi:.1f}) oversold — potential reversal up"
                )
            elif rsi > 70:
                reasons.append(
                    f"RSI({rsi:.1f}) overbought — potential reversal down"
                )
            elif 40 <= rsi <= 60:
                reasons.append(f"RSI({rsi:.1f}) neutral zone")

        # ── 波動特徵：布林通道 ──────────────────────────────
        if len(closes) >= 20:
            sma_20 = float(np.mean(closes[-20:]))
            std_20 = float(np.std(closes[-20:]))
            upper_band = sma_20 + 2 * std_20
            lower_band = sma_20 - 2 * std_20
            bb_width = (
                (upper_band - lower_band) / sma_20 if sma_20 > 0 else 0.0
            )
            bb_range = upper_band - lower_band
            bb_position = (
                (price - lower_band) / bb_range if bb_range > 0 else 0.5
            )

            indicators["bb_width"] = bb_width
            indicators["bb_position"] = bb_position
            indicators["upper_band"] = upper_band
            indicators["lower_band"] = lower_band

            if price <= lower_band:
                reasons.append(
                    "Price at lower Bollinger Band (potential bounce)"
                )
            elif price >= upper_band:
                reasons.append(
                    "Price at upper Bollinger Band (potential pullback)"
                )

            if bb_width < 0.05:
                reasons.append(
                    f"Bollinger squeeze (width={bb_width:.3f}) — breakout imminent"
                )

        # ── ATR 波動體制 ─────────────────────────────────────
        atr = self._compute_atr(highs, lows, closes, 14)
        indicators["atr"] = atr
        if atr is not None and price > 0:
            atr_pct = atr / price * 100
            indicators["atr_pct"] = atr_pct
            if atr_pct > 3.0:
                reasons.append(
                    f"High volatility regime (ATR {atr_pct:.1f}% of price)"
                )

        # ── 非策略維度補充：MACD 柱狀圖 ─────────────────────
        macd_hist = self._compute_macd_histogram(closes)
        if macd_hist is not None and len(macd_hist) >= 2:
            indicators["macd_histogram"] = float(macd_hist[-1])
            if macd_hist[-1] > 0 and macd_hist[-1] > macd_hist[-2]:
                bonus_score += 15
                reasons.append(
                    "MACD histogram rising above zero (bullish momentum)"
                )
            elif macd_hist[-1] < 0 and macd_hist[-1] < macd_hist[-2]:
                bonus_score -= 15
                reasons.append(
                    "MACD histogram falling below zero (bearish momentum)"
                )

        # ── 非策略維度補充：成交量異常 ───────────────────────
        if len(volumes) >= 20:
            avg_vol = float(np.mean(volumes[-20:]))
            if avg_vol > 0 and volumes[-1] > avg_vol * 2.0:
                vol_direction = 5 if closes[-1] > closes[-2] else -5
                bonus_score += vol_direction
                vol_ratio = volumes[-1] / avg_vol
                reasons.append(
                    f"Volume spike ({vol_ratio:.1f}x avg) — "
                    f"{'buying' if vol_direction > 0 else 'selling'} pressure"
                )

        indicators["_reasons"] = reasons
        indicators["_bonus_score"] = bonus_score
        return indicators

    # ══════════════════════════════════════════════════════════════
    # OPRO 動態權重計算
    # ══════════════════════════════════════════════════════════════

    def _compute_weighted_score(self, ind: dict[str, Any]) -> float:
        """
        利用 OPRO 動態權重進行策略維度評分。

        三大策略維度：
          1. 順勢交易 (trend_following)  — SMA 交叉 + 價格位置
          2. 均值回歸 (mean_reversion)   — RSI 超買超賣
          3. 波動率   (volatility)       — 布林通道位置

        如果近期「均值回歸」策略一直虧錢，
        OPRO 會把 self.dynamic_weights["mean_reversion"] 降到接近 0。

        Returns
        -------
        float : 加權分數，大約在 -50 到 +50 之間
        """
        # ── 順勢策略分數 (Trend-Following) ────────────────────
        tf_score = 0.0
        if ind.get("sma_50_above_200") is not None:
            tf_score += 15.0 if ind["sma_50_above_200"] else -15.0
        if ind.get("price_above_sma200") is not None:
            tf_score += 10.0 if ind["price_above_sma200"] else -10.0
        elif ind.get("price_to_sma20_pct") is not None:
            # 若不足 200 根，用 SMA(20) 偏離度作為替代
            pct = ind["price_to_sma20_pct"]
            tf_score += 25.0 if pct > 0 else -25.0

        # ── 均值回歸策略分數 (Mean-Reversion) ─────────────────
        mr_score = 0.0
        rsi = ind.get("rsi_14")
        if rsi is not None:
            if rsi < 30:
                mr_score = 25.0    # 超賣 → 看多
            elif rsi > 70:
                mr_score = -25.0   # 超買 → 看空
            else:
                # RSI 50 為中性，偏離越大分數越大
                mr_score = (50 - rsi) * 0.5  # RSI=30 → +10, RSI=70 → -10

        # ── 波動率策略分數 (Volatility) ───────────────────────
        vol_score = 0.0
        bb_pos = ind.get("bb_position")
        if bb_pos is not None:
            if bb_pos < 0.1:
                vol_score = 20.0    # 超跌反彈
            elif bb_pos > 0.9:
                vol_score = -20.0   # 超買回落
            else:
                # 通道中間位置 → 小幅信號
                vol_score = (0.5 - bb_pos) * 10.0

        # ── 加權融合 ──────────────────────────────────────────
        weighted = (
            tf_score * self.dynamic_weights["trend_following"]
            + mr_score * self.dynamic_weights["mean_reversion"]
            + vol_score * self.dynamic_weights["volatility"]
        )

        return weighted

    # ══════════════════════════════════════════════════════════════
    # OPRO 介面 — AI 自我進化入口
    # ══════════════════════════════════════════════════════════════

    def update_weights_from_opro(self, new_weights: dict[str, float]) -> None:
        """
        【機器學習介面】
        由 Adaptive-OPRO 模組呼叫，根據過去 N 天的實盤勝率，
        動態更新三大策略維度的權重。

        Parameters
        ----------
        new_weights : dict
            e.g. {"trend_following": 0.5, "mean_reversion": 0.3, "volatility": 0.2}
            只更新提供的 key，未提供的 key 保持不變。
        """
        for key in ("trend_following", "mean_reversion", "volatility"):
            if key in new_weights:
                self.dynamic_weights[key] = new_weights[key]

        logger.info(
            "從 OPRO 接收到新的機器學習權重: %s", self.dynamic_weights,
        )

    # ══════════════════════════════════════════════════════════════
    # 資料取得
    # ══════════════════════════════════════════════════════════════

    async def _fetch_daily(self, ticker: str) -> list[OHLCVBar]:
        """
        取得日線 K 線，優先用 Polygon（速度快），降級用 Alpha Vantage。
        返回標準化 OHLCVBar 清單（由舊到新）。
        """
        # ── 優先：Polygon ──────────────────────────────────────
        try:
            bars = await self._polygon.daily_bars(ticker, days=365)
            if bars:
                return bars
        except Exception:
            self.logger.warning(
                "Polygon daily fetch failed for %s, falling back to Alpha Vantage",
                ticker,
            )

        # ── 降級：Alpha Vantage ────────────────────────────────
        try:
            bars = await self._av.daily(ticker, outputsize="full")
            if bars:
                # AV 返回由新到舊，反轉為由舊到新
                return list(reversed(bars))
        except Exception:
            self.logger.exception(
                "All daily data sources failed for %s", ticker,
            )

        return []

    async def _fetch_weekly(self, ticker: str) -> list[OHLCVBar]:
        """
        取得週線 K 線 (過去 2 年，約 104 根)。
        用於判斷大趨勢方向 — 多時間框架的「望遠鏡」。
        """
        try:
            from datetime import datetime, timedelta
            from_date = (datetime.utcnow() - timedelta(days=730)).strftime("%Y-%m-%d")
            to_date = datetime.utcnow().strftime("%Y-%m-%d")
            bars = await self._polygon.aggregates(
                ticker, multiplier=1, timespan="week",
                from_date=from_date, to_date=to_date, limit=120,
            )
            if bars:
                return bars
        except Exception:
            self.logger.debug(
                "Weekly bars fetch failed for %s (non-critical)", ticker,
            )
        return []

    async def _fetch_hourly(self, ticker: str) -> list[OHLCVBar]:
        """
        取得小時線 K 線 (過去 14 天，約 98 根)。
        用於進場確認 — 多時間框架的「顯微鏡」。

        Note: Polygon 免費版僅提供 2 年內延遲數據，
              小時線足夠取得近期資料。
        """
        try:
            from datetime import datetime, timedelta
            from_date = (datetime.utcnow() - timedelta(days=14)).strftime("%Y-%m-%d")
            to_date = datetime.utcnow().strftime("%Y-%m-%d")
            bars = await self._polygon.aggregates(
                ticker, multiplier=1, timespan="hour",
                from_date=from_date, to_date=to_date, limit=200,
            )
            if bars:
                return bars
        except Exception:
            self.logger.debug(
                "Hourly bars fetch failed for %s (non-critical)", ticker,
            )
        return []

    # ══════════════════════════════════════════════════════════════
    # 指標計算 (靜態方法 — 可被外部模組直接呼叫)
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_rsi(closes: np.ndarray, period: int = 14) -> float | None:
        """
        計算 RSI (Relative Strength Index)。

        Returns
        -------
        float | None : RSI 值 (0-100)，資料不足時返回 None
        """
        if len(closes) < period + 1:
            return None
        deltas = np.diff(closes[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    @staticmethod
    def _compute_macd_histogram(
        closes: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> np.ndarray | None:
        """
        計算 MACD 柱狀圖。

        Returns
        -------
        np.ndarray | None : MACD histogram 序列，資料不足時返回 None
        """
        if len(closes) < slow + signal:
            return None

        def ema(data: np.ndarray, span: int) -> np.ndarray:
            alpha = 2.0 / (span + 1)
            out = np.empty_like(data)
            out[0] = data[0]
            for i in range(1, len(data)):
                out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
            return out

        ema_fast = ema(closes, fast)
        ema_slow = ema(closes, slow)
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signal)
        return macd_line - signal_line

    @staticmethod
    def _compute_atr(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
    ) -> float | None:
        """
        計算 ATR (Average True Range)。

        Returns
        -------
        float | None : ATR 值，資料不足時返回 None
        """
        if len(closes) < period + 1:
            return None
        tr_values = []
        for i in range(-period, 0):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            tr_values.append(tr)
        return float(np.mean(tr_values))

    @staticmethod
    def _score_to_direction(score: float) -> SignalDirection:
        """將數值分數映射到信號方向。"""
        if score >= 35:
            return SignalDirection.STRONG_BUY
        if score >= 10:
            return SignalDirection.BUY
        if score <= -35:
            return SignalDirection.STRONG_SELL
        if score <= -10:
            return SignalDirection.SELL
        return SignalDirection.NEUTRAL
