"""
Sentiment Analysis Agent — AI 情緒分析代理
============================================
抓取市場新聞、分析師評級與盈餘驚喜，透過文本情感分析
來量化市場的「貪婪與恐懼」。

三大策略維度（由 OPRO 動態調整權重）：
  1. 新聞情緒 (news_sentiment)     — 標題/摘要關鍵字 + Finnhub API 情緒分數
  2. 分析師共識 (analyst_consensus) — 升評/降評/持有比例
  3. 社群動量 (social_momentum)     — 盈餘驚喜連動 + 新聞量異常（預留擴充）

補充維度（不受權重影響）：
  - 新聞量異常偵測（放大既有方向）
  - 單次盈餘驚喜幅度

使用者：
  - IntelligenceOrchestrator：多代理信號融合
  - CognitiveLoop：績效反饋 → 權重調整

數據來源：
  - Finnhub：公司新聞 + 情緒 API + 分析師評級 + 盈餘驚喜

防禦性設計：
  - Finnhub API 掛掉或標的沒有新聞 → 安全降級回傳 NEUTRAL
  - 新聞篇數不足 3 篇 → 自動減弱分數權重，避免被單一冷門新聞帶風向
  - 使用基本詞幹還原（去除 -s/-es/-ed/-ing 字尾）提高關鍵字命中率
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from src.core.base_agent import AnalysisSignal, BaseAgent, SignalDirection
from src.data.providers.finnhub import FinnhubProvider

logger = logging.getLogger(__name__)


class SentimentAgent(BaseAgent):
    """
    情緒分析代理 (Sentiment Agent)
    負責抓取市場新聞、分析師評級與社群討論，
    透過文本情感分析來量化市場的「貪婪與恐懼」。

    指標分三大策略維度：
      1. 新聞情緒 (news_sentiment)     — 關鍵字極性 + API 情緒分數
      2. 分析師共識 (analyst_consensus) — Buy/Sell/Hold 比例
      3. 社群動量 (social_momentum)     — 盈餘驚喜動能（預留擴充）
    """

    # ── 預設 OPRO 動態權重 ──────────────────────────────────────
    _DEFAULT_STRATEGY_WEIGHTS = {
        "news_sentiment": 0.50,       # 新聞標題與摘要情緒
        "analyst_consensus": 0.30,    # 分析師共識 (升評/降評)
        "social_momentum": 0.20,      # 盈餘驚喜動能 / 社群擴充
    }

    # ── 情緒關鍵字字典 ──────────────────────────────────────────
    _POSITIVE_KEYWORDS = frozenset({
        "beat", "surge", "rally", "upgrade", "profit", "growth", "record",
        "outperform", "bullish", "gain", "soar", "strong", "exceed",
        "optimistic", "breakthrough", "positive", "buy", "dividend", "jump",
    })
    _NEGATIVE_KEYWORDS = frozenset({
        "miss", "crash", "plunge", "downgrade", "loss", "decline", "cut",
        "underperform", "bearish", "drop", "fall", "weak", "warning",
        "pessimistic", "lawsuit", "negative", "recall", "fraud",
        "sell", "investigation", "bankrupt",
    })

    def __init__(self, finnhub: FinnhubProvider) -> None:
        super().__init__("sentiment")
        self._finnhub = finnhub
        # OPRO 動態權重 — 初始為平衡值，後續由機器學習迴圈調整
        self.dynamic_weights = dict(self._DEFAULT_STRATEGY_WEIGHTS)

    # ══════════════════════════════════════════════════════════════
    # 主分析流程
    # ══════════════════════════════════════════════════════════════

    async def analyse(self, context: dict[str, Any]) -> AnalysisSignal:
        """
        執行完整的情緒分析與 AI 推理流程。

        Parameters
        ----------
        context : dict
            Must contain "ticker" (str). Optional: "isin", "current_price".

        Returns
        -------
        AnalysisSignal : 標準化信號 (direction, confidence, reasoning, data)
        """
        ticker = context["ticker"]

        # ── Step 1: 並行抓取所有情緒數據 ───────────────────────
        news, sentiment_raw, recommendations, earnings = (
            await self._fetch_data(ticker)
        )

        # ── Step 2: 特徵提取 — 文本情緒量化 ───────────────────
        metrics = self._extract_features(
            news, sentiment_raw, recommendations, earnings,
        )

        # ── Step 3: OPRO 動態權重融合 ──────────────────────────
        weighted_score = self._compute_weighted_score(metrics)

        # ── Step 4: 補充維度 (新聞量異常) ──────────────────────
        bonus_score = metrics.pop("_bonus_score", 0.0)
        reasons: list[str] = metrics.pop("_reasons", [])

        # ── Step 5: 最終合成 ───────────────────────────────────
        composite_score = weighted_score + bonus_score
        composite_score = max(-100.0, min(100.0, composite_score))

        if not reasons:
            reasons.append("Insufficient sentiment data")

        direction = self._score_to_direction(composite_score)
        confidence = min(abs(composite_score) / 30.0, 1.0)
        # Floor: if we got data and computed a score, confidence >= 0.20
        if abs(composite_score) > 1.0:
            confidence = max(confidence, 0.20)

        logger.info(
            "[SentimentAgent] %s 分析完成: 訊號=%s, 信心度=%.2f, "
            "加權分=%+.1f (新聞/分析師/社群 動態融合)",
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
                "api_sentiment": metrics.get("api_sentiment"),
                "news_score": metrics.get("news_score"),
                "positive_hits": metrics.get("positive_hits"),
                "negative_hits": metrics.get("negative_hits"),
                "news_count": metrics.get("total_articles"),
                "analyst_score": metrics.get("analyst_score"),
                "earnings_surprise_score": metrics.get(
                    "earnings_surprise_score",
                ),
            },
        )

    # ══════════════════════════════════════════════════════════════
    # 資料取得
    # ══════════════════════════════════════════════════════════════

    async def _fetch_data(
        self, ticker: str,
    ) -> tuple[list, dict, list, list]:
        """並行抓取四類情緒數據，任一失敗安全降級。"""
        news_task = asyncio.create_task(
            self._safe_fetch(self._finnhub.company_news(ticker), []),
        )
        sentiment_task = asyncio.create_task(
            self._safe_fetch(self._finnhub.news_sentiment(ticker), {}),
        )
        reco_task = asyncio.create_task(
            self._safe_fetch(
                self._finnhub.recommendation_trends(ticker), [],
            ),
        )
        earnings_task = asyncio.create_task(
            self._safe_fetch(
                self._finnhub.earnings_surprises(ticker), [],
            ),
        )

        news = await news_task
        sentiment_raw = await sentiment_task
        recommendations = await reco_task
        earnings = await earnings_task

        return news, sentiment_raw, recommendations, earnings

    async def _safe_fetch(self, coro: Any, default: Any) -> Any:
        """安全包裝：失敗時返回 default 而非拋出例外"""
        try:
            return await coro
        except Exception:
            self.logger.warning("情緒數據取得失敗", exc_info=True)
            return default

    # ══════════════════════════════════════════════════════════════
    # 特徵提取 — 將所有原始數據轉為量化指標
    # ══════════════════════════════════════════════════════════════

    def _extract_features(
        self,
        news: list[dict[str, Any]],
        sentiment_raw: dict[str, Any],
        recommendations: list[dict[str, Any]],
        earnings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        將四類原始數據轉為量化指標字典。

        特殊 key:
          _reasons: list[str]  — 人類可讀的分析原因
          _bonus_score: float  — 新聞量異常等補充分數
        """
        metrics: dict[str, Any] = {}
        reasons: list[str] = []
        bonus = 0.0

        # ── 1. Finnhub API 情緒分數 ──────────────────────────
        api_sentiment = self._extract_sentiment_score(sentiment_raw)
        metrics["api_sentiment"] = api_sentiment

        if api_sentiment is not None:
            # Finnhub returns 0-1 (0.5 = neutral) → 轉為 -1 to +1
            centered = (api_sentiment - 0.5) * 2
            metrics["api_sentiment_centered"] = centered
            label = (
                "bullish" if centered > 0.1
                else "bearish" if centered < -0.1
                else "neutral"
            )
            reasons.append(
                f"Finnhub sentiment: {api_sentiment:.2f} ({label})"
            )

        # ── 2. 關鍵字情緒分析 ────────────────────────────────
        article_count = len(news) if isinstance(news, list) else 0
        metrics["total_articles"] = float(article_count)

        if isinstance(news, list) and news:
            pos, neg = self._keyword_sentiment(news)
            metrics["positive_hits"] = float(pos)
            metrics["negative_hits"] = float(neg)

            total_hits = pos + neg
            if total_hits > 0:
                news_score = (pos - neg) / total_hits  # -1.0 to +1.0
            else:
                news_score = 0.0

            # 若新聞篇數不足 3 篇，降低權重（避免單一冷門新聞帶風向）
            if article_count < 3:
                news_score *= 0.5

            metrics["news_score"] = news_score
            if total_hits > 0:
                reasons.append(
                    f"Headline keywords: {pos} positive, {neg} negative "
                    f"(ratio={news_score:+.2f})"
                )

            # 新聞量異常 → 作為補充 bonus
            if article_count > 30:
                vol_bonus = 5 if news_score > 0 else -5
                bonus += vol_bonus
                reasons.append(
                    f"High news volume ({article_count} articles) — "
                    f"heightened attention"
                )
            elif article_count == 0:
                reasons.append("No recent news — low attention")
        else:
            metrics["positive_hits"] = 0.0
            metrics["negative_hits"] = 0.0
            metrics["news_score"] = 0.0

        # ── 3. 分析師共識 ────────────────────────────────────
        analyst_score = 0.0
        if recommendations and isinstance(recommendations, list):
            latest = recommendations[0]
            buy = latest.get("buy", 0) + latest.get("strongBuy", 0)
            sell = latest.get("sell", 0) + latest.get("strongSell", 0)
            hold = latest.get("hold", 0)
            total = buy + sell + hold
            if total > 0:
                consensus = (buy - sell) / total  # -1.0 to +1.0
                analyst_score = consensus
                reasons.append(
                    f"Analyst consensus: {buy} buy, {hold} hold, "
                    f"{sell} sell (score={consensus:+.2f})"
                )
        metrics["analyst_score"] = analyst_score

        # ── 4. 盈餘驚喜動能 ─────────────────────────────────
        surprise_score = 0.0
        if earnings and isinstance(earnings, list) and len(earnings) >= 2:
            surprises = [
                e.get("surprisePercent", 0)
                for e in earnings[:4]
                if e.get("surprisePercent") is not None
            ]
            if surprises:
                avg_surprise = sum(surprises) / len(surprises)
                # 正規化到 -1.0 to +1.0 範圍
                surprise_score = max(-1.0, min(1.0, avg_surprise / 15.0))
                reasons.append(
                    f"Avg earnings surprise: {avg_surprise:+.1f}%"
                )
        metrics["earnings_surprise_score"] = surprise_score

        metrics["_reasons"] = reasons
        metrics["_bonus_score"] = bonus
        return metrics

    # ══════════════════════════════════════════════════════════════
    # OPRO 動態權重計算
    # ══════════════════════════════════════════════════════════════

    def _compute_weighted_score(self, metrics: dict[str, Any]) -> float:
        """
        利用 OPRO 動態權重進行策略維度評分。

        三大策略維度：
          1. 新聞情緒 (news_sentiment)     — 關鍵字 + API 情緒
          2. 分析師共識 (analyst_consensus) — Buy/Sell/Hold
          3. 社群動量 (social_momentum)     — 盈餘驚喜動能

        若市場處於「看基本面/技術面不如看消息面」的極端狂熱期
        （如迷因股熱潮），OPRO 可透過權重介面大幅提升新聞/社群權重。

        Returns
        -------
        float : 加權分數，大約在 -50 到 +50 之間
        """
        # ── 新聞情緒維度分數 ─────────────────────────────────
        # 結合 API 情緒 + 關鍵字情緒
        news_dim = 0.0
        api_centered = metrics.get("api_sentiment_centered")
        news_keyword = metrics.get("news_score", 0.0)

        if api_centered is not None:
            # API 情緒權重較大（30分滿分），關鍵字補充（15分滿分）
            news_dim = api_centered * 30 + news_keyword * 15
        else:
            # 無 API 情緒時，關鍵字佔全部
            news_dim = news_keyword * 30

        # 正規化到 ±25
        news_dim = max(-25.0, min(25.0, news_dim))

        # ── 分析師共識維度分數 ───────────────────────────────
        analyst_dim = metrics.get("analyst_score", 0.0) * 25.0
        analyst_dim = max(-25.0, min(25.0, analyst_dim))

        # ── 社群動量維度分數（盈餘驚喜作為代理） ─────────────
        social_dim = metrics.get("earnings_surprise_score", 0.0) * 25.0
        social_dim = max(-25.0, min(25.0, social_dim))

        # ── 加權融合 ──────────────────────────────────────────
        weighted = (
            news_dim * self.dynamic_weights["news_sentiment"]
            + analyst_dim * self.dynamic_weights["analyst_consensus"]
            + social_dim * self.dynamic_weights["social_momentum"]
        )

        return weighted

    # ══════════════════════════════════════════════════════════════
    # OPRO 介面 — AI 自我進化入口
    # ══════════════════════════════════════════════════════════════

    def update_weights_from_opro(self, new_weights: dict[str, float]) -> None:
        """
        【機器學習介面】
        由 Adaptive-OPRO 模組呼叫。

        若市場處於「看消息面不如看基本面」的極端恐慌期，
        OPRO 可透過此介面降低 news_sentiment 權重。
        反之，迷因股熱潮期間，OPRO 會提升 social_momentum 權重。

        Parameters
        ----------
        new_weights : dict
            e.g. {"news_sentiment": 0.6, "analyst_consensus": 0.2, "social_momentum": 0.2}
            只更新提供的 key，未提供的 key 保持不變。
        """
        for key in ("news_sentiment", "analyst_consensus", "social_momentum"):
            if key in new_weights:
                self.dynamic_weights[key] = new_weights[key]

        logger.info(
            "[SentimentAgent] 接收到 OPRO 的機器學習權重更新: %s",
            self.dynamic_weights,
        )

    # ══════════════════════════════════════════════════════════════
    # 輔助方法
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _extract_sentiment_score(raw: dict[str, Any]) -> float | None:
        """從 Finnhub news-sentiment 回應中提取情緒分數"""
        if not raw:
            return None
        # Finnhub news-sentiment endpoint
        sentiment = raw.get("sentiment")
        if isinstance(sentiment, dict):
            return sentiment.get("bullishPercent")
        # Fallback: companyNewsHQSentiment
        cs = raw.get("companyNewsHQSentiment")
        if isinstance(cs, dict):
            return cs.get("score")
        return None

    @staticmethod
    def _keyword_sentiment(
        articles: list[dict[str, Any]],
    ) -> tuple[int, int]:
        """
        正負極性量化 — 將新聞標題與摘要轉為正/負命中次數。

        使用基本詞幹還原（去除 -s/-es/-ed/-ing 字尾）
        提高關鍵字命中率（e.g. "beats" → "beat", "surges" → "surge"）。

        Returns
        -------
        (positive_hits, negative_hits)
        """
        pos = neg = 0
        for article in articles:
            headline = (article.get("headline") or "").lower()
            summary = (article.get("summary") or "").lower()
            text = f"{headline} {summary}"
            for word in text.split():
                clean = word.strip(".,!?;:'\"()[]")
                # 基本詞幹還原
                stem = clean
                if stem.endswith("ing") and len(stem) > 4:
                    stem = stem[:-3]
                elif stem.endswith("ed") and len(stem) > 3:
                    stem = stem[:-2]
                elif stem.endswith("es") and len(stem) > 3:
                    stem = stem[:-2]
                elif stem.endswith("s") and len(stem) > 3:
                    stem = stem[:-1]
                if clean in SentimentAgent._POSITIVE_KEYWORDS or stem in SentimentAgent._POSITIVE_KEYWORDS:
                    pos += 1
                elif clean in SentimentAgent._NEGATIVE_KEYWORDS or stem in SentimentAgent._NEGATIVE_KEYWORDS:
                    neg += 1
        return pos, neg

    @staticmethod
    def _score_to_direction(score: float) -> SignalDirection:
        """將數值分數映射到信號方向。"""
        if score >= 20:
            return SignalDirection.STRONG_BUY
        if score >= 5:
            return SignalDirection.BUY
        if score <= -20:
            return SignalDirection.STRONG_SELL
        if score <= -5:
            return SignalDirection.SELL
        return SignalDirection.NEUTRAL
