"""
Episodic Memory Store
=====================
Vector-based memory bank that stores trade episodes as dense embeddings.
Each episode captures the full context of a trade decision and its
outcome, enabling:
  - Similarity search: "Have we seen this pattern before?"
  - Regime detection: clustering episodes by market conditions
  - Contrastive learning: comparing winning vs losing episodes
  - Counterfactual retrieval: finding similar setups with different outcomes

Storage: numpy arrays on disk (.npz) for zero-dependency persistence.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Embedding dimension for episode vectors
_EMBED_DIM = 32


@dataclass
class Episode:
    """A single trade episode stored in episodic memory."""
    episode_id: str = ""
    timestamp: float = field(default_factory=time.time)

    # Audit reference
    audit_record_id: str = ""
    ticker: str = ""
    isin: str = ""

    # Context vector (normalised feature representation)
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(_EMBED_DIM))

    # Raw features (for interpretability)
    features: dict[str, float] = field(default_factory=dict)

    # Outcome
    action: str = ""
    roi: float = 0.0
    pnl: float = 0.0
    failure_layer: str = ""       # "semantic", "temporal", "execution"
    failure_detail: str = ""      # 錯誤歸因引擎寫入的診斷摘要

    # Intelligence snapshot
    fused_score: float = 0.0
    fused_confidence: float = 0.0
    agent_scores: dict[str, float] = field(default_factory=dict)

    # Market regime tags
    regime_tag: str = ""          # "trending", "mean_reverting", "volatile", "quiet"
    volatility_regime: str = ""   # "low", "normal", "high", "extreme"


class EpisodicMemory:
    """
    Persistent vector store for trade episodes.
    Supports cosine-similarity retrieval and regime-based filtering.
    """

    def __init__(self, store_dir: str = "logs/memory") -> None:
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._episodes: list[Episode] = []
        self._embeddings: np.ndarray | None = None   # (N, _EMBED_DIM) matrix
        self._counter = 0
        self._load()

    # ── episode creation ──────────────────────────────────────────────

    def store(self, episode: Episode) -> str:
        """Store a new episode and rebuild the embedding index."""
        self._counter += 1
        episode.episode_id = f"EP-{self._counter:06d}"
        self._episodes.append(episode)
        self._rebuild_index()
        self._persist()
        logger.info(
            "Stored episode %s for %s (ROI=%.4f, regime=%s)",
            episode.episode_id, episode.ticker, episode.roi, episode.regime_tag,
        )
        return episode.episode_id

    def store_from_audit(
        self,
        audit_record: dict[str, Any],
        market_features: dict[str, float],
    ) -> str:
        """
        Create and store an episode from an AuditRecord dict and
        raw market features.
        """
        features = self._normalise_features(market_features)
        embedding = self._features_to_embedding(features)

        episode = Episode(
            audit_record_id=audit_record.get("record_id", ""),
            ticker=audit_record.get("ticker", ""),
            isin=audit_record.get("isin", ""),
            embedding=embedding,
            features=features,
            action=audit_record.get("action", ""),
            roi=float(audit_record.get("realised_roi", 0) or 0),
            pnl=float(audit_record.get("realised_pnl", 0) or 0),
            failure_layer=audit_record.get("failure_layer", ""),
            fused_score=float(audit_record.get("fused_score", 0)),
            fused_confidence=float(audit_record.get("fused_confidence", 0)),
            regime_tag=self._detect_regime(features),
            volatility_regime=self._detect_volatility_regime(features),
        )

        return self.store(episode)

    # ── retrieval ─────────────────────────────────────────────────────

    def query_similar(
        self,
        embedding: np.ndarray,
        k: int = 10,
        min_similarity: float = 0.5,
        regime_filter: str | None = None,
    ) -> list[tuple[Episode, float]]:
        """
        Find the k most similar episodes by cosine similarity.

        Returns list of (Episode, similarity_score) sorted descending.
        """
        if self._embeddings is None or len(self._episodes) == 0:
            return []

        # Cosine similarity: dot(a, b) / (|a| * |b|)
        query_norm = np.linalg.norm(embedding)
        if query_norm == 0:
            return []

        norms = np.linalg.norm(self._embeddings, axis=1)
        valid = norms > 0
        similarities = np.zeros(len(self._episodes))
        similarities[valid] = (
            self._embeddings[valid] @ embedding
        ) / (norms[valid] * query_norm)

        # Apply filters
        candidates: list[tuple[int, float]] = []
        for i, sim in enumerate(similarities):
            if sim < min_similarity:
                continue
            if regime_filter and self._episodes[i].regime_tag != regime_filter:
                continue
            candidates.append((i, float(sim)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [
            (self._episodes[i], sim) for i, sim in candidates[:k]
        ]

    def query_by_features(
        self,
        features: dict[str, float],
        k: int = 10,
        **kwargs: Any,
    ) -> list[tuple[Episode, float]]:
        """Convenience: convert features to embedding and query."""
        normed = self._normalise_features(features)
        emb = self._features_to_embedding(normed)
        return self.query_similar(emb, k=k, **kwargs)

    def get_winning_episodes(self, min_roi: float = 0.01) -> list[Episode]:
        return [e for e in self._episodes if e.roi >= min_roi]

    def get_losing_episodes(self, max_roi: float = -0.01) -> list[Episode]:
        return [e for e in self._episodes if e.roi <= max_roi]

    def get_by_regime(self, regime: str) -> list[Episode]:
        return [e for e in self._episodes if e.regime_tag == regime]

    # ── contrastive pairs ─────────────────────────────────────────────

    def find_contrastive_pairs(
        self,
        embedding: np.ndarray,
        k: int = 5,
    ) -> list[tuple[Episode, Episode, float]]:
        """
        For a given context, find pairs of similar episodes where one
        won and one lost — ideal for decoupled contrastive learning.

        Returns list of (winning_episode, losing_episode, similarity).
        """
        similar = self.query_similar(embedding, k=k * 4, min_similarity=0.3)
        winners = [(ep, sim) for ep, sim in similar if ep.roi > 0.005]
        losers = [(ep, sim) for ep, sim in similar if ep.roi < -0.005]

        pairs = []
        for w_ep, w_sim in winners[:k]:
            for l_ep, l_sim in losers[:k]:
                avg_sim = (w_sim + l_sim) / 2
                pairs.append((w_ep, l_ep, avg_sim))

        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:k]

    # ── statistics ────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        return len(self._episodes)

    def regime_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for ep in self._episodes:
            dist[ep.regime_tag] = dist.get(ep.regime_tag, 0) + 1
        return dist

    def avg_roi_by_regime(self) -> dict[str, float]:
        regime_rois: dict[str, list[float]] = {}
        for ep in self._episodes:
            regime_rois.setdefault(ep.regime_tag, []).append(ep.roi)
        return {
            regime: sum(rois) / len(rois)
            for regime, rois in regime_rois.items()
            if rois
        }

    # ── RiskAgent 動態參數介面 ─────────────────────────────────────

    # 當記憶不足時回傳的保守預設值
    _DEFAULT_WIN_RATE = 0.55
    _DEFAULT_WIN_LOSS_RATIO = 1.5

    def get_recent_performance(
        self,
        limit: int = 30,
        action_filter: str | None = None,
    ) -> dict[str, float]:
        """
        計算近期交易的勝率與盈虧比，供 RiskAgent 的半凱利公式使用。

        取代寫死的 ``historical_win_rate=0.55`` 和
        ``historical_win_loss_ratio=1.5``。

        Parameters
        ----------
        limit : 取最近 N 筆記憶
        action_filter : 只統計 "BUY" 或 "SELL" 方向 (可選)

        Returns
        -------
        dict with keys: win_rate, win_loss_ratio, total_trades, avg_roi
        """
        episodes = self._episodes
        if action_filter:
            episodes = [e for e in episodes if e.action == action_filter]

        if not episodes:
            return {
                "win_rate": self._DEFAULT_WIN_RATE,
                "win_loss_ratio": self._DEFAULT_WIN_LOSS_RATIO,
                "total_trades": 0,
                "avg_roi": 0.0,
            }

        recent = episodes[-limit:]

        wins = [e for e in recent if e.roi > 0]
        losses = [e for e in recent if e.roi <= 0]

        win_rate = len(wins) / len(recent) if recent else self._DEFAULT_WIN_RATE

        avg_win = float(np.mean([e.roi for e in wins])) if wins else 0.0
        avg_loss = float(np.mean([abs(e.roi) for e in losses])) if losses else 0.0

        if avg_loss > 0:
            win_loss_ratio = avg_win / avg_loss
        elif wins:
            # 有盈利但無虧損 → 非常好的表現
            win_loss_ratio = 3.0
        else:
            win_loss_ratio = self._DEFAULT_WIN_LOSS_RATIO

        avg_roi = float(np.mean([e.roi for e in recent]))

        return {
            "win_rate": win_rate,
            "win_loss_ratio": win_loss_ratio,
            "total_trades": len(recent),
            "avg_roi": avg_roi,
        }

    def get_regime_performance(
        self,
        regime: str,
        limit: int = 30,
    ) -> dict[str, float]:
        """
        按市場體制 (regime) 篩選的績效統計。

        當 RiskAgent 知道目前處於 "trending" 或 "volatile" 環境時，
        可以用此方法取得該體制下的歷史表現，進一步微調 Kelly 分數。
        """
        regime_eps = [e for e in self._episodes if e.regime_tag == regime]
        if not regime_eps:
            return self.get_recent_performance(limit=limit)

        recent = regime_eps[-limit:]
        wins = [e for e in recent if e.roi > 0]
        losses = [e for e in recent if e.roi <= 0]

        win_rate = len(wins) / len(recent) if recent else self._DEFAULT_WIN_RATE
        avg_win = float(np.mean([e.roi for e in wins])) if wins else 0.0
        avg_loss = float(np.mean([abs(e.roi) for e in losses])) if losses else 0.0

        win_loss_ratio = (
            avg_win / avg_loss if avg_loss > 0
            else 3.0 if wins
            else self._DEFAULT_WIN_LOSS_RATIO
        )

        return {
            "win_rate": win_rate,
            "win_loss_ratio": win_loss_ratio,
            "total_trades": len(recent),
            "avg_roi": float(np.mean([e.roi for e in recent])),
            "regime": regime,
        }

    # ── failure attribution writeback ──────────────────────────────────

    def update_episode_failure(
        self,
        episode_id: str,
        failure_layer: str,
        failure_detail: str,
    ) -> bool:
        """
        由錯誤歸因引擎回寫診斷結果到指定 Episode。

        Parameters
        ----------
        episode_id : 要更新的 Episode ID（例如 "EP-000001"）
        failure_layer : 主要歸因維度 ("semantic" / "temporal" / "execution")
        failure_detail : 診斷摘要文字

        Returns
        -------
        True if episode was found and updated, False otherwise.
        """
        for ep in self._episodes:
            if ep.episode_id == episode_id:
                ep.failure_layer = failure_layer
                ep.failure_detail = failure_detail
                self._persist()
                logger.info(
                    "Updated episode %s failure: layer=%s, detail=%s",
                    episode_id, failure_layer, failure_detail[:80],
                )
                return True

        logger.warning("Episode %s not found for failure update", episode_id)
        return False

    def get_episode(self, episode_id: str) -> Episode | None:
        """Retrieve a single episode by ID."""
        for ep in self._episodes:
            if ep.episode_id == episode_id:
                return ep
        return None

    # ── feature engineering ───────────────────────────────────────────

    @staticmethod
    def _normalise_features(raw: dict[str, float]) -> dict[str, float]:
        """Min-max style normalisation with safe defaults."""
        normed: dict[str, float] = {}
        _RANGES = {
            "rsi": (0, 100),
            "macd_hist": (-5, 5),
            "bb_width": (0, 0.5),
            "atr_pct": (0, 10),
            "volume_ratio": (0, 5),
            "sentiment_score": (-1, 1),
            "pe_ratio": (0, 100),
            "roe": (-0.5, 1.0),
            "fused_score": (-2, 2),
            "fused_confidence": (0, 1),
            "sma_50_200_ratio": (0.5, 1.5),
            "price_vs_sma200": (-0.5, 0.5),
            "earnings_surprise": (-50, 50),
            "news_sentiment": (-1, 1),
            "analyst_consensus": (-1, 1),
            "var_95_pct": (0, 10),
            "kelly_fraction": (0, 1),
        }
        for key, val in raw.items():
            lo, hi = _RANGES.get(key, (0, 1))
            span = hi - lo
            if span > 0:
                normed[key] = max(0.0, min(1.0, (val - lo) / span))
            else:
                normed[key] = 0.5
        return normed

    @staticmethod
    def _features_to_embedding(features: dict[str, float]) -> np.ndarray:
        """
        Convert a feature dict into a fixed-size dense vector.
        Features are placed in canonical order; missing features get 0.
        """
        _CANONICAL_KEYS = [
            "rsi", "macd_hist", "bb_width", "atr_pct", "volume_ratio",
            "sentiment_score", "pe_ratio", "roe", "fused_score",
            "fused_confidence", "sma_50_200_ratio", "price_vs_sma200",
            "earnings_surprise", "news_sentiment", "analyst_consensus",
            "var_95_pct", "kelly_fraction",
            # Padding to _EMBED_DIM
            "_pad_17", "_pad_18", "_pad_19", "_pad_20", "_pad_21",
            "_pad_22", "_pad_23", "_pad_24", "_pad_25", "_pad_26",
            "_pad_27", "_pad_28", "_pad_29", "_pad_30", "_pad_31",
        ]
        vec = np.zeros(_EMBED_DIM, dtype=np.float64)
        for i, key in enumerate(_CANONICAL_KEYS[:_EMBED_DIM]):
            vec[i] = features.get(key, 0.0)
        return vec

    @staticmethod
    def _detect_regime(features: dict[str, float]) -> str:
        """Classify current market regime from normalised features."""
        rsi = features.get("rsi", 0.5)
        bb_width = features.get("bb_width", 0.5)
        sma_ratio = features.get("sma_50_200_ratio", 0.5)

        # Trending: SMA50 clearly above/below SMA200, moderate RSI
        if sma_ratio > 0.65 or sma_ratio < 0.35:
            return "trending"

        # Volatile: wide Bollinger bands
        if bb_width > 0.7:
            return "volatile"

        # Mean-reverting: narrow bands, RSI near extremes
        if bb_width < 0.3 and (rsi < 0.3 or rsi > 0.7):
            return "mean_reverting"

        return "quiet"

    @staticmethod
    def _detect_volatility_regime(features: dict[str, float]) -> str:
        atr = features.get("atr_pct", 0.5)
        if atr > 0.8:
            return "extreme"
        if atr > 0.6:
            return "high"
        if atr > 0.3:
            return "normal"
        return "low"

    # ── persistence ───────────────────────────────────────────────────

    def _rebuild_index(self) -> None:
        if not self._episodes:
            self._embeddings = None
            return
        self._embeddings = np.vstack([ep.embedding for ep in self._episodes])

    def _persist(self) -> None:
        # Save embeddings as .npz
        emb_path = self._store_dir / "embeddings.npz"
        if self._embeddings is not None:
            np.savez_compressed(str(emb_path), embeddings=self._embeddings)

        # Save metadata as JSONL
        meta_path = self._store_dir / "episodes.jsonl"
        with open(meta_path, "w", encoding="utf-8") as f:
            for ep in self._episodes:
                record = {
                    "episode_id": ep.episode_id,
                    "timestamp": ep.timestamp,
                    "audit_record_id": ep.audit_record_id,
                    "ticker": ep.ticker,
                    "isin": ep.isin,
                    "features": ep.features,
                    "action": ep.action,
                    "roi": ep.roi,
                    "pnl": ep.pnl,
                    "failure_layer": ep.failure_layer,
                    "fused_score": ep.fused_score,
                    "fused_confidence": ep.fused_confidence,
                    "agent_scores": ep.agent_scores,
                    "regime_tag": ep.regime_tag,
                    "volatility_regime": ep.volatility_regime,
                    "failure_detail": ep.failure_detail,
                }
                f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")

    def _load(self) -> None:
        meta_path = self._store_dir / "episodes.jsonl"
        emb_path = self._store_dir / "embeddings.npz"

        if not meta_path.exists():
            return

        try:
            episodes = []
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    ep = Episode(
                        episode_id=data["episode_id"],
                        timestamp=data.get("timestamp", 0),
                        audit_record_id=data.get("audit_record_id", ""),
                        ticker=data.get("ticker", ""),
                        isin=data.get("isin", ""),
                        features=data.get("features", {}),
                        action=data.get("action", ""),
                        roi=data.get("roi", 0),
                        pnl=data.get("pnl", 0),
                        failure_layer=data.get("failure_layer", ""),
                        fused_score=data.get("fused_score", 0),
                        fused_confidence=data.get("fused_confidence", 0),
                        agent_scores=data.get("agent_scores", {}),
                        regime_tag=data.get("regime_tag", ""),
                        volatility_regime=data.get("volatility_regime", ""),
                    )
                    ep.failure_detail = data.get("failure_detail", "")
                    episodes.append(ep)

            self._episodes = episodes
            self._counter = len(episodes)

            if emb_path.exists():
                loaded = np.load(str(emb_path))
                self._embeddings = loaded["embeddings"]
            else:
                self._rebuild_index()

            logger.info("Loaded %d episodes from memory store", self.count)

        except Exception:
            logger.exception("Failed to load episodic memory — starting fresh")
            self._episodes = []
            self._embeddings = None
