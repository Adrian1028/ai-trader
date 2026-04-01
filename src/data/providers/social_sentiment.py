"""
Social Sentiment Provider — 社群情緒數據源
==========================================
從 Reddit API 取得投資社群討論數據，量化散戶情緒。

追蹤社群：
  - r/wallstreetbets — 散戶動量指標
  - r/stocks — 穩健投資者情緒
  - r/investing — 長期投資者觀點

使用者：
  - SocialSentimentAgent：社群情緒信號分析
"""
from __future__ import annotations

import logging
import time
from typing import Any

import aiohttp

from src.data.providers.base_provider import BaseDataProvider

logger = logging.getLogger(__name__)

_DEFAULT_SUBREDDITS = ["wallstreetbets", "stocks", "investing"]

# Basic sentiment keywords
_BULLISH_WORDS = frozenset({
    "buy", "calls", "moon", "rocket", "bull", "long", "squeeze",
    "diamond", "hold", "yolo", "tendies", "green", "pump", "breakout",
    "undervalued", "dip", "loading",
})
_BEARISH_WORDS = frozenset({
    "sell", "puts", "crash", "bear", "short", "dump", "red",
    "overvalued", "bubble", "correction", "bag", "loss", "tank",
    "drill", "rug", "scam",
})


class SocialSentimentProvider(BaseDataProvider):
    """
    Reddit 社群情緒數據提供者。

    使用 Reddit API (OAuth2) 取得投資社群的貼文與討論。

    Parameters
    ----------
    client_id : Reddit App Client ID
    client_secret : Reddit App Client Secret
    user_agent : Reddit API User-Agent
    cache_ttl : 快取 TTL（社群數據更新頻繁，預設 5 分鐘）
    """

    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        user_agent: str = "TradingBot/1.0",
        cache_ttl: float = 300.0,
    ) -> None:
        super().__init__(
            name="social_sentiment",
            api_key="",
            base_url="https://oauth.reddit.com",
            cache_ttl_seconds=cache_ttl,
        )
        self._client_id = client_id
        self._client_secret = client_secret
        self._reddit_user_agent = user_agent
        self._access_token: str | None = None
        self._token_expires: float = 0.0

    async def _authenticate(self) -> str | None:
        """取得 Reddit OAuth2 access token。"""
        if not self._client_id or not self._client_secret:
            return None

        # Check if current token is still valid
        if self._access_token and time.time() < self._token_expires:
            return self._access_token

        try:
            auth = aiohttp.BasicAuth(self._client_id, self._client_secret)
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://www.reddit.com/api/v1/access_token",
                    auth=auth,
                    data={
                        "grant_type": "client_credentials",
                    },
                    headers={"User-Agent": self._reddit_user_agent},
                ) as resp:
                    if resp.status != 200:
                        self.logger.warning("[Reddit] Auth failed: %d", resp.status)
                        return None
                    data = await resp.json()
                    self._access_token = data.get("access_token")
                    expires_in = data.get("expires_in", 3600)
                    self._token_expires = time.time() + expires_in - 60
                    return self._access_token
        except Exception:
            self.logger.warning("[Reddit] Auth error", exc_info=True)
            return None

    async def _reddit_get(
        self, endpoint: str, params: dict[str, Any] | None = None,
    ) -> Any:
        """帶 OAuth2 認證的 Reddit API 請求。"""
        token = await self._authenticate()
        if not token:
            return {}

        session = await self._ensure_session()
        url = f"{self._base_url}/{endpoint}"

        headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": self._reddit_user_agent,
        }

        try:
            async with session.get(url, params=params, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json(content_type=None)
                self.logger.warning("[Reddit] API %d for %s", resp.status, endpoint)
                return {}
        except Exception:
            self.logger.warning("[Reddit] Request failed for %s", endpoint, exc_info=True)
            return {}

    async def reddit_mentions(
        self,
        ticker: str,
        subreddits: list[str] | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """
        搜尋 Reddit 中提及指定股票的貼文。

        Returns list of:
        [{
            "title": str,
            "subreddit": str,
            "score": int (upvotes),
            "num_comments": int,
            "created_utc": float,
            "sentiment": "bullish"|"bearish"|"neutral"
        }]
        """
        subs = subreddits or _DEFAULT_SUBREDDITS

        all_posts: list[dict[str, Any]] = []
        clean_ticker = ticker.replace(".L", "").upper()

        for sub in subs:
            data = await self._reddit_get(
                f"r/{sub}/search",
                {
                    "q": f"${clean_ticker} OR {clean_ticker}",
                    "sort": "new",
                    "restrict_sr": "on",
                    "limit": str(limit),
                    "t": "week",
                },
            )

            children = (
                data.get("data", {}).get("children", [])
                if isinstance(data, dict) else []
            )

            for child in children:
                post = child.get("data", {})
                title = post.get("title", "")
                body = post.get("selftext", "")

                sentiment = self._analyse_sentiment(f"{title} {body}")

                all_posts.append({
                    "title": title,
                    "subreddit": sub,
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "created_utc": post.get("created_utc", 0),
                    "sentiment": sentiment,
                })

        # Sort by score (engagement)
        all_posts.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_posts[:limit]

    async def reddit_sentiment_score(
        self, ticker: str,
    ) -> dict[str, Any]:
        """
        計算 Reddit 整體情緒分數。

        Returns {
            "score": float (-1.0 to +1.0),
            "total_mentions": int,
            "bullish_count": int,
            "bearish_count": int,
            "neutral_count": int,
            "avg_engagement": float,
            "mention_velocity": float (mentions per subreddit)
        }
        """
        posts = await self.reddit_mentions(ticker)

        if not posts:
            return {
                "score": 0.0,
                "total_mentions": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "avg_engagement": 0.0,
                "mention_velocity": 0.0,
            }

        bullish = sum(1 for p in posts if p["sentiment"] == "bullish")
        bearish = sum(1 for p in posts if p["sentiment"] == "bearish")
        neutral = sum(1 for p in posts if p["sentiment"] == "neutral")
        total = len(posts)

        # Weighted score by engagement (upvotes)
        weighted_bull = sum(
            max(p["score"], 1) for p in posts if p["sentiment"] == "bullish"
        )
        weighted_bear = sum(
            max(p["score"], 1) for p in posts if p["sentiment"] == "bearish"
        )
        total_weight = weighted_bull + weighted_bear
        if total_weight > 0:
            score = (weighted_bull - weighted_bear) / total_weight
        else:
            score = 0.0

        avg_engagement = sum(p["score"] for p in posts) / total if total > 0 else 0.0

        return {
            "score": score,
            "total_mentions": total,
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": neutral,
            "avg_engagement": avg_engagement,
            "mention_velocity": total / len(_DEFAULT_SUBREDDITS),
        }

    @staticmethod
    def _analyse_sentiment(text: str) -> str:
        """簡單關鍵字情緒分析。"""
        text_lower = text.lower()
        words = set(text_lower.split())

        bull_hits = len(words & _BULLISH_WORDS)
        bear_hits = len(words & _BEARISH_WORDS)

        if bull_hits > bear_hits:
            return "bullish"
        elif bear_hits > bull_hits:
            return "bearish"
        return "neutral"

    async def health_check(self) -> bool:
        """驗證 Reddit API 連線。"""
        if not self._client_id or not self._client_secret:
            self.logger.warning("[Reddit] No credentials — degraded mode")
            return False
        try:
            token = await self._authenticate()
            return token is not None
        except Exception:
            return False
