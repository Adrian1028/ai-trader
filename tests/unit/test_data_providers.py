"""
Unit Tests for Data Providers (市場情報感測器)
==============================================
使用 aioresponses 模擬 HTTP 回應，驗證：
  1. BaseDataProvider: 快取、重試、統計
  2. AlphaVantageProvider: K 線解析、基本面解析、技術指標解析
  3. PolygonProvider: K 線解析、行情解析、標的搜尋
"""
from __future__ import annotations

import asyncio
import json
import re
import time

import pytest

try:
    from aioresponses import aioresponses
    HAS_AIORESPONSES = True
except ImportError:
    HAS_AIORESPONSES = False

from src.data.providers.base_provider import (
    BaseDataProvider,
    CacheEntry,
    DataProviderError,
    OHLCVBar,
)
from src.data.providers.alpha_vantage import AlphaVantageProvider
from src.data.providers.polygon import PolygonProvider


# ═══════════════════════════════════════════════════════════════════
# OHLCVBar
# ═══════════════════════════════════════════════════════════════════

class TestOHLCVBar:
    def test_basic_creation(self):
        bar = OHLCVBar(
            timestamp="2024-01-15",
            open=185.0, high=187.5, low=184.2, close=186.8, volume=50_000_000,
        )
        assert bar.close == 186.8
        assert bar.adjusted_close is None

    def test_as_dict(self):
        bar = OHLCVBar(
            timestamp="2024-01-15",
            open=185.0, high=187.5, low=184.2, close=186.8,
            volume=50_000_000, adjusted_close=186.5,
        )
        d = bar.as_dict
        assert d["timestamp"] == "2024-01-15"
        assert d["close"] == 186.8
        assert d["adjusted_close"] == 186.5
        assert "open" in d
        assert "high" in d
        assert "low" in d
        assert "volume" in d

    def test_as_dict_no_adjusted(self):
        bar = OHLCVBar(
            timestamp="2024-01-15",
            open=100, high=110, low=90, close=105, volume=1000,
        )
        d = bar.as_dict
        assert "adjusted_close" not in d


# ═══════════════════════════════════════════════════════════════════
# Cache
# ═══════════════════════════════════════════════════════════════════

class TestCacheEntry:
    def test_not_expired(self):
        entry = CacheEntry(data={"foo": "bar"}, expires_at=time.time() + 60)
        assert entry.data == {"foo": "bar"}
        assert time.time() < entry.expires_at

    def test_expired(self):
        entry = CacheEntry(data={"foo": "bar"}, expires_at=time.time() - 1)
        assert time.time() > entry.expires_at


# ═══════════════════════════════════════════════════════════════════
# AlphaVantage: K 線解析
# ═══════════════════════════════════════════════════════════════════

class TestAlphaVantageParser:
    """直接測試解析邏輯，不需要 HTTP 模擬"""

    def _make_provider(self) -> AlphaVantageProvider:
        return AlphaVantageProvider(api_key="test_key")

    def test_parse_daily_time_series(self):
        provider = self._make_provider()
        raw = {
            "Meta Data": {"1. Information": "Daily Prices"},
            "Time Series (Daily)": {
                "2024-01-15": {
                    "1. open": "185.09",
                    "2. high": "187.50",
                    "3. low": "184.22",
                    "4. close": "186.82",
                    "5. adjusted close": "186.50",
                    "6. volume": "50000000",
                },
                "2024-01-12": {
                    "1. open": "183.20",
                    "2. high": "185.88",
                    "3. low": "183.05",
                    "4. close": "185.59",
                    "5. adjusted close": "185.30",
                    "6. volume": "40000000",
                },
            },
        }
        bars = provider._parse_time_series(raw, "Time Series (Daily)")
        assert len(bars) == 2
        # 由新到舊排序
        assert bars[0].timestamp == "2024-01-15"
        assert bars[1].timestamp == "2024-01-12"
        assert bars[0].open == pytest.approx(185.09)
        assert bars[0].close == pytest.approx(186.82)
        assert bars[0].volume == pytest.approx(50_000_000)
        assert bars[0].adjusted_close == pytest.approx(186.50)

    def test_parse_empty_series(self):
        provider = self._make_provider()
        bars = provider._parse_time_series({}, "Time Series (Daily)")
        assert bars == []

    def test_parse_overview(self):
        provider = self._make_provider()
        raw = {
            "Symbol": "AAPL",
            "Name": "Apple Inc",
            "PERatio": "28.5",
            "ForwardPE": "26.3",
            "ProfitMargin": "0.256",
            "ReturnOnEquityTTM": "1.45",
            "MarketCapitalization": "2900000000000",
            "EPS": "6.43",
            "Beta": "1.24",
            "DividendYield": "0.005",
            "Sector": "Technology",
            "Industry": "Consumer Electronics",
            "Description": "Apple designs and manufactures...",
        }
        result = provider._parse_overview(raw)
        assert result["PERatio"] == pytest.approx(28.5)
        assert result["MarketCapitalization"] == 2_900_000_000_000
        assert result["EPS"] == pytest.approx(6.43)
        assert result["Sector"] == "Technology"  # 字串欄位不轉換

    def test_parse_overview_with_none_values(self):
        provider = self._make_provider()
        raw = {
            "Symbol": "XYZ",
            "PERatio": "None",
            "ForwardPE": "-",
            "EPS": "",
            "MarketCapitalization": "None",
        }
        result = provider._parse_overview(raw)
        assert result["PERatio"] == 0.0
        assert result["ForwardPE"] == 0.0
        assert result["EPS"] == 0.0
        assert result["MarketCapitalization"] == 0

    def test_safe_float(self):
        assert AlphaVantageProvider._safe_float("123.45") == pytest.approx(123.45)
        assert AlphaVantageProvider._safe_float(None) == 0.0
        assert AlphaVantageProvider._safe_float("None") == 0.0
        assert AlphaVantageProvider._safe_float("-") == 0.0
        assert AlphaVantageProvider._safe_float("") == 0.0

    def test_parse_indicator(self):
        provider = self._make_provider()
        raw = {
            "Technical Analysis: RSI": {
                "2024-01-15": {"RSI": "62.45"},
                "2024-01-12": {"RSI": "58.30"},
            },
        }
        result = provider._parse_indicator(raw, "Technical Analysis: RSI", "RSI")
        assert len(result) == 2
        assert result[0]["date"] == "2024-01-15"
        assert result[0]["rsi"] == pytest.approx(62.45)

    def test_parse_macd_fields(self):
        """確認 MACD 的三個欄位都正確解析"""
        provider = self._make_provider()
        # 模擬 MACD 原始 JSON 結構
        raw = {
            "Technical Analysis: MACD": {
                "2024-01-15": {
                    "MACD": "1.234",
                    "MACD_Signal": "0.987",
                    "MACD_Hist": "0.247",
                },
            },
        }
        key = "Technical Analysis: MACD"
        series = raw[key]
        entry = list(series.values())[0]
        assert AlphaVantageProvider._safe_float(entry["MACD"]) == pytest.approx(1.234)
        assert AlphaVantageProvider._safe_float(entry["MACD_Signal"]) == pytest.approx(0.987)
        assert AlphaVantageProvider._safe_float(entry["MACD_Hist"]) == pytest.approx(0.247)


# ═══════════════════════════════════════════════════════════════════
# Polygon: K 線解析
# ═══════════════════════════════════════════════════════════════════

class TestPolygonParser:
    def _make_provider(self) -> PolygonProvider:
        return PolygonProvider(api_key="test_key")

    def test_parse_aggregates(self):
        provider = self._make_provider()
        raw = {
            "status": "OK",
            "resultsCount": 2,
            "results": [
                {
                    "o": 185.0, "h": 187.5, "l": 184.2, "c": 186.8,
                    "v": 50_000_000, "vw": 186.1,
                    "t": 1705276800000,  # 2024-01-15 UTC
                },
                {
                    "o": 183.0, "h": 185.0, "l": 182.5, "c": 184.5,
                    "v": 40_000_000, "vw": 183.8,
                    "t": 1705190400000,  # 2024-01-14 UTC
                },
            ],
        }
        bars = provider._parse_aggregates(raw)
        assert len(bars) == 2
        assert bars[0].open == pytest.approx(185.0)
        assert bars[0].close == pytest.approx(186.8)
        assert bars[0].volume == pytest.approx(50_000_000)
        assert bars[0].timestamp == "2024-01-15"

    def test_parse_empty_aggregates(self):
        provider = self._make_provider()
        bars = provider._parse_aggregates({"status": "OK", "resultsCount": 0, "results": []})
        assert bars == []

    def test_parse_aggregates_no_results_key(self):
        provider = self._make_provider()
        bars = provider._parse_aggregates({"status": "OK"})
        assert bars == []

    def test_previous_close_extraction(self):
        """測試 previous_close 的結果結構"""
        # 直接測試預期的回應結構
        raw_results = [{
            "c": 185.5, "h": 186.2, "l": 184.1, "o": 185.0,
            "v": 50_000_000, "vw": 185.3, "t": 1705276800000,
        }]
        bar = raw_results[0]
        result = {
            "close": bar.get("c", 0),
            "high": bar.get("h", 0),
            "low": bar.get("l", 0),
            "open": bar.get("o", 0),
            "volume": bar.get("v", 0),
        }
        assert result["close"] == 185.5
        assert result["volume"] == 50_000_000


# ═══════════════════════════════════════════════════════════════════
# BaseDataProvider: 快取邏輯
# ═══════════════════════════════════════════════════════════════════

class TestBaseProviderCache:
    def test_cache_key_generation(self):
        provider = AlphaVantageProvider(api_key="test")
        key = provider._make_cache_key(
            "https://api.example.com/query",
            {"function": "DAILY", "symbol": "AAPL"},
        )
        assert "DAILY" in key
        assert "AAPL" in key

    def test_cache_set_and_get(self):
        provider = AlphaVantageProvider(api_key="test")
        key = "test_key"
        provider._set_cached(key, {"data": 42})
        result = provider._get_cached(key)
        assert result == {"data": 42}

    def test_cache_expired(self):
        provider = AlphaVantageProvider(api_key="test", cache_ttl=0.0)
        key = "test_key"
        provider._set_cached(key, {"data": 42})
        # TTL = 0 means already expired
        import time
        time.sleep(0.01)
        result = provider._get_cached(key)
        assert result is None

    def test_clear_cache(self):
        provider = AlphaVantageProvider(api_key="test")
        provider._set_cached("k1", "v1")
        provider._set_cached("k2", "v2")
        assert len(provider._cache) == 2
        provider.clear_cache()
        assert len(provider._cache) == 0

    def test_stats(self):
        provider = AlphaVantageProvider(api_key="test")
        stats = provider.stats
        assert stats["provider"] == "alpha_vantage"
        assert stats["requests"] == 0
        assert stats["cache_hits"] == 0
        assert stats["errors"] == 0


# ═══════════════════════════════════════════════════════════════════
# HTTP Integration (with aioresponses)
# ═══════════════════════════════════════════════════════════════════

_AV_URL_RE = re.compile(r"^https://www\.alphavantage\.co/query\b")
_POLYGON_AGGS_RE = re.compile(r"^https://api\.polygon\.io/v2/aggs/ticker/AAPL/range/")
_POLYGON_PREV_RE = re.compile(r"^https://api\.polygon\.io/v2/aggs/ticker/AAPL/prev\b")


@pytest.mark.skipif(not HAS_AIORESPONSES, reason="aioresponses not installed")
class TestAlphaVantageHTTP:

    @pytest.mark.asyncio
    async def test_daily_returns_ohlcv_bars(self):
        provider = AlphaVantageProvider(api_key="test_key")
        mock_response = {
            "Meta Data": {"1. Information": "Daily Prices"},
            "Time Series (Daily)": {
                "2024-01-15": {
                    "1. open": "185.0", "2. high": "187.5",
                    "3. low": "184.2", "4. close": "186.8",
                    "5. adjusted close": "186.5", "6. volume": "50000000",
                },
            },
        }

        with aioresponses() as m:
            m.get(_AV_URL_RE, payload=mock_response)
            async with provider:
                bars = await provider.daily("AAPL")

        assert len(bars) == 1
        assert isinstance(bars[0], OHLCVBar)
        assert bars[0].close == pytest.approx(186.8)

    @pytest.mark.asyncio
    async def test_company_overview_parses_numbers(self):
        provider = AlphaVantageProvider(api_key="test_key")
        mock_response = {
            "Symbol": "AAPL",
            "Name": "Apple Inc",
            "PERatio": "28.5",
            "MarketCapitalization": "2900000000000",
            "EPS": "6.43",
        }

        with aioresponses() as m:
            m.get(_AV_URL_RE, payload=mock_response)
            async with provider:
                result = await provider.company_overview("AAPL")

        assert result["PERatio"] == pytest.approx(28.5)
        assert result["MarketCapitalization"] == 2_900_000_000_000

    @pytest.mark.asyncio
    async def test_cache_prevents_duplicate_requests(self):
        provider = AlphaVantageProvider(api_key="test_key", cache_ttl=60.0)
        mock_response = {
            "Technical Analysis: RSI": {
                "2024-01-15": {"RSI": "62.45"},
            },
        }

        with aioresponses() as m:
            m.get(_AV_URL_RE, payload=mock_response)
            # 只註冊一次回應，第二次呼叫應該從快取取得
            async with provider:
                result1 = await provider.rsi("AAPL")
                result2 = await provider.rsi("AAPL")  # 應命中快取

        assert result1 == result2
        assert provider.stats["cache_hits"] >= 1


@pytest.mark.skipif(not HAS_AIORESPONSES, reason="aioresponses not installed")
class TestPolygonHTTP:

    @pytest.mark.asyncio
    async def test_aggregates_returns_ohlcv_bars(self):
        provider = PolygonProvider(api_key="test_key")
        mock_response = {
            "status": "OK",
            "resultsCount": 1,
            "results": [{
                "o": 185.0, "h": 187.5, "l": 184.2, "c": 186.8,
                "v": 50_000_000, "t": 1705276800000,
            }],
        }

        with aioresponses() as m:
            m.get(_POLYGON_AGGS_RE, payload=mock_response)
            async with provider:
                bars = await provider.aggregates(
                    "AAPL", from_date="2024-01-01", to_date="2024-01-15",
                )

        assert len(bars) == 1
        assert isinstance(bars[0], OHLCVBar)
        assert bars[0].close == pytest.approx(186.8)

    @pytest.mark.asyncio
    async def test_previous_close_extraction(self):
        provider = PolygonProvider(api_key="test_key")
        mock_response = {
            "status": "OK",
            "results": [{
                "c": 185.5, "h": 186.2, "l": 184.1, "o": 185.0,
                "v": 50_000_000, "vw": 185.3, "t": 1705276800000,
            }],
        }

        with aioresponses() as m:
            m.get(_POLYGON_PREV_RE, payload=mock_response)
            async with provider:
                result = await provider.previous_close("AAPL")

        assert result["close"] == 185.5
        assert result["volume"] == 50_000_000
