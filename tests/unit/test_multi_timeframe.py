"""
Unit tests for Multi-Timeframe Analysis (Phase 1 Upgrade)
==========================================================
Tests cover:
  - _timeframe_bias(): bias extraction from indicator dicts
  - _multi_timeframe_confluence(): 3-timeframe alignment scoring
  - _bars_to_indicators(): OHLCVBar → indicator dict conversion
  - Full analyse() with mocked multi-timeframe data
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from src.agents.intelligence.technical import TechnicalAgent
from src.core.base_agent import SignalDirection
from src.data.providers.base_provider import OHLCVBar


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_bars(prices: list[float], volume: float = 1_000_000) -> list[OHLCVBar]:
    """Create synthetic OHLCVBar list from closing prices."""
    bars = []
    for i, p in enumerate(prices):
        bars.append(OHLCVBar(
            timestamp=f"2025-01-{i+1:02d}",
            open=p * 0.99,
            high=p * 1.01,
            low=p * 0.98,
            close=p,
            volume=volume,
        ))
    return bars


def _uptrend_bars(n: int = 250, start: float = 100.0) -> list[OHLCVBar]:
    """Generate a clean uptrend (good for testing bullish bias)."""
    prices = [start + i * 0.5 for i in range(n)]
    return _make_bars(prices)


def _downtrend_bars(n: int = 250, start: float = 200.0) -> list[OHLCVBar]:
    """Generate a clean downtrend (good for testing bearish bias)."""
    prices = [start - i * 0.5 for i in range(n)]
    return _make_bars(prices)


def _flat_bars(n: int = 250, price: float = 100.0) -> list[OHLCVBar]:
    """Generate flat/sideways bars."""
    prices = [price + np.sin(i * 0.1) * 2 for i in range(n)]
    return _make_bars(prices)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def agent():
    av = AsyncMock()
    polygon = AsyncMock()
    return TechnicalAgent(av, polygon)


# ══════════════════════════════════════════════════════════════
# _timeframe_bias tests
# ══════════════════════════════════════════════════════════════

class TestTimeframeBias:
    def test_bullish_bias(self, agent):
        ind = {
            "sma_50_above_200": True,
            "price_above_sma200": True,
            "rsi_14": 65.0,
        }
        bias = agent._timeframe_bias(ind)
        assert bias > 0, f"Expected positive bias, got {bias}"

    def test_bearish_bias(self, agent):
        ind = {
            "sma_50_above_200": False,
            "price_above_sma200": False,
            "rsi_14": 35.0,
        }
        bias = agent._timeframe_bias(ind)
        assert bias < 0, f"Expected negative bias, got {bias}"

    def test_neutral_bias(self, agent):
        ind = {
            "sma_50_above_200": True,
            "price_above_sma200": False,
            "rsi_14": 50.0,
        }
        bias = agent._timeframe_bias(ind)
        assert abs(bias) < 0.5, f"Expected near-neutral bias, got {bias}"

    def test_empty_indicators(self, agent):
        assert agent._timeframe_bias({}) == 0.0

    def test_partial_indicators_sma_only(self, agent):
        ind = {"sma_50_above_200": True}
        bias = agent._timeframe_bias(ind)
        assert bias > 0

    def test_partial_indicators_rsi_only(self, agent):
        ind = {"rsi_14": 25.0}
        bias = agent._timeframe_bias(ind)
        assert bias < 0

    def test_uses_sma20_fallback(self, agent):
        ind = {"price_to_sma20_pct": 0.05}  # 5% above SMA20
        bias = agent._timeframe_bias(ind)
        assert bias > 0


# ══════════════════════════════════════════════════════════════
# _multi_timeframe_confluence tests
# ══════════════════════════════════════════════════════════════

class TestMultiTimeframeConfluence:
    def test_all_bullish(self, agent):
        score, reasons = agent._multi_timeframe_confluence(0.8, 0.6, 0.5)
        assert score == 20.0
        assert any("CONFLUENCE" in r and "bullish" in r for r in reasons)

    def test_all_bearish(self, agent):
        score, reasons = agent._multi_timeframe_confluence(-0.7, -0.5, -0.3)
        assert score == -20.0
        assert any("CONFLUENCE" in r and "bearish" in r for r in reasons)

    def test_divergence_bull_vs_bear(self, agent):
        score, reasons = agent._multi_timeframe_confluence(0.8, -0.6, 0.5)
        assert score == -15.0
        assert any("DIVERGENCE" in r for r in reasons)

    def test_insufficient_timeframes(self, agent):
        score, reasons = agent._multi_timeframe_confluence(0.0, 0.5, 0.0)
        assert score == 0.0
        assert any("insufficient" in r for r in reasons)

    def test_two_bullish_one_neutral(self, agent):
        # Weekly bullish, Daily bullish, Hourly near-zero
        score, reasons = agent._multi_timeframe_confluence(0.8, 0.6, 0.05)
        # Only 2 active timeframes, both bullish → all same direction
        assert score == 20.0

    def test_all_neutral(self, agent):
        score, reasons = agent._multi_timeframe_confluence(0.0, 0.0, 0.0)
        assert score == 0.0


# ══════════════════════════════════════════════════════════════
# _bars_to_indicators tests
# ══════════════════════════════════════════════════════════════

class TestBarsToIndicators:
    def test_uptrend_has_bullish_sma(self, agent):
        bars = _uptrend_bars(250)
        ind = agent._bars_to_indicators(bars)
        assert ind["sma_50_above_200"] == True
        assert ind["price_above_sma200"] == True

    def test_downtrend_has_bearish_sma(self, agent):
        bars = _downtrend_bars(250)
        ind = agent._bars_to_indicators(bars)
        assert ind["sma_50_above_200"] == False
        assert ind["price_above_sma200"] == False

    def test_rsi_present(self, agent):
        bars = _uptrend_bars(50)
        ind = agent._bars_to_indicators(bars)
        assert "rsi_14" in ind
        assert ind["rsi_14"] is not None

    def test_bollinger_present_with_20_bars(self, agent):
        bars = _flat_bars(30)
        ind = agent._bars_to_indicators(bars)
        assert "bb_width" in ind
        assert "bb_position" in ind

    def test_short_bars_graceful(self, agent):
        bars = _make_bars([100, 101, 102])
        ind = agent._bars_to_indicators(bars)
        # Should not crash, RSI will be None
        assert ind.get("rsi_14") is None


# ══════════════════════════════════════════════════════════════
# Full analyse() integration tests
# ══════════════════════════════════════════════════════════════

class TestAnalyseMultiTimeframe:
    def test_all_timeframes_bullish(self, agent):
        agent._polygon = AsyncMock()

        async def mock_daily(ticker, **kw):
            return _uptrend_bars(250)

        async def mock_agg(ticker, **kw):
            timespan = kw.get("timespan", "day")
            if timespan == "week":
                return _uptrend_bars(104)
            elif timespan == "hour":
                return _uptrend_bars(100)
            return _uptrend_bars(250)

        agent._polygon.daily_bars = AsyncMock(side_effect=mock_daily)
        agent._polygon.aggregates = AsyncMock(side_effect=mock_agg)

        result = run(agent.analyse({"ticker": "AAPL"}))
        assert result.direction in (SignalDirection.BUY, SignalDirection.STRONG_BUY)
        assert result.data["confluence_score"] == 20.0
        assert result.data["weekly_bias"] > 0
        assert result.data["hourly_bias"] > 0

    def test_timeframe_divergence_reduces_confidence(self, agent):
        async def mock_daily(ticker, **kw):
            return _uptrend_bars(250)

        async def mock_agg(ticker, **kw):
            timespan = kw.get("timespan", "day")
            if timespan == "week":
                return _downtrend_bars(104)  # weekly bearish
            elif timespan == "hour":
                return _uptrend_bars(100)    # hourly bullish
            return _uptrend_bars(250)

        agent._polygon.daily_bars = AsyncMock(side_effect=mock_daily)
        agent._polygon.aggregates = AsyncMock(side_effect=mock_agg)

        result = run(agent.analyse({"ticker": "AAPL"}))
        assert result.data["confluence_score"] == -15.0
        assert result.data["weekly_bias"] < 0

    def test_missing_hourly_weekly_still_works(self, agent):
        """When hourly/weekly data unavailable, daily alone still produces a signal."""
        async def mock_daily(ticker, **kw):
            return _uptrend_bars(250)

        async def mock_agg(ticker, **kw):
            return []  # no data

        agent._polygon.daily_bars = AsyncMock(side_effect=mock_daily)
        agent._polygon.aggregates = AsyncMock(side_effect=mock_agg)
        agent._av = AsyncMock()
        agent._av.daily = AsyncMock(return_value=[])

        result = run(agent.analyse({"ticker": "AAPL"}))
        assert result.source == "technical"
        # Daily uptrend detected (SMA bullish), but monotonic prices give RSI=100
        # which triggers mean-reversion bearish offset → composite near zero.
        # Key check: daily_bias should be positive (uptrend correctly detected).
        assert result.data["daily_bias"] > 0
        assert result.data["weekly_bias"] == 0.0  # no weekly data
        assert result.data["hourly_bias"] == 0.0  # no hourly data

    def test_no_data_returns_neutral(self, agent):
        agent._polygon.daily_bars = AsyncMock(return_value=[])
        agent._polygon.aggregates = AsyncMock(return_value=[])
        agent._av = AsyncMock()
        agent._av.daily = AsyncMock(return_value=[])

        result = run(agent.analyse({"ticker": "AAPL"}))
        assert result.direction == SignalDirection.NEUTRAL
        assert "無法獲取" in result.reasoning

    def test_data_dict_contains_mtf_fields(self, agent):
        async def mock_daily(ticker, **kw):
            return _uptrend_bars(250)

        async def mock_agg(ticker, **kw):
            return _uptrend_bars(104)

        agent._polygon.daily_bars = AsyncMock(side_effect=mock_daily)
        agent._polygon.aggregates = AsyncMock(side_effect=mock_agg)

        result = run(agent.analyse({"ticker": "AAPL"}))
        # Verify all MTF fields are present in data
        assert "weekly_bias" in result.data
        assert "daily_bias" in result.data
        assert "hourly_bias" in result.data
        assert "confluence_score" in result.data
        assert "weekly_rsi" in result.data
        assert "hourly_rsi" in result.data
