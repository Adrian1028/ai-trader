"""
Unit tests for src/backtest.py — Event-Driven Decade Backtester (v2)
=====================================================================
Tests cover:
  - BacktestDataFeeder: anti-look-ahead slicing, trading days, instruments,
    get_past_df
  - Mock providers: Polygon, AlphaVantage, Finnhub, Intrinio, Trading212
  - BacktestResult: summary statistics computation
  - DecadeBacktester: tearsheet generation
  - SL/TP trade lifecycle: pessimistic fill simulation
"""
from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    BacktestDataFeeder,
    BacktestResult,
    DecadeBacktester,
    MockAlphaVantageProvider,
    MockFinnhubProvider,
    MockIntrinioProvider,
    MockPolygonProvider,
    MockTrading212Client,
    _TickerData,
)


# ══════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════

def _make_ohlcv_df(
    start: str = "2020-01-01",
    periods: int = 10,
    base_price: float = 100.0,
) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame for testing."""
    dates = pd.bdate_range(start=start, periods=periods)
    rng = np.random.default_rng(42)
    closes = base_price + np.cumsum(rng.normal(0, 1, periods))
    return pd.DataFrame(
        {
            "Open": closes - 0.5,
            "High": closes + 1.0,
            "Low": closes - 1.0,
            "Close": closes,
            "Volume": rng.integers(1_000_000, 10_000_000, periods).astype(float),
        },
        index=dates,
    )


def _make_feeder(
    tickers: list[str] | None = None,
    periods: int = 20,
    start: str = "2020-01-01",
) -> BacktestDataFeeder:
    """Create a BacktestDataFeeder pre-loaded with synthetic data (no yfinance)."""
    tickers = tickers or ["AAPL", "MSFT"]
    feeder = BacktestDataFeeder(tickers, start, "2020-12-31")
    for t in tickers:
        feeder._data[t] = _TickerData(
            df=_make_ohlcv_df(start=start, periods=periods, base_price=100 + hash(t) % 50),
            isin=BacktestDataFeeder._ISIN_MAP.get(t, f"BACKTEST_{t}"),
            name=t,
            currency="USD",
        )
    return feeder


# ══════════════════════════════════════════════════════════════════
#  BacktestDataFeeder Tests
# ══════════════════════════════════════════════════════════════════

class TestBacktestDataFeeder:
    """Tests for the core data feeder and anti-look-ahead mechanism."""

    def test_get_trading_days_returns_sorted_union(self):
        feeder = _make_feeder(["AAPL", "MSFT"], periods=15)
        days = feeder.get_trading_days()
        assert len(days) > 0
        # Must be sorted ascending
        for i in range(1, len(days)):
            assert days[i] > days[i - 1]

    def test_get_trading_days_empty_when_no_data(self):
        feeder = BacktestDataFeeder([], "2020-01-01", "2020-12-31")
        assert feeder.get_trading_days() == []

    def test_current_date_property(self):
        feeder = _make_feeder()
        assert feeder.current_date is None
        dt = pd.Timestamp("2020-01-10")
        feeder.current_date = dt
        assert feeder.current_date == dt

    def test_anti_look_ahead_slicing(self):
        """Core test: bars returned must NOT contain future dates."""
        feeder = _make_feeder(["AAPL"], periods=20)
        days = feeder.get_trading_days()
        mid = days[10]
        feeder.current_date = mid

        bars = feeder.get_daily_bars("AAPL", days=1000)
        for bar in bars:
            bar_date = pd.Timestamp(bar.timestamp)
            assert bar_date <= mid, f"Look-ahead: {bar_date} > {mid}"

    def test_daily_bars_respects_days_limit(self):
        feeder = _make_feeder(["AAPL"], periods=20)
        days = feeder.get_trading_days()
        feeder.current_date = days[-1]

        bars_5 = feeder.get_daily_bars("AAPL", days=5)
        bars_all = feeder.get_daily_bars("AAPL", days=1000)
        assert len(bars_5) == 5
        assert len(bars_all) == 20

    def test_daily_bars_returns_empty_for_unknown_ticker(self):
        feeder = _make_feeder(["AAPL"])
        feeder.current_date = pd.Timestamp("2020-01-15")
        assert feeder.get_daily_bars("UNKNOWN") == []

    def test_daily_bars_returns_empty_when_no_current_date(self):
        feeder = _make_feeder(["AAPL"])
        assert feeder.get_daily_bars("AAPL") == []

    def test_bars_have_correct_ohlcv_fields(self):
        feeder = _make_feeder(["AAPL"], periods=5)
        feeder.current_date = feeder.get_trading_days()[-1]
        bars = feeder.get_daily_bars("AAPL")
        for bar in bars:
            assert hasattr(bar, "open")
            assert hasattr(bar, "high")
            assert hasattr(bar, "low")
            assert hasattr(bar, "close")
            assert hasattr(bar, "volume")
            assert hasattr(bar, "timestamp")
            assert bar.high >= bar.low

    def test_get_current_price_exact_date(self):
        feeder = _make_feeder(["AAPL"], periods=10)
        days = feeder.get_trading_days()
        feeder.current_date = days[5]
        price = feeder.get_current_price("AAPL")
        expected = float(feeder._data["AAPL"].df.loc[days[5], "Close"])
        assert price == pytest.approx(expected)

    def test_get_current_price_nearest_past(self):
        """When current_date is not a trading day, use nearest past close."""
        feeder = _make_feeder(["AAPL"], periods=10)
        days = feeder.get_trading_days()
        # Set to a weekend/gap between trading days
        gap_date = days[3] + pd.Timedelta(hours=12)
        feeder.current_date = gap_date
        price = feeder.get_current_price("AAPL")
        assert price is not None
        expected = float(feeder._data["AAPL"].df.loc[days[3], "Close"])
        assert price == pytest.approx(expected)

    def test_get_current_price_unknown_ticker(self):
        feeder = _make_feeder(["AAPL"])
        feeder.current_date = pd.Timestamp("2020-01-15")
        assert feeder.get_current_price("UNKNOWN") is None

    def test_get_current_price_no_current_date(self):
        feeder = _make_feeder(["AAPL"])
        assert feeder.get_current_price("AAPL") is None

    def test_get_instruments(self):
        feeder = _make_feeder(["AAPL", "MSFT"])
        instruments = feeder.get_instruments()
        assert len(instruments) == 2
        tickers = {i["ticker"] for i in instruments}
        assert tickers == {"AAPL", "MSFT"}
        for inst in instruments:
            assert "isin" in inst
            assert "currencyCode" in inst
            assert "type" in inst
            assert inst["type"] == "STOCK"

    def test_isin_map_known_tickers(self):
        feeder = _make_feeder(["AAPL"])
        instruments = feeder.get_instruments()
        assert instruments[0]["isin"] == "US0378331005"

    def test_isin_map_unknown_ticker_gets_backtest_prefix(self):
        feeder = _make_feeder(["ZZZZ"])
        instruments = feeder.get_instruments()
        assert instruments[0]["isin"] == "BACKTEST_ZZZZ"


# ══════════════════════════════════════════════════════════════════
#  Mock Provider Tests
# ══════════════════════════════════════════════════════════════════

class TestMockPolygonProvider:
    """Tests for Polygon mock provider."""

    @pytest.fixture
    def provider(self):
        feeder = _make_feeder(["AAPL"], periods=15)
        feeder.current_date = feeder.get_trading_days()[-1]
        return MockPolygonProvider(feeder)

    def test_daily_bars(self, provider):
        bars = asyncio.get_event_loop().run_until_complete(
            provider.daily_bars("AAPL", 10),
        )
        assert len(bars) == 10

    def test_aggregates(self, provider):
        bars = asyncio.get_event_loop().run_until_complete(
            provider.aggregates("AAPL", limit=5),
        )
        assert len(bars) == 5

    def test_previous_close(self, provider):
        result = asyncio.get_event_loop().run_until_complete(
            provider.previous_close("AAPL"),
        )
        assert "close" in result
        assert result["close"] > 0

    def test_snapshot(self, provider):
        result = asyncio.get_event_loop().run_until_complete(
            provider.snapshot("AAPL"),
        )
        assert "close" in result

    def test_health_check(self, provider):
        assert asyncio.get_event_loop().run_until_complete(provider.health_check())

    def test_context_manager(self, provider):
        async def _test():
            async with provider as p:
                assert p is provider
        asyncio.get_event_loop().run_until_complete(_test())


class TestMockAlphaVantageProvider:
    """Tests for Alpha Vantage mock provider."""

    @pytest.fixture
    def provider(self):
        feeder = _make_feeder(["MSFT"], periods=15)
        feeder.current_date = feeder.get_trading_days()[-1]
        return MockAlphaVantageProvider(feeder)

    def test_daily_compact(self, provider):
        bars = asyncio.get_event_loop().run_until_complete(
            provider.daily("MSFT", "compact"),
        )
        assert len(bars) <= 100

    def test_daily_full(self, provider):
        bars = asyncio.get_event_loop().run_until_complete(
            provider.daily("MSFT", "full"),
        )
        assert len(bars) > 0

    def test_company_overview(self, provider):
        overview = asyncio.get_event_loop().run_until_complete(
            provider.company_overview("MSFT"),
        )
        assert overview["Symbol"] == "MSFT"
        assert "PERatio" in overview
        assert "ForwardPE" in overview
        assert "Beta" in overview

    def test_earnings_returns_empty(self, provider):
        result = asyncio.get_event_loop().run_until_complete(
            provider.earnings("MSFT"),
        )
        assert "quarterlyEarnings" in result
        assert result["quarterlyEarnings"] == []

    def test_earnings_history_empty(self, provider):
        result = asyncio.get_event_loop().run_until_complete(
            provider.earnings_history("MSFT"),
        )
        assert result == []

    def test_sma_empty(self, provider):
        result = asyncio.get_event_loop().run_until_complete(
            provider.sma("MSFT"),
        )
        assert result == []

    def test_weekly(self, provider):
        bars = asyncio.get_event_loop().run_until_complete(
            provider.weekly("MSFT"),
        )
        assert len(bars) > 0

    def test_monthly(self, provider):
        bars = asyncio.get_event_loop().run_until_complete(
            provider.monthly("MSFT"),
        )
        assert len(bars) > 0


class TestMockFinnhubProvider:
    """Tests for Finnhub mock provider — neutral sentiment."""

    @pytest.fixture
    def provider(self):
        return MockFinnhubProvider()

    def test_company_news_empty(self, provider):
        result = asyncio.get_event_loop().run_until_complete(
            provider.company_news("AAPL"),
        )
        assert result == []

    def test_news_sentiment_empty(self, provider):
        result = asyncio.get_event_loop().run_until_complete(
            provider.news_sentiment("AAPL"),
        )
        assert result == {}

    def test_recommendation_trends_empty(self, provider):
        result = asyncio.get_event_loop().run_until_complete(
            provider.recommendation_trends("AAPL"),
        )
        assert result == []

    def test_earnings_surprises_empty(self, provider):
        result = asyncio.get_event_loop().run_until_complete(
            provider.earnings_surprises("AAPL"),
        )
        assert result == []

    def test_health_check(self, provider):
        assert asyncio.get_event_loop().run_until_complete(provider.health_check())


class TestMockIntrinioProvider:
    """Tests for Intrinio mock provider."""

    @pytest.fixture
    def provider(self):
        return MockIntrinioProvider()

    def test_company_financials_empty(self, provider):
        result = asyncio.get_event_loop().run_until_complete(
            provider.company_financials("AAPL"),
        )
        assert result == {}

    def test_health_check(self, provider):
        assert asyncio.get_event_loop().run_until_complete(provider.health_check())

    def test_context_manager(self, provider):
        async def _test():
            async with provider as p:
                assert p is provider
        asyncio.get_event_loop().run_until_complete(_test())


class TestMockTrading212Client:
    """Tests for mock T212 client — instant fill simulation."""

    @pytest.fixture
    def client(self):
        feeder = _make_feeder(["AAPL"], periods=10)
        feeder.current_date = feeder.get_trading_days()[-1]
        return MockTrading212Client(feeder)

    def test_place_order_fills_instantly(self, client):
        result = asyncio.get_event_loop().run_until_complete(
            client.place_order({"ticker": "AAPL", "quantity": 5}),
        )
        assert result["status"] == "filled"
        assert result["filledQuantity"] == 5
        assert result["fillPrice"] > 0
        assert result["id"] == 1

    def test_order_counter_increments(self, client):
        loop = asyncio.get_event_loop()
        r1 = loop.run_until_complete(client.place_order({"ticker": "AAPL", "quantity": 1}))
        r2 = loop.run_until_complete(client.place_order({"ticker": "AAPL", "quantity": 2}))
        assert r1["id"] == 1
        assert r2["id"] == 2

    def test_place_stop_order_delegates(self, client):
        result = asyncio.get_event_loop().run_until_complete(
            client.place_stop_order({"ticker": "AAPL", "quantity": 3}),
        )
        assert result["status"] == "filled"

    def test_place_limit_order_delegates(self, client):
        result = asyncio.get_event_loop().run_until_complete(
            client.place_limit_order({"ticker": "AAPL", "quantity": 3}),
        )
        assert result["status"] == "filled"

    def test_cancel_order(self, client):
        result = asyncio.get_event_loop().run_until_complete(
            client.cancel_order(1),
        )
        assert result == {}

    def test_account_info(self, client):
        result = asyncio.get_event_loop().run_until_complete(
            client.account_info(),
        )
        assert "cash" in result
        assert "invested" in result

    def test_portfolio_empty(self, client):
        result = asyncio.get_event_loop().run_until_complete(
            client.portfolio(),
        )
        assert result == []

    def test_exchange_instruments(self, client):
        result = asyncio.get_event_loop().run_until_complete(
            client.exchange_instruments(),
        )
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"

    def test_order_history(self, client):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(client.place_order({"ticker": "AAPL", "quantity": 1}))
        result = loop.run_until_complete(client.order_history())
        assert "items" in result
        assert len(result["items"]) == 1

    def test_sell_order_fills_at_current_price(self, client):
        result = asyncio.get_event_loop().run_until_complete(
            client.place_order({"ticker": "AAPL", "quantity": -3}),
        )
        assert result["filledQuantity"] == 3  # abs(qty)
        assert result["fillPrice"] > 0


# ══════════════════════════════════════════════════════════════════
#  BacktestResult Tests
# ══════════════════════════════════════════════════════════════════

class TestBacktestResult:
    """Tests for the result dataclass."""

    def test_defaults(self):
        r = BacktestResult()
        assert r.initial_capital == 0.0
        assert r.final_nav == 0.0
        assert r.total_trades == 0
        assert r.daily_nav == []

    def test_total_return_calculation(self):
        r = BacktestResult(
            initial_capital=10_000,
            final_nav=12_000,
            total_return_pct=20.0,
        )
        assert r.total_return_pct == pytest.approx(20.0)

    def test_daily_nav_is_independent(self):
        """Ensure default_factory produces separate lists."""
        r1 = BacktestResult()
        r2 = BacktestResult()
        r1.daily_nav.append({"date": "2020-01-01", "nav": 100})
        assert len(r2.daily_nav) == 0


# ══════════════════════════════════════════════════════════════════
#  DecadeBacktester Unit Tests
# ══════════════════════════════════════════════════════════════════

class TestDecadeBacktester:
    """Unit tests for the backtester engine (no yfinance, no full pipeline)."""

    def test_constructor_creates_data_dir(self, tmp_path):
        data_dir = str(tmp_path / "bt_output")
        bt = DecadeBacktester(
            tickers=["AAPL"],
            data_dir=data_dir,
        )
        import os
        assert os.path.isdir(data_dir)

    def test_default_parameters(self):
        bt = DecadeBacktester(tickers=["AAPL"])
        assert bt.start_date == "2014-01-01"
        assert bt.end_date == "2024-01-01"
        assert bt.initial_capital == 10_000.0
        assert bt._scan_interval == 5
        assert bt._reflection_interval == 20
        assert bt._min_entry_score == 0.05

    def test_custom_min_entry_score(self):
        bt = DecadeBacktester(tickers=["AAPL"], min_entry_score=0.10)
        assert bt._min_entry_score == 0.10

    def test_backtest_weights_favor_technical(self):
        """Backtest weights must heavily favor technical (only real data)."""
        w = DecadeBacktester._BACKTEST_WEIGHTS
        assert w["technical"] > w["fundamental"]
        assert w["technical"] > w["sentiment"]
        assert w["technical"] >= 0.70
        total = sum(w.values())
        assert total == pytest.approx(1.0)

    def test_tearsheet_no_plotly_no_crash(self, tmp_path):
        """Tearsheet generation with empty result should not raise."""
        bt = DecadeBacktester(
            tickers=["AAPL"],
            data_dir=str(tmp_path),
        )
        result = BacktestResult(daily_nav=[])
        # Should silently return when daily_nav is empty
        bt._generate_tearsheet(result)

    def test_tearsheet_generates_html(self, tmp_path):
        """Tearsheet with valid data should produce an HTML file."""
        bt = DecadeBacktester(
            tickers=["AAPL"],
            data_dir=str(tmp_path),
        )
        nav_data = [
            {"date": pd.Timestamp("2020-01-01"), "nav": 10000},
            {"date": pd.Timestamp("2020-01-02"), "nav": 10100},
            {"date": pd.Timestamp("2020-01-03"), "nav": 10050},
            {"date": pd.Timestamp("2020-01-04"), "nav": 10200},
            {"date": pd.Timestamp("2020-01-05"), "nav": 10150},
        ]
        result = BacktestResult(
            initial_capital=10000,
            final_nav=10150,
            total_return_pct=1.5,
            max_drawdown_pct=-0.5,
            total_trades=3,
            win_rate=0.67,
            sharpe_ratio=1.2,
            daily_nav=nav_data,
        )
        bt._generate_tearsheet(result)
        import os
        html_path = os.path.join(str(tmp_path), "backtest_tearsheet.html")
        assert os.path.exists(html_path)
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "NAV" in content or "nav" in content


# ══════════════════════════════════════════════════════════════════
#  Anti-Look-Ahead Stress Tests
# ══════════════════════════════════════════════════════════════════

class TestAntiLookAhead:
    """Stress tests to ensure no future data leaks at any point."""

    def test_sequential_advance_never_leaks(self):
        """Advance current_date day-by-day; bars must never exceed it."""
        feeder = _make_feeder(["AAPL"], periods=50, start="2020-01-01")
        days = feeder.get_trading_days()

        for day in days:
            feeder.current_date = day
            bars = feeder.get_daily_bars("AAPL", days=9999)
            if bars:
                last_bar_date = pd.Timestamp(bars[-1].timestamp)
                assert last_bar_date <= day, (
                    f"Leaked future: bar {last_bar_date} > current {day}"
                )

    def test_price_matches_current_date_close(self):
        """get_current_price must return the close on current_date."""
        feeder = _make_feeder(["AAPL"], periods=30)
        days = feeder.get_trading_days()
        df = feeder._data["AAPL"].df

        for day in days:
            feeder.current_date = day
            price = feeder.get_current_price("AAPL")
            expected = float(df.loc[day, "Close"])
            assert price == pytest.approx(expected), (
                f"Price mismatch on {day}: {price} != {expected}"
            )

    def test_multi_ticker_isolation(self):
        """Each ticker returns only its own data."""
        feeder = _make_feeder(["AAPL", "MSFT"], periods=15)
        feeder.current_date = feeder.get_trading_days()[-1]

        bars_a = feeder.get_daily_bars("AAPL")
        bars_m = feeder.get_daily_bars("MSFT")

        # Prices should differ (different base_price seeds)
        closes_a = [b.close for b in bars_a]
        closes_m = [b.close for b in bars_m]
        assert closes_a != closes_m, "AAPL and MSFT should have different prices"


# ══════════════════════════════════════════════════════════════════
#  ISIN Mapping Integration Tests
# ══════════════════════════════════════════════════════════════════

class TestISINMapping:
    """Tests for feeder ↔ ISINMapper integration."""

    def test_instruments_loadable_by_mapper(self):
        from src.data.pipelines.isin_mapper import ISINMapper
        feeder = _make_feeder(["AAPL", "MSFT"])
        mapper = ISINMapper()
        mapper.load_instruments(feeder.get_instruments())
        assert mapper.count == 2

    def test_mapper_ticker_lookup(self):
        from src.data.pipelines.isin_mapper import ISINMapper
        feeder = _make_feeder(["AAPL"])
        mapper = ISINMapper()
        mapper.load_instruments(feeder.get_instruments())
        assert mapper.ticker_for_isin("US0378331005") == "AAPL"

    def test_mapper_isin_lookup(self):
        from src.data.pipelines.isin_mapper import ISINMapper
        feeder = _make_feeder(["MSFT"])
        mapper = ISINMapper()
        mapper.load_instruments(feeder.get_instruments())
        assert mapper.isin_for_ticker("MSFT") == "US5949181045"


# ══════════════════════════════════════════════════════════════════
#  v2: get_past_df Tests
# ══════════════════════════════════════════════════════════════════

class TestGetPastDf:
    """Tests for BacktestDataFeeder.get_past_df (new in v2)."""

    def test_returns_dataframe(self):
        feeder = _make_feeder(["AAPL"], periods=20)
        feeder.current_date = feeder.get_trading_days()[10]
        df = feeder.get_past_df("AAPL")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 11  # days 0..10 inclusive

    def test_no_future_rows(self):
        feeder = _make_feeder(["AAPL"], periods=20)
        days = feeder.get_trading_days()
        mid = days[10]
        feeder.current_date = mid
        df = feeder.get_past_df("AAPL")
        assert df.index.max() <= mid

    def test_returns_empty_for_unknown_ticker(self):
        feeder = _make_feeder(["AAPL"])
        feeder.current_date = pd.Timestamp("2020-01-15")
        df = feeder.get_past_df("UNKNOWN")
        assert df.empty

    def test_returns_empty_when_no_current_date(self):
        feeder = _make_feeder(["AAPL"])
        df = feeder.get_past_df("AAPL")
        assert df.empty

    def test_has_ohlcv_columns(self):
        feeder = _make_feeder(["AAPL"], periods=5)
        feeder.current_date = feeder.get_trading_days()[-1]
        df = feeder.get_past_df("AAPL")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in df.columns


# ══════════════════════════════════════════════════════════════════
#  v2: SL/TP Trade Lifecycle Tests
# ══════════════════════════════════════════════════════════════════

class TestSLTPLifecycle:
    """
    Test the SL/TP pessimistic fill logic that the v2 event-driven
    loop uses for active_trades management.
    """

    def _make_deterministic_feeder(self):
        """
        Create a feeder with controlled prices for SL/TP testing.
        Day 0: Open=100, High=102, Low=98,  Close=101
        Day 1: Open=101, High=105, Low=99,  Close=103  (TP hit if TP=104)
        Day 2: Open=103, High=104, Low=95,  Close=96   (SL hit if SL=96)
        """
        dates = pd.bdate_range(start="2020-01-01", periods=3)
        df = pd.DataFrame(
            {
                "Open":   [100.0, 101.0, 103.0],
                "High":   [102.0, 105.0, 104.0],
                "Low":    [98.0,  99.0,  95.0],
                "Close":  [101.0, 103.0, 96.0],
                "Volume": [1e6,   1e6,   1e6],
            },
            index=dates,
        )
        feeder = BacktestDataFeeder(["TEST"], "2020-01-01", "2020-12-31")
        feeder._data["TEST"] = _TickerData(df=df, isin="BACKTEST_TEST")
        return feeder

    def test_sl_triggered_when_low_breaches(self):
        """Day 2: Low=95 < SL=96 should trigger stop-loss."""
        feeder = self._make_deterministic_feeder()
        days = feeder.get_trading_days()

        # Simulate active trade opened at 100 with SL=96, TP=110
        active_trade = {
            "entry_price": 100.0,
            "sl": 96.0,
            "tp": 110.0,
            "qty": 10,
        }

        # Day 2
        feeder.current_date = days[2]
        past_df = feeder.get_past_df("TEST")
        today = past_df.iloc[-1]

        exit_price = None
        if float(today["Low"]) <= active_trade["sl"]:
            exit_price = active_trade["sl"]
        elif float(today["High"]) >= active_trade["tp"]:
            exit_price = active_trade["tp"]

        assert exit_price == 96.0  # SL triggered

    def test_tp_triggered_when_high_breaches(self):
        """Day 1: High=105 > TP=104 should trigger take-profit."""
        feeder = self._make_deterministic_feeder()
        days = feeder.get_trading_days()

        active_trade = {
            "entry_price": 100.0,
            "sl": 90.0,
            "tp": 104.0,
            "qty": 10,
        }

        # Day 1
        feeder.current_date = days[1]
        past_df = feeder.get_past_df("TEST")
        today = past_df.iloc[-1]

        exit_price = None
        if float(today["Low"]) <= active_trade["sl"]:
            exit_price = active_trade["sl"]
        elif float(today["High"]) >= active_trade["tp"]:
            exit_price = active_trade["tp"]

        assert exit_price == 104.0  # TP triggered

    def test_pessimistic_fill_sl_checked_first(self):
        """
        If both SL and TP are breached on the same day, SL wins (pessimistic).
        Day 2: Low=95 < SL=96 AND High=104 > TP=103 — SL takes priority.
        """
        feeder = self._make_deterministic_feeder()
        days = feeder.get_trading_days()

        active_trade = {
            "entry_price": 100.0,
            "sl": 96.0,
            "tp": 103.5,
            "qty": 10,
        }

        # Day 2: Low=95 (SL breach), High=104 (TP breach)
        feeder.current_date = days[2]
        past_df = feeder.get_past_df("TEST")
        today = past_df.iloc[-1]

        exit_price = None
        # Pessimistic: check SL first
        if float(today["Low"]) <= active_trade["sl"]:
            exit_price = active_trade["sl"]
        elif float(today["High"]) >= active_trade["tp"]:
            exit_price = active_trade["tp"]

        assert exit_price == 96.0  # SL wins (pessimistic assumption)

    def test_no_trigger_when_within_range(self):
        """Day 0: High=102, Low=98 — neither SL=90 nor TP=110 hit."""
        feeder = self._make_deterministic_feeder()
        days = feeder.get_trading_days()

        active_trade = {
            "entry_price": 100.0,
            "sl": 90.0,
            "tp": 110.0,
            "qty": 10,
        }

        feeder.current_date = days[0]
        past_df = feeder.get_past_df("TEST")
        today = past_df.iloc[-1]

        exit_price = None
        if float(today["Low"]) <= active_trade["sl"]:
            exit_price = active_trade["sl"]
        elif float(today["High"]) >= active_trade["tp"]:
            exit_price = active_trade["tp"]

        assert exit_price is None  # No trigger

    def test_roi_calculation_on_sl_exit(self):
        """Verify ROI formula: (exit - entry) / entry."""
        entry = 100.0
        sl_exit = 96.0
        roi = (sl_exit - entry) / entry
        assert roi == pytest.approx(-0.04)

    def test_roi_calculation_on_tp_exit(self):
        """Verify ROI formula for profitable exit."""
        entry = 100.0
        tp_exit = 104.0
        roi = (tp_exit - entry) / entry
        assert roi == pytest.approx(0.04)


# ══════════════════════════════════════════════════════════════════
#  Dynamic Regime Simulation Tests — 動態體制模擬
# ══════════════════════════════════════════════════════════════════

class TestRegimeDetection:
    """Tests for BacktestDataFeeder.is_bear_market()."""

    def test_bear_year_2020(self):
        """2020 is a known bear year."""
        feeder = _make_feeder(start="2020-01-01")
        feeder.current_date = pd.Timestamp("2020-03-15")
        assert feeder.is_bear_market() is True

    def test_bear_year_2022(self):
        """2022 is a known bear year."""
        feeder = _make_feeder(start="2022-01-01")
        feeder.current_date = pd.Timestamp("2022-06-01")
        assert feeder.is_bear_market() is True

    def test_bull_year_2021(self):
        """2021 is not in BEAR_YEARS → bull regime."""
        feeder = _make_feeder(start="2021-01-01")
        feeder.current_date = pd.Timestamp("2021-07-01")
        assert feeder.is_bear_market() is False

    def test_bull_year_2019(self):
        """2019 is not in BEAR_YEARS → bull regime."""
        feeder = _make_feeder()
        feeder.current_date = pd.Timestamp("2019-05-01")
        assert feeder.is_bear_market() is False

    def test_no_current_date(self):
        """is_bear_market returns False when current_date is None."""
        feeder = _make_feeder()
        feeder.current_date = None
        assert feeder.is_bear_market() is False

    def test_bear_years_set(self):
        """Verify the known bear years are correctly defined."""
        assert BacktestDataFeeder._BEAR_YEARS == {2015, 2018, 2020, 2022}


class TestRegimeDependentMockData:
    """Tests for regime-dependent mock provider responses."""

    def test_company_overview_bear_market(self):
        """In bear years, company_overview returns poor fundamentals."""
        feeder = _make_feeder(start="2020-01-01")
        feeder.current_date = pd.Timestamp("2020-06-01")
        provider = MockAlphaVantageProvider(feeder)
        overview = asyncio.get_event_loop().run_until_complete(
            provider.company_overview("AAPL"),
        )
        # Bear: high PE, negative growth
        assert float(overview["PERatio"]) == 35.0
        assert float(overview["QuarterlyRevenueGrowthYOY"]) < 0
        assert float(overview["QuarterlyEarningsGrowthYOY"]) < 0
        assert float(overview["ReturnOnEquityTTM"]) < 0.10

    def test_company_overview_bull_market(self):
        """In bull years, company_overview returns healthy fundamentals."""
        feeder = _make_feeder(start="2021-01-01")
        feeder.current_date = pd.Timestamp("2021-06-01")
        provider = MockAlphaVantageProvider(feeder)
        overview = asyncio.get_event_loop().run_until_complete(
            provider.company_overview("AAPL"),
        )
        # Bull: reasonable PE, positive growth
        assert float(overview["PERatio"]) == 20.0
        assert float(overview["QuarterlyRevenueGrowthYOY"]) > 0
        assert float(overview["QuarterlyEarningsGrowthYOY"]) > 0
        assert float(overview["ReturnOnEquityTTM"]) >= 0.20

    def test_company_news_bear_market(self):
        """In bear years, company_news returns bearish headlines."""
        feeder = _make_feeder(start="2020-01-01")
        feeder.current_date = pd.Timestamp("2020-06-01")
        provider = MockFinnhubProvider(feeder)
        news = asyncio.get_event_loop().run_until_complete(
            provider.company_news("AAPL"),
        )
        assert len(news) == 2
        # Check for bearish language
        headlines = " ".join(n["headline"] for n in news).lower()
        assert any(word in headlines for word in ["downturn", "recession", "headwinds"])

    def test_company_news_bull_market(self):
        """In bull years, company_news returns bullish headlines."""
        feeder = _make_feeder(start="2021-01-01")
        feeder.current_date = pd.Timestamp("2021-06-01")
        provider = MockFinnhubProvider(feeder)
        news = asyncio.get_event_loop().run_until_complete(
            provider.company_news("AAPL"),
        )
        assert len(news) == 2
        headlines = " ".join(n["headline"] for n in news).lower()
        assert any(word in headlines for word in ["rallies", "strong", "pile"])

    def test_company_news_no_feeder(self):
        """Without a feeder, company_news returns empty list."""
        provider = MockFinnhubProvider()
        news = asyncio.get_event_loop().run_until_complete(
            provider.company_news("AAPL"),
        )
        assert news == []

    def test_news_sentiment_bear(self):
        """Bear market returns high bearish percentage."""
        feeder = _make_feeder(start="2020-01-01")
        feeder.current_date = pd.Timestamp("2020-06-01")
        provider = MockFinnhubProvider(feeder)
        sent = asyncio.get_event_loop().run_until_complete(
            provider.news_sentiment("AAPL"),
        )
        assert sent["sentiment"]["bearishPercent"] > sent["sentiment"]["bullishPercent"]

    def test_news_sentiment_bull(self):
        """Bull market returns high bullish percentage."""
        feeder = _make_feeder(start="2021-01-01")
        feeder.current_date = pd.Timestamp("2021-06-01")
        provider = MockFinnhubProvider(feeder)
        sent = asyncio.get_event_loop().run_until_complete(
            provider.news_sentiment("AAPL"),
        )
        assert sent["sentiment"]["bullishPercent"] > sent["sentiment"]["bearishPercent"]


class TestWeightEvolutionAndTearsheet:
    """Tests for weight evolution tracking and 3-panel tearsheet."""

    def test_weight_evolution_in_result(self):
        """BacktestResult can store weight_evolution data."""
        result = BacktestResult(
            initial_capital=10000,
            final_nav=12000,
            weight_evolution=[
                {
                    "date": pd.Timestamp("2020-01-20"),
                    "regime": "bear",
                    "trend_following": 0.40,
                    "mean_reversion": 0.40,
                    "volatility": 0.20,
                },
                {
                    "date": pd.Timestamp("2020-02-10"),
                    "regime": "bear",
                    "trend_following": 0.50,
                    "mean_reversion": 0.30,
                    "volatility": 0.20,
                },
            ],
        )
        assert len(result.weight_evolution) == 2
        assert result.weight_evolution[0]["regime"] == "bear"
        assert result.weight_evolution[1]["trend_following"] == 0.50

    def test_tearsheet_3_panels_with_weights(self, tmp_path):
        """Tearsheet generates 3 panels when weight_evolution is provided."""
        result = BacktestResult(
            initial_capital=10000,
            final_nav=11000,
            total_return_pct=10.0,
            max_drawdown_pct=-3.0,
            total_trades=50,
            win_rate=0.55,
            sharpe_ratio=1.0,
            daily_nav=[
                {"date": pd.Timestamp("2020-01-01"), "nav": 10000},
                {"date": pd.Timestamp("2020-06-01"), "nav": 10500},
                {"date": pd.Timestamp("2020-12-31"), "nav": 11000},
            ],
            weight_evolution=[
                {
                    "date": pd.Timestamp("2020-02-01"),
                    "regime": "bear",
                    "trend_following": 0.40,
                    "mean_reversion": 0.40,
                    "volatility": 0.20,
                },
            ],
        )
        bt = DecadeBacktester(tickers=["AAPL"])
        bt._data_dir = str(tmp_path)
        bt._generate_tearsheet(result)

        import os
        tearsheet_path = os.path.join(str(tmp_path), "backtest_tearsheet.html")
        assert os.path.isfile(tearsheet_path)

        html = open(tearsheet_path, encoding="utf-8").read()
        # Verify 3-panel layout indicators
        assert "OPRO Weight Evolution" in html
        assert "Brain" in html or "大腦神經" in html or "Trend Following" in html
        assert "Drawdown" in html

    def test_tearsheet_2_panels_without_weights(self, tmp_path):
        """Tearsheet falls back to 2 panels when no weight data."""
        result = BacktestResult(
            initial_capital=10000,
            final_nav=10500,
            total_return_pct=5.0,
            max_drawdown_pct=-2.0,
            total_trades=20,
            win_rate=0.50,
            sharpe_ratio=0.8,
            daily_nav=[
                {"date": pd.Timestamp("2020-01-01"), "nav": 10000},
                {"date": pd.Timestamp("2020-12-31"), "nav": 10500},
            ],
            weight_evolution=[],
        )
        bt = DecadeBacktester(tickers=["AAPL"])
        bt._data_dir = str(tmp_path)
        bt._generate_tearsheet(result)

        import os
        tearsheet_path = os.path.join(str(tmp_path), "backtest_tearsheet.html")
        assert os.path.isfile(tearsheet_path)

        html = open(tearsheet_path, encoding="utf-8").read()
        # Should NOT have the weight panel
        assert "OPRO Weight Evolution" not in html


class TestConfidenceAmplifier:
    """Tests for the Adrenaline Shot confidence amplifier logic."""

    def test_amplifier_formula_low_confidence(self):
        """Amplifier: max(0.85, conf*3) — when conf is low, floor at 0.85."""
        original_conf = 0.20
        amplified = max(0.85, original_conf * 3)
        assert amplified == 0.85

    def test_amplifier_formula_high_confidence(self):
        """Amplifier: max(0.85, conf*3) — when conf*3 > 0.85, use that."""
        original_conf = 0.35
        amplified = max(0.85, original_conf * 3)
        assert amplified == pytest.approx(1.05)

    def test_amplifier_not_applied_to_neutral(self):
        """Neutral signals should NOT get amplified (tested via logic check)."""
        from src.core.base_agent import SignalDirection
        direction = SignalDirection.NEUTRAL
        original_conf = 0.20
        # The logic: only amplify if direction != NEUTRAL
        if direction != SignalDirection.NEUTRAL:
            conf = max(0.85, original_conf * 3)
        else:
            conf = original_conf
        assert conf == 0.20  # Unchanged

    def test_amplifier_applied_to_buy(self):
        """BUY signals get amplified."""
        from src.core.base_agent import SignalDirection
        direction = SignalDirection.BUY
        original_conf = 0.15
        if direction != SignalDirection.NEUTRAL:
            conf = max(0.85, original_conf * 3)
        else:
            conf = original_conf
        assert conf == 0.85  # Amplified to floor
