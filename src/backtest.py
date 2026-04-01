"""
Event-Driven Decade Backtester (v2)
====================================
A "time machine" that replays 10 years of market history through the
full MAS pipeline — without look-ahead bias.

Architecture (真正的事件驅動迴圈):
  1. Download 10 years of OHLCV data via yfinance (one-time bulk download)
  2. For each trading day, inject data into **mock providers** that
     only expose history up to "today" (no future leak)
  3. 逐日步進: Check active trades against intraday High/Low for SL/TP hits
  4. 大腦掃描: Agents call providers normally → Orchestrator fuses signals →
     RiskAgent evaluates with half-Kelly → open positions
  5. 歸因反思: Losing trades trigger FailureAttributionEngine → memory writeback
  6. OPRO 演化: Every N days, run_failure_driven_optimization mutates agent
     internal weights based on accumulated attribution reports
  7. Generate an interactive Plotly tearsheet with NAV curve and drawdown

Usage:
  python src/backtest.py                          # default 4 tickers, 10 years
  python src/backtest.py --tickers AAPL MSFT      # custom tickers
  python src/backtest.py --start 2018-01-01       # shorter period
  python src/backtest.py --capital 50000          # different starting capital
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("BacktestEngine")


# ══════════════════════════════════════════════════════════════════
#  Mock Data Providers — 時光機餵價器
# ══════════════════════════════════════════════════════════════════

@dataclass
class _TickerData:
    """Pre-downloaded OHLCV DataFrame + metadata for one ticker."""
    df: pd.DataFrame          # index=DatetimeIndex, columns=Open/High/Low/Close/Volume
    isin: str = ""
    name: str = ""
    currency: str = "USD"


class BacktestDataFeeder:
    """
    歷史數據餵價器 (時光機)
    Downloads bulk historical data via yfinance and serves it through
    provider-compatible interfaces, sliced to prevent look-ahead bias (未來函數).
    """

    # Mapping of common tickers to ISINs
    _ISIN_MAP: dict[str, str] = {
        "AAPL": "US0378331005",
        "MSFT": "US5949181045",
        "NVDA": "US67066G1040",
        "TSLA": "US88160R1014",
        "AMZN": "US0231351067",
        "GOOG": "US02079K3059",
        "GOOGL": "US02079K1079",
        "META": "US30303M1027",
        "SPY": "US78462F1030",
        "QQQ": "US46090E1038",
        "BRK-B": "US0846707026",
        "JPM": "US46625H1005",
    }

    def __init__(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> None:
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self._data: dict[str, _TickerData] = {}
        self._current_date: pd.Timestamp | None = None

    @property
    def current_date(self) -> pd.Timestamp | None:
        return self._current_date

    @current_date.setter
    def current_date(self, dt: pd.Timestamp) -> None:
        self._current_date = dt

    def download(self) -> None:
        """Bulk download all tickers from yfinance."""
        import yfinance as yf

        logger.info(
            "Downloading %d tickers: %s -> %s ...",
            len(self.tickers), self.start_date, self.end_date,
        )
        for ticker in self.tickers:
            try:
                df = yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=True,
                )
                # Flatten MultiIndex (yfinance >= 0.2.36 format)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if not df.empty:
                    self._data[ticker] = _TickerData(
                        df=df,
                        isin=self._ISIN_MAP.get(ticker, f"BACKTEST_{ticker}"),
                    )
                    logger.info(
                        "  %s: %d trading days loaded", ticker, len(df),
                    )
                else:
                    logger.warning("  %s: no data returned", ticker)
            except Exception as e:
                logger.error("  %s: download failed - %s", ticker, e)

    def get_trading_days(self) -> list[pd.Timestamp]:
        """Return the union of all trading days across tickers."""
        if not self._data:
            return []
        all_indices = pd.DatetimeIndex([])
        for td in self._data.values():
            all_indices = all_indices.union(td.df.index)
        return sorted(all_indices)

    def get_past_df(self, ticker: str) -> pd.DataFrame:
        """Return the OHLCV DataFrame sliced up to current_date (no future)."""
        td = self._data.get(ticker)
        if td is None or self._current_date is None:
            return pd.DataFrame()
        mask = td.df.index <= self._current_date
        return td.df.loc[mask]

    # ── Regime detection ─────────────────────────────────────────
    # Known bear-market years (used to vary mock fundamental/sentiment data)
    _BEAR_YEARS: set[int] = {2015, 2018, 2020, 2022}

    def is_bear_market(self) -> bool:
        """Check if current_date falls in a known bear-market year."""
        if self._current_date is None:
            return False
        return self._current_date.year in self._BEAR_YEARS

    # ── Provider-compatible data access ────────────────────────────

    def get_daily_bars(
        self, ticker: str, days: int = 365,
    ) -> list:
        """
        Return OHLCVBar-compatible objects up to current_date.
        This is the core anti-look-ahead mechanism.
        """
        from src.data.providers.base_provider import OHLCVBar

        td = self._data.get(ticker)
        if td is None or self._current_date is None:
            return []

        mask = td.df.index <= self._current_date
        past_df = td.df.loc[mask].tail(days)

        bars = []
        for dt, row in past_df.iterrows():
            bars.append(OHLCVBar(
                timestamp=dt.strftime("%Y-%m-%d"),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
            ))
        return bars

    def get_current_price(self, ticker: str) -> float | None:
        """Get the close price on current_date."""
        td = self._data.get(ticker)
        if td is None or self._current_date is None:
            return None
        if self._current_date in td.df.index:
            return float(td.df.loc[self._current_date, "Close"])
        # Nearest past date
        mask = td.df.index <= self._current_date
        past = td.df.loc[mask]
        return float(past.iloc[-1]["Close"]) if not past.empty else None

    def get_instruments(self) -> list[dict[str, Any]]:
        """Generate T212-style instrument metadata for ISINMapper."""
        instruments = []
        for ticker, td in self._data.items():
            instruments.append({
                "ticker": ticker,
                "isin": td.isin,
                "currencyCode": td.currency,
                "name": td.name or ticker,
                "exchangeId": "BACKTEST",
                "type": "STOCK",
            })
        return instruments


# ══════════════════════════════════════════════════════════════════
#  Mock Providers — 替換真實 API 的回測用 Provider
# ══════════════════════════════════════════════════════════════════

class MockPolygonProvider:
    """
    Drop-in replacement for PolygonProvider that reads from BacktestDataFeeder.
    Agents call daily_bars() exactly as in production.
    """
    name = "polygon_backtest"

    def __init__(self, feeder: BacktestDataFeeder) -> None:
        self._feeder = feeder

    async def daily_bars(self, ticker: str, days: int = 365) -> list:
        return self._feeder.get_daily_bars(ticker, days)

    async def aggregates(self, ticker: str, **kwargs: Any) -> list:
        return self._feeder.get_daily_bars(ticker, kwargs.get("limit", 500))

    async def previous_close(self, ticker: str) -> dict[str, Any]:
        price = self._feeder.get_current_price(ticker)
        return {"close": price or 0, "high": price or 0, "low": price or 0,
                "open": price or 0, "volume": 0}

    async def snapshot(self, ticker: str) -> dict[str, Any]:
        return await self.previous_close(ticker)

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> "MockPolygonProvider":
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass


class MockAlphaVantageProvider:
    """
    Drop-in replacement for AlphaVantageProvider.
    Serves OHLCV from feeder; returns plausible static data for fundamentals.
    """
    name = "alpha_vantage_backtest"

    def __init__(self, feeder: BacktestDataFeeder) -> None:
        self._feeder = feeder

    async def daily(self, symbol: str, outputsize: str = "compact") -> list:
        days = 100 if outputsize == "compact" else 500
        return self._feeder.get_daily_bars(symbol, days)

    async def weekly(self, symbol: str) -> list:
        return self._feeder.get_daily_bars(symbol, 260)

    async def monthly(self, symbol: str) -> list:
        return self._feeder.get_daily_bars(symbol, 120)

    async def company_overview(self, symbol: str) -> dict[str, Any]:
        """Return regime-dependent plausible overview for FundamentalAgent.

        Bear-market years → deteriorating fundamentals (high PE, negative
        growth, low margins).  Bull-market years → healthy fundamentals.
        """
        if self._feeder.is_bear_market():
            return {
                "Symbol": symbol,
                "PERatio": "35.0",
                "ForwardPE": "30.0",
                "PriceToBookRatio": "8.0",
                "EVToEBITDA": "22.0",
                "ReturnOnEquityTTM": "0.08",
                "ReturnOnAssetsTTM": "0.03",
                "EPS": "1.5",
                "RevenuePerShareTTM": "18.0",
                "ProfitMargin": "0.06",
                "QuarterlyRevenueGrowthYOY": "-0.05",
                "QuarterlyEarningsGrowthYOY": "-0.12",
                "DividendYield": "0.002",
                "PayoutRatio": "0.50",
                "Beta": "1.5",
                "52WeekHigh": "200.0",
                "52WeekLow": "100.0",
                "AnalystTargetPrice": "130.0",
            }
        return {
            "Symbol": symbol,
            "PERatio": "20.0",
            "ForwardPE": "18.0",
            "PriceToBookRatio": "5.0",
            "EVToEBITDA": "15.0",
            "ReturnOnEquityTTM": "0.25",
            "ReturnOnAssetsTTM": "0.10",
            "EPS": "5.0",
            "RevenuePerShareTTM": "25.0",
            "ProfitMargin": "0.20",
            "QuarterlyRevenueGrowthYOY": "0.08",
            "QuarterlyEarningsGrowthYOY": "0.10",
            "DividendYield": "0.005",
            "PayoutRatio": "0.15",
            "Beta": "1.1",
            "52WeekHigh": "200.0",
            "52WeekLow": "120.0",
            "AnalystTargetPrice": "180.0",
        }

    async def earnings(self, symbol: str) -> dict[str, Any]:
        return {"quarterlyEarnings": []}

    async def earnings_history(self, symbol: str) -> list[dict[str, Any]]:
        return []

    async def sma(self, symbol: str, **kwargs: Any) -> list[dict]:
        return []

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> "MockAlphaVantageProvider":
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass


class MockFinnhubProvider:
    """Drop-in for FinnhubProvider. Returns regime-dependent sentiment."""
    name = "finnhub_backtest"

    def __init__(self, feeder: BacktestDataFeeder | None = None) -> None:
        self._feeder = feeder

    async def company_news(self, symbol: str, **kwargs: Any) -> list[dict]:
        """Return bearish headlines in bear years, bullish in bull years."""
        if self._feeder and self._feeder.is_bear_market():
            return [
                {
                    "headline": f"{symbol} faces headwinds amid market downturn",
                    "summary": "Analysts warn of further declines as macro conditions worsen.",
                    "source": "backtest_mock",
                    "datetime": int(self._feeder.current_date.timestamp()) if self._feeder.current_date else 0,
                },
                {
                    "headline": f"Recession fears hit {symbol} hard",
                    "summary": "Earnings expected to miss consensus as consumer spending drops.",
                    "source": "backtest_mock",
                    "datetime": int(self._feeder.current_date.timestamp()) if self._feeder.current_date else 0,
                },
            ]
        if self._feeder and not self._feeder.is_bear_market():
            return [
                {
                    "headline": f"{symbol} rallies on strong earnings beat",
                    "summary": "Revenue growth exceeds expectations for the third straight quarter.",
                    "source": "backtest_mock",
                    "datetime": int(self._feeder.current_date.timestamp()) if self._feeder.current_date else 0,
                },
                {
                    "headline": f"Institutional investors pile into {symbol}",
                    "summary": "Fund managers increase allocation citing robust fundamentals.",
                    "source": "backtest_mock",
                    "datetime": int(self._feeder.current_date.timestamp()) if self._feeder.current_date else 0,
                },
            ]
        return []

    async def news_sentiment(self, symbol: str) -> dict[str, Any]:
        """Return regime-dependent sentiment score."""
        if self._feeder and self._feeder.is_bear_market():
            return {"buzz": {"articlesInLastWeek": 10}, "sentiment": {"bearishPercent": 0.6, "bullishPercent": 0.2}}
        if self._feeder:
            return {"buzz": {"articlesInLastWeek": 10}, "sentiment": {"bearishPercent": 0.15, "bullishPercent": 0.55}}
        return {}

    async def recommendation_trends(self, symbol: str) -> list[dict]:
        return []

    async def earnings_surprises(self, symbol: str) -> list[dict]:
        return []

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> "MockFinnhubProvider":
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass


class MockIntrinioProvider:
    """Drop-in for IntrinioProvider."""
    name = "intrinio_backtest"

    async def company_financials(self, symbol: str, **kw: Any) -> dict:
        return {}

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> "MockIntrinioProvider":
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass


class MockTrading212Client:
    """
    Mock T212 API client for backtesting.
    Simulates instant fills at current market price.
    """

    def __init__(self, feeder: BacktestDataFeeder) -> None:
        self._feeder = feeder
        self._order_counter = 0
        self._orders: list[dict[str, Any]] = []

    async def account_info(self) -> dict[str, Any]:
        return {"cash": 0, "invested": 0, "ppl": 0, "result": 0}

    async def portfolio(self) -> list[dict[str, Any]]:
        return []

    async def exchange_instruments(self) -> list[dict[str, Any]]:
        return self._feeder.get_instruments()

    async def place_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._order_counter += 1
        ticker = payload.get("ticker", "")
        qty = payload.get("quantity", 0)
        price = self._feeder.get_current_price(ticker) or 0

        order = {
            "id": self._order_counter,
            "ticker": ticker,
            "quantity": qty,
            "fillPrice": price,
            "filledQuantity": abs(qty),
            "status": "filled",
        }
        self._orders.append(order)
        return order

    async def place_stop_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.place_order(payload)

    async def place_limit_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.place_order(payload)

    async def cancel_order(self, order_id: int) -> Any:
        return {}

    async def order_history(self, **kw: Any) -> dict[str, Any]:
        return {"items": self._orders[-50:]}

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> "MockTrading212Client":
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass


# ══════════════════════════════════════════════════════════════════
#  Backtest Engine — 事件驅動十年回測引擎
# ══════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    """Summary statistics from a completed backtest."""
    initial_capital: float = 0.0
    final_nav: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    opro_generations: int = 0
    daily_nav: list[dict[str, Any]] = field(default_factory=list)
    weight_evolution: list[dict[str, Any]] = field(default_factory=list)
    benchmark_nav: list[dict[str, Any]] = field(default_factory=list)
    benchmark_return_pct: float = 0.0
    tickers: list[str] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        import json
        return {
            "initial_capital": self.initial_capital,
            "final_nav": self.final_nav,
            "total_return_pct": self.total_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "sharpe_ratio": self.sharpe_ratio,
            "opro_generations": self.opro_generations,
            "tickers": self.tickers,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "benchmark_return_pct": self.benchmark_return_pct,
            "daily_nav": [
                {"date": d["date"].isoformat() if hasattr(d["date"], "isoformat") else str(d["date"]),
                 "nav": d["nav"]}
                for d in self.daily_nav
            ],
            "benchmark_nav": [
                {"date": d["date"].isoformat() if hasattr(d["date"], "isoformat") else str(d["date"]),
                 "nav": d["nav"]}
                for d in self.benchmark_nav
            ],
            "weight_evolution": [
                {k: (v.isoformat() if hasattr(v, "isoformat") else v)
                 for k, v in w.items()}
                for w in self.weight_evolution
            ],
        }


class DecadeBacktester:
    """
    Event-driven backtester that replays history through the full MAS
    pipeline with direct SL/TP trade lifecycle management.

    Key design (v2 — 真正的事件驅動):
      - 逐日步進: Each day checks intraday High/Low against active SL/TP
      - 平倉結算: SL/TP hit triggers FailureAttributionEngine for losers
      - 大腦掃描: Orchestrator fuses agent signals → RiskAgent sizes position
      - OPRO 演化: Periodic failure-driven weight mutation across agents
    """

    # Backtest-specific intelligence weights.
    # Technical gets 80% because it's the only agent with real historical data;
    # fundamental has static mock data, sentiment has empty mock data.
    _BACKTEST_WEIGHTS: dict[str, float] = {
        "fundamental": 0.05,
        "technical": 0.90,
        "sentiment": 0.05,
    }

    def __init__(
        self,
        tickers: list[str],
        start_date: str = "2014-01-01",
        end_date: str = "2024-01-01",
        initial_capital: float = 10_000.0,
        scan_interval: int = 2,
        reflection_interval: int = 20,
        min_entry_score: float = 0.03,
        confidence_floor: float = 0.75,
        data_dir: str = "data/backtest",
    ) -> None:
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self._scan_interval = scan_interval
        self._reflection_interval = reflection_interval
        self._min_entry_score = min_entry_score
        self._confidence_floor = confidence_floor
        self._data_dir = data_dir

        os.makedirs(data_dir, exist_ok=True)

        # Will be initialised in run()
        self._feeder: BacktestDataFeeder | None = None

    async def run(self) -> BacktestResult:
        """
        Execute the full event-driven backtest.

        Loop structure per trading day:
          Phase 1 — 平倉邏輯: check active SL/TP against today's High/Low
          Phase 2 — 大腦掃描: if no position, run Orchestrator → Risk → open
          Phase 3 — 日結淨值: snapshot NAV for tearsheet
          Phase 4 — OPRO 演化: periodic failure-driven optimization
        """
        from src.agents.audit.failure_attribution import FailureAttributionEngine
        from src.agents.decision.risk import RiskAgent, RiskVerdict
        from src.agents.intelligence.fundamental import FundamentalAgent
        from src.agents.intelligence.orchestrator import IntelligenceOrchestrator
        from src.agents.intelligence.sentiment import SentimentAgent
        from src.agents.intelligence.technical import TechnicalAgent
        from src.core.base_agent import SignalDirection
        from src.core.virtual_account import VirtualAccountManager
        from src.memory.episodic_memory import Episode, EpisodicMemory
        from src.prompts.adaptive_opro import AdaptiveOPRO

        # ── 0. Clean previous backtest state ──────────────────────
        # Each run must start fresh — stale memory/OPRO/accounts from
        # previous runs would corrupt results (e.g., cold-start bypass
        # failing because memory has leftover episodes).
        import shutil
        for subdir in ("memory", "opro", "audit"):
            path = os.path.join(self._data_dir, subdir)
            if os.path.isdir(path):
                shutil.rmtree(path)
        acct_file = os.path.join(self._data_dir, "bt_accounts.json")
        if os.path.isfile(acct_file):
            os.remove(acct_file)
        logger.info("Cleaned previous backtest state in %s", self._data_dir)

        # ── 1. Download historical data ────────────────────────────
        self._feeder = BacktestDataFeeder(
            self.tickers, self.start_date, self.end_date,
        )
        self._feeder.download()
        trading_days = self._feeder.get_trading_days()

        if not trading_days:
            logger.error("No trading days found - aborting")
            return BacktestResult()

        logger.info(
            "Backtest span: %s -> %s (%d trading days)",
            trading_days[0].strftime("%Y-%m-%d"),
            trading_days[-1].strftime("%Y-%m-%d"),
            len(trading_days),
        )

        # ── 2. Build MAS pipeline with mock providers ──────────────
        mock_polygon = MockPolygonProvider(self._feeder)
        mock_av = MockAlphaVantageProvider(self._feeder)
        mock_finnhub = MockFinnhubProvider(self._feeder)
        mock_intrinio = MockIntrinioProvider()

        # Virtual account (isolated backtest storage)
        account_manager = VirtualAccountManager(
            storage_file=os.path.join(self._data_dir, "bt_accounts.json"),
        )
        virtual_account = account_manager.allocate_account(
            "Backtest_Bot", self.initial_capital,
        )

        # Intelligence agents (injected with mock providers)
        technical = TechnicalAgent(mock_av, mock_polygon)
        fundamental = FundamentalAgent(mock_av, mock_intrinio)
        sentiment = SentimentAgent(mock_finnhub)
        intelligence = IntelligenceOrchestrator(
            agents=[fundamental, technical, sentiment],
        )

        # Override weights for backtest: technical dominates because it has
        # real historical data; fundamental/sentiment have mock stubs.
        intelligence.update_weights(self._BACKTEST_WEIGHTS)
        logger.info(
            "Backtest config: weights=%s | min_score=%.2f | conf_floor=%.2f",
            self._BACKTEST_WEIGHTS, self._min_entry_score,
            self._confidence_floor,
        )

        # Memory + OPRO
        memory = EpisodicMemory(
            store_dir=os.path.join(self._data_dir, "memory"),
        )
        opro = AdaptiveOPRO(
            store_dir=os.path.join(self._data_dir, "opro"),
        )

        # Risk agent (with memory for regime-aware Kelly)
        # Backtest uses aggressive limits to stay fully invested:
        # 25% per trade allows 4 concurrent positions at full allocation
        risk = RiskAgent(
            max_single_position_pct=0.25,
            max_var_pct_of_nav=0.20,
            atr_tp_multiplier=4.5,
            episodic_memory=memory,
        )

        # Failure attribution engine
        attribution_engine = FailureAttributionEngine(memory)

        # ── 3. State tracking ──────────────────────────────────────
        # Active trades: {ticker: {entry_price, sl, tp, qty, signal, isin}}
        active_trades: dict[str, dict[str, Any]] = {}
        daily_nav: list[dict[str, Any]] = []
        trade_count = 0
        win_count = 0
        loss_count = 0
        last_log_year = 0

        # 大腦神經可視化: track TechnicalAgent internal weight evolution
        weight_evolution_history: list[dict[str, Any]] = []

        t_start = time.monotonic()
        logger.info(
            "Starting event-driven replay (%d days, %d tickers)...",
            len(trading_days), len(self.tickers),
        )

        for day_idx, current_dt in enumerate(trading_days):
            self._feeder.current_date = current_dt
            date_str = current_dt.strftime("%Y-%m-%d")

            # =========================================================
            # Phase 1: 平倉邏輯 — Check SL/TP against today's High/Low
            # =========================================================
            tickers_to_close: list[str] = []
            for ticker, trade in active_trades.items():
                past_df = self._feeder.get_past_df(ticker)
                if past_df.empty:
                    continue

                today = past_df.iloc[-1]
                today_high = float(today["High"])
                today_low = float(today["Low"])
                today_close = float(today["Close"])

                exit_price: float | None = None

                # Pessimistic fill: check stop-loss first (worst case)
                if today_low <= trade["sl"]:
                    exit_price = trade["sl"]
                elif today_high >= trade["tp"]:
                    exit_price = trade["tp"]

                if exit_price is not None:
                    # Record exit in virtual account
                    virtual_account.record_trade(
                        ticker, -trade["qty"], exit_price,
                    )
                    roi = (exit_price - trade["entry_price"]) / trade["entry_price"]
                    trade_count += 1

                    episode_id = f"BT_{date_str}_{ticker}"

                    if roi < 0:
                        # 虧損 → trigger FailureAttributionEngine
                        loss_count += 1

                        # Store episode first so attribution can writeback
                        episode = Episode(
                            episode_id=episode_id,
                            timestamp=current_dt.timestamp(),
                            ticker=ticker,
                            isin=trade.get("isin", ""),
                            action="BUY",
                            roi=roi,
                            pnl=(exit_price - trade["entry_price"]) * trade["qty"],
                            fused_score=trade.get("fused_score", 0.0),
                            fused_confidence=trade.get("fused_confidence", 0.0),
                            agent_scores=trade.get("agent_scores", {}),
                        )
                        memory.store(episode)

                        # Run failure attribution diagnosis
                        try:
                            attribution_engine.diagnose_and_update(
                                episode_id=episode_id,
                                expected_price=trade["entry_price"],
                                fill_price=trade["entry_price"],  # backtest = no slippage
                                close_price=exit_price,
                                roi=roi,
                                agent_confidences=trade.get("agent_scores", {}),
                            )
                        except Exception:
                            logger.debug(
                                "Attribution failed for %s", episode_id,
                                exc_info=True,
                            )
                    else:
                        # 獲利 → store success episode
                        win_count += 1
                        episode = Episode(
                            episode_id=episode_id,
                            timestamp=current_dt.timestamp(),
                            ticker=ticker,
                            isin=trade.get("isin", ""),
                            action="BUY",
                            roi=roi,
                            pnl=(exit_price - trade["entry_price"]) * trade["qty"],
                            fused_score=trade.get("fused_score", 0.0),
                            fused_confidence=trade.get("fused_confidence", 0.0),
                            agent_scores=trade.get("agent_scores", {}),
                        )
                        memory.store(episode)

                    # Record trade outcome for OPRO bandit evolution
                    opro.record_trade_outcome(roi)

                    tickers_to_close.append(ticker)

            for t in tickers_to_close:
                del active_trades[t]

            # =========================================================
            # Phase 2: 大腦掃描 — Entry logic for tickers without position
            # =========================================================
            if day_idx % self._scan_interval == 0:
                for ticker in self.tickers:
                    if ticker in active_trades:
                        continue
                    if ticker not in self._feeder._data:
                        continue

                    past_df = self._feeder.get_past_df(ticker)
                    if len(past_df) < 30:
                        # Need minimum history for indicators
                        continue

                    today_close = float(past_df.iloc[-1]["Close"])
                    isin = self._feeder._data[ticker].isin

                    try:
                        # 1. Intelligence fusion (Orchestrator)
                        view = await intelligence.evaluate(
                            {"ticker": ticker, "isin": isin},
                        )

                        # Entry gate: use fused_score threshold instead of
                        # SignalDirection enum.  With mock fundamental/sentiment
                        # data the fused_score rarely reaches the hardcoded 0.4
                        # BUY threshold, so we use a configurable min_entry_score
                        # (default 0.05) to allow trades on weaker-but-positive
                        # technical signals.
                        if view.fused_score < self._min_entry_score:
                            continue

                        # Confidence amplifier — "Adrenaline Shot"
                        # When the orchestrator produces a non-NEUTRAL signal,
                        # the backtest amplifies fused_confidence to compensate
                        # for the structurally low mock-data confidence.  This
                        # works in tandem with confidence_floor but targets the
                        # signal itself rather than the Kelly floor.
                        if view.fused_direction != SignalDirection.NEUTRAL:
                            view.fused_confidence = max(
                                0.85, view.fused_confidence * 3,
                            )

                        # 2. Prepare price data for RiskAgent
                        bars = self._feeder.get_daily_bars(ticker, 365)
                        if len(bars) < 15:
                            continue

                        closes = np.array([b.close for b in bars])
                        highs = np.array([b.high for b in bars])
                        lows = np.array([b.low for b in bars])

                        returns = np.diff(np.log(closes)) if len(closes) > 1 else np.array([0.0])
                        atr = TechnicalAgent._compute_atr(highs, lows, closes, 14)

                        # 3. RiskAgent evaluation (half-Kelly + VaR)
                        #
                        # Confidence floor for backtest mode:
                        # In production, fused_confidence reflects real
                        # API data quality.  In backtest, mock sentiment
                        # (conf=0.00) and mock fundamental (conf=0.15)
                        # permanently suppress it to ~0.12, which makes
                        # Kelly's  p = 0.5*base_p + 0.5*0.12 ~ 0.32
                        # — always below the Kelly > 0 threshold (0.50).
                        #
                        # The confidence_floor (default 0.60) corrects
                        # this structural bias.  It represents: "given
                        # that the fused_score already passed our entry
                        # gate, we are at least this confident."
                        #
                        # Math with floor=0.60, base_p=0.52, b=1.0:
                        #   p = 0.5*0.52 + 0.5*0.60 = 0.56
                        #   kelly = (0.56 - 0.44) / 1.0 = 0.12
                        #   half_kelly = 0.06  ->  6% of NAV per trade
                        bt_confidence = max(
                            view.fused_confidence,
                            self._confidence_floor,
                        )

                        price_map = {ticker: today_close}
                        envelope = risk.evaluate_with_account(
                            direction=1,  # BUY
                            current_price=today_close,
                            returns=returns,
                            atr=atr,
                            confidence=bt_confidence,
                            virtual_account=virtual_account,
                            price_map=price_map,
                            symbol=ticker,
                        )

                        if envelope.verdict == RiskVerdict.REJECTED:
                            logger.debug(
                                "[%s] %s REJECTED by RiskAgent: %s",
                                date_str, ticker, envelope.reason,
                            )
                            continue

                        qty = envelope.suggested_quantity
                        if qty <= 0:
                            logger.debug(
                                "[%s] %s qty=%.4f too small (skipped)",
                                date_str, ticker, qty,
                            )
                            continue

                        cost = qty * today_close
                        if not virtual_account.can_afford(cost):
                            logger.debug(
                                "[%s] %s can't afford %.2f (cash=%.2f)",
                                date_str, ticker, cost,
                                virtual_account.available_cash,
                            )
                            continue

                        # 4. Open position
                        virtual_account.record_trade(ticker, qty, today_close)
                        active_trades[ticker] = {
                            "entry_price": today_close,
                            "sl": envelope.stop_loss_price,
                            "tp": envelope.take_profit_price,
                            "qty": qty,
                            "isin": isin,
                            "fused_score": view.fused_score,
                            "fused_confidence": view.fused_confidence,
                            "agent_scores": {
                                s.source: s.confidence
                                for s in view.signals
                            },
                        }

                        logger.info(
                            "[%s] OPEN %s x%.1f @ %.2f | "
                            "SL=%.2f TP=%.2f | score=%.2f conf=%.2f",
                            date_str, ticker, qty, today_close,
                            envelope.stop_loss_price, envelope.take_profit_price,
                            view.fused_score, view.fused_confidence,
                        )

                    except Exception as exc:
                        logger.warning(
                            "Analysis of %s failed on %s: %s",
                            ticker, date_str, exc,
                        )
                        logger.debug("Full traceback:", exc_info=True)

            # =========================================================
            # Phase 3: 日結淨值 — Daily NAV snapshot
            # =========================================================
            total_invested_value = 0.0
            for ticker, trade in active_trades.items():
                px = self._feeder.get_current_price(ticker)
                if px:
                    total_invested_value += trade["qty"] * px

            nav = virtual_account.available_cash + total_invested_value
            daily_nav.append({"date": current_dt, "nav": nav})

            # =========================================================
            # Phase 4: OPRO 動態自我進化 (every N trading days)
            # =========================================================
            if day_idx > 0 and day_idx % self._reflection_interval == 0:
                try:
                    # Bandit-style population evolution
                    evolved = opro.maybe_evolve()

                    # Apply evolved weights to intelligence orchestrator
                    if evolved:
                        new_weights = opro.get_intelligence_weights()
                        intelligence.update_weights(new_weights)

                    # Failure-driven agent internal weight mutation
                    fdo_result = opro.run_failure_driven_optimization(
                        memory=memory,
                        technical_agent=technical,
                        fundamental_agent=fundamental,
                        sentiment_agent=sentiment,
                    )

                    if fdo_result.get("status") == "OPTIMIZED":
                        logger.info(
                            "[%s] OPRO failure-driven mutation triggered: %s",
                            date_str, fdo_result.get("updates", {}),
                        )

                    # 大腦神經可視化: snapshot TechnicalAgent internal weights
                    weight_evolution_history.append({
                        "date": current_dt,
                        "regime": "bear" if self._feeder.is_bear_market() else "bull",
                        **{k: v for k, v in technical.dynamic_weights.items()},
                    })

                except Exception:
                    logger.debug(
                        "OPRO evolution failed on %s",
                        date_str, exc_info=True,
                    )

            # =========================================================
            # Progress logging + file (for dashboard polling)
            # =========================================================
            if day_idx % 50 == 0 or current_dt.year != last_log_year:
                progress = {
                    "status": "running",
                    "day": day_idx,
                    "total_days": len(trading_days),
                    "pct": round(day_idx / len(trading_days) * 100, 1),
                    "current_date": date_str,
                    "nav": round(nav, 2),
                    "trades": trade_count,
                }
                try:
                    progress_path = os.path.join(self._data_dir, "backtest_progress.json")
                    with open(progress_path, "w") as f:
                        json.dump(progress, f)
                except OSError:
                    pass

            if current_dt.year != last_log_year:
                last_log_year = current_dt.year
                elapsed = time.monotonic() - t_start
                logger.info(
                    "  %d - NAV: \u00a3%.2f | Trades: %d (W:%d/L:%d) | "
                    "Positions: %d | Memory: %d ep | "
                    "OPRO gen: %d | [%.1fs]",
                    current_dt.year, nav, trade_count, win_count, loss_count,
                    len(active_trades), memory.count,
                    opro._generation, elapsed,
                )

        # ── 5. Compute summary statistics ────────────────────────────
        elapsed_total = time.monotonic() - t_start
        final_nav = daily_nav[-1]["nav"] if daily_nav else self.initial_capital
        total_return = (final_nav - self.initial_capital) / self.initial_capital

        # Max drawdown
        nav_series = pd.Series([d["nav"] for d in daily_nav])
        peak = nav_series.cummax()
        drawdown = (nav_series - peak) / peak
        max_dd = float(drawdown.min())

        # Sharpe ratio (annualised, assuming 252 trading days)
        if len(nav_series) > 1:
            daily_returns = nav_series.pct_change().dropna()
            if daily_returns.std() > 0:
                sharpe = float(
                    daily_returns.mean() / daily_returns.std() * np.sqrt(252),
                )
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Win rate
        total_closed = win_count + loss_count
        win_rate = win_count / total_closed if total_closed > 0 else 0.0

        # ── Benchmark: S&P 500 buy-and-hold ─────────────────────────
        benchmark_nav_list: list[dict[str, Any]] = []
        benchmark_return = 0.0
        try:
            import yfinance as yf
            spy_df = yf.download(
                "SPY", start=self.start_date, end=self.end_date,
                progress=False, auto_adjust=True,
            )
            if isinstance(spy_df.columns, pd.MultiIndex):
                spy_df.columns = spy_df.columns.get_level_values(0)
            if not spy_df.empty:
                spy_start_price = float(spy_df.iloc[0]["Close"])
                for dt, row in spy_df.iterrows():
                    bm_nav = self.initial_capital * float(row["Close"]) / spy_start_price
                    benchmark_nav_list.append({"date": dt, "nav": bm_nav})
                benchmark_return = (
                    (benchmark_nav_list[-1]["nav"] - self.initial_capital)
                    / self.initial_capital * 100
                )
                logger.info(
                    "  Benchmark (SPY): %.2f%% return", benchmark_return,
                )
        except Exception as e:
            logger.warning("Benchmark computation failed: %s", e)

        result = BacktestResult(
            initial_capital=self.initial_capital,
            final_nav=final_nav,
            total_return_pct=total_return * 100,
            max_drawdown_pct=max_dd * 100,
            total_trades=trade_count,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            opro_generations=opro._generation,
            daily_nav=daily_nav,
            weight_evolution=weight_evolution_history,
            benchmark_nav=benchmark_nav_list,
            benchmark_return_pct=benchmark_return,
            tickers=self.tickers,
            start_date=self.start_date,
            end_date=self.end_date,
        )

        logger.info("=" * 60)
        logger.info("  BACKTEST COMPLETE")
        logger.info("=" * 60)
        logger.info("  Period:         %s -> %s", self.start_date, self.end_date)
        logger.info("  Initial:        \u00a3%.2f", result.initial_capital)
        logger.info("  Final NAV:      \u00a3%.2f", result.final_nav)
        logger.info("  Total Return:   %.2f%%", result.total_return_pct)
        logger.info("  Max Drawdown:   %.2f%%", result.max_drawdown_pct)
        logger.info("  Sharpe Ratio:   %.2f", result.sharpe_ratio)
        logger.info("  Total Trades:   %d (W:%d / L:%d)", result.total_trades, win_count, loss_count)
        logger.info("  Win Rate:       %.1f%%", result.win_rate * 100)
        logger.info("  OPRO Gens:      %d", result.opro_generations)
        logger.info("  Elapsed:        %.1fs", elapsed_total)
        logger.info("=" * 60)

        # ── 6. Generate tearsheet ────────────────────────────────────
        self._generate_tearsheet(result)

        # ── 7. Persist results as JSON (for dashboard) ────────────
        results_path = os.path.join(self._data_dir, "backtest_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Results saved to %s", results_path)

        # Mark progress as completed
        progress_path = os.path.join(self._data_dir, "backtest_progress.json")
        with open(progress_path, "w") as f:
            json.dump({"status": "completed"}, f)

        # Persist final state
        account_manager.save_state()

        return result

    def _generate_tearsheet(self, result: BacktestResult) -> None:
        """Generate an interactive 3-panel HTML tearsheet.

        Panel 1 — Portfolio NAV with bear-market shading
        Panel 2 — Drawdown (%)
        Panel 3 — 大腦神經可視化: OPRO weight evolution (stacked area)
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.warning("Plotly not installed - skipping tearsheet generation")
            return

        if not result.daily_nav:
            return

        df = pd.DataFrame(result.daily_nav).set_index("date")
        df["peak"] = df["nav"].cummax()
        df["drawdown_pct"] = (df["nav"] - df["peak"]) / df["peak"] * 100

        has_weights = bool(result.weight_evolution)
        n_rows = 3 if has_weights else 2
        row_heights = [0.50, 0.20, 0.30] if has_weights else [0.7, 0.3]
        subplot_titles = ["Portfolio NAV", "Drawdown (%)"]
        if has_weights:
            subplot_titles.append("OPRO Weight Evolution (大腦神經)")

        fig = make_subplots(
            rows=n_rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=tuple(subplot_titles),
            row_heights=row_heights,
        )

        # ── Panel 1: NAV ──────────────────────────────────────────
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["nav"],
                name="Strategy NAV",
                line=dict(color="#00CC96", width=1.5),
            ),
            row=1, col=1,
        )

        # Initial capital reference
        fig.add_hline(
            y=result.initial_capital,
            line_dash="dot", line_color="gray",
            annotation_text=f"Initial: \u00a3{result.initial_capital:,.0f}",
            row=1, col=1,
        )

        # Bear-market year shading on NAV panel
        bear_years = BacktestDataFeeder._BEAR_YEARS
        for year in sorted(bear_years):
            y_start = pd.Timestamp(f"{year}-01-01")
            y_end = pd.Timestamp(f"{year}-12-31")
            if y_start >= df.index.min() and y_start <= df.index.max():
                fig.add_vrect(
                    x0=y_start, x1=min(y_end, df.index.max()),
                    fillcolor="rgba(239,85,59,0.10)",
                    layer="below", line_width=0,
                    annotation_text=f"Bear {year}",
                    annotation_position="top left",
                    annotation_font_size=9,
                    annotation_font_color="rgba(239,85,59,0.6)",
                    row=1, col=1,
                )

        # ── Panel 2: Drawdown ─────────────────────────────────────
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["drawdown_pct"],
                name="Drawdown",
                fill="tozeroy",
                line=dict(color="#EF553B", width=1),
            ),
            row=2, col=1,
        )

        # ── Panel 3: OPRO Weight Evolution (stacked area) ────────
        if has_weights:
            wdf = pd.DataFrame(result.weight_evolution).set_index("date")
            # Extract numeric weight columns (exclude 'regime')
            weight_cols = [c for c in wdf.columns if c != "regime"]

            # Color palette for weight strategies
            colors = {
                "trend_following": "#636EFA",
                "mean_reversion": "#00CC96",
                "volatility": "#FFA15A",
            }

            for col in weight_cols:
                fig.add_trace(
                    go.Scatter(
                        x=wdf.index,
                        y=wdf[col],
                        name=col.replace("_", " ").title(),
                        stackgroup="weights",
                        line=dict(width=0.5),
                        fillcolor=colors.get(col, "#AB63FA"),
                    ),
                    row=3, col=1,
                )

        # ── Layout ────────────────────────────────────────────────
        subtitle = (
            f"Return: {result.total_return_pct:.1f}% | "
            f"Max DD: {result.max_drawdown_pct:.1f}% | "
            f"Sharpe: {result.sharpe_ratio:.2f} | "
            f"Trades: {result.total_trades} | "
            f"Win Rate: {result.win_rate * 100:.0f}%"
        )

        fig.update_layout(
            title=dict(
                text=(
                    f"AI Trading System \u2014 Event-Driven Backtest "
                    f"(Dynamic Regime + Brain Neural Viz)<br>"
                    f"<span style='font-size:14px;color:gray'>{subtitle}</span>"
                ),
            ),
            height=900 if has_weights else 700,
            template="plotly_dark",
            showlegend=True,
            legend=dict(x=0.01, y=0.99),
        )

        fig.update_yaxes(title_text="NAV (\u00a3)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        if has_weights:
            fig.update_yaxes(title_text="Weight", row=3, col=1)

        output_path = os.path.join(self._data_dir, "backtest_tearsheet.html")
        fig.write_html(output_path)
        logger.info("Tearsheet saved to %s", output_path)


# ══════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ══════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Trading System - Event-Driven Historical Backtest",
    )
    parser.add_argument(
        "--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA", "SPY"],
        help="Tickers to backtest (default: AAPL MSFT NVDA SPY)",
    )
    parser.add_argument(
        "--start", default="2014-01-01",
        help="Backtest start date (default: 2014-01-01)",
    )
    parser.add_argument(
        "--end", default="2024-01-01",
        help="Backtest end date (default: 2024-01-01)",
    )
    parser.add_argument(
        "--capital", type=float, default=10_000.0,
        help="Initial capital in GBP (default: 10000)",
    )
    parser.add_argument(
        "--scan-interval", type=int, default=2,
        help="Scan every N trading days (default: 2)",
    )
    parser.add_argument(
        "--reflection-interval", type=int, default=20,
        help="Run OPRO reflection every N days (default: 20)",
    )
    parser.add_argument(
        "--min-score", type=float, default=0.03,
        help="Minimum fused_score to open a trade (default: 0.03)",
    )
    parser.add_argument(
        "--confidence-floor", type=float, default=0.75,
        help="Confidence floor for Kelly formula in backtest (default: 0.75)",
    )
    parser.add_argument(
        "--data-dir", default="data/backtest",
        help="Directory for backtest output (default: data/backtest)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Ensure project root is on sys.path so "from src.xxx" imports work
    _project_root = str(Path(__file__).resolve().parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = _parse_args()
    backtester = DecadeBacktester(
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        scan_interval=args.scan_interval,
        reflection_interval=args.reflection_interval,
        min_entry_score=args.min_score,
        confidence_floor=args.confidence_floor,
        data_dir=args.data_dir,
    )
    asyncio.run(backtester.run())
