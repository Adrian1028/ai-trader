"""
Minimal Event-Driven Backtester — Phase 1 Alpha Verification
=============================================================

Completely independent of the existing MAS system.
No look-ahead bias. Conservative cost model. Randomisation test.

Usage:
    python -m backtest.minimal_backtest
    python -m backtest.minimal_backtest --tickers AAPL MSFT GOOGL AMZN NVDA
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.agents.technical_agent_minimal import compute_signal as tech_signal
from backtest.agents.fundamental_agent_minimal import (
    compute_signal as fund_signal,
    clear_cache as clear_fund_cache,
)

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

# Data splits
TRAIN_START, TRAIN_END = "2014-01-01", "2020-12-31"
VALID_START, VALID_END = "2021-01-01", "2022-12-31"
TEST_START, TEST_END = "2023-01-01", "2024-12-31"

INITIAL_CAPITAL = 100_000.0

# Signal thresholds
BUY_THRESHOLD = 0.3
SELL_THRESHOLD = -0.2

# Portfolio constraints
POSITION_SIZE_PCT = 0.10  # 10% per position
MAX_POSITIONS = 5

# Cost model
COMMISSION_PCT = 0.0  # Trading 212 = zero commission on shares
SLIPPAGE_BPS = 5  # 0.05% for large-cap
SPREAD_BPS = 2  # ~0.02% for highly liquid names

# Signal fusion weights
TECH_WEIGHT = 0.6
FUND_WEIGHT = 0.4

# Randomisation test
N_RANDOM_SHUFFLES = 1000


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class Position:
    ticker: str
    shares: float
    entry_price: float
    entry_date: str


@dataclass
class Trade:
    ticker: str
    action: Literal["BUY", "SELL"]
    shares: float
    price: float
    cost: float  # transaction cost
    date: str
    signal: float


@dataclass
class DailySnapshot:
    date: str
    nav: float
    cash: float
    positions: dict  # ticker -> market value
    benchmark_nav: float


@dataclass
class BacktestResult:
    period: str
    start_date: str
    end_date: str
    initial_capital: float
    final_nav: float
    total_return_pct: float
    annualised_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    win_rate_pct: float
    avg_win_loss_ratio: float
    total_trades: int
    avg_holding_days: float
    monthly_win_rate_pct: float
    benchmark_return_pct: float
    benchmark_sharpe: float
    excess_return_pct: float
    daily_returns: list = field(default_factory=list, repr=False)
    monthly_returns: list = field(default_factory=list, repr=False)
    trades: list = field(default_factory=list, repr=False)
    nav_series: list = field(default_factory=list, repr=False)


# ─── Data fetching ────────────────────────────────────────────────────────────

def download_data(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Download OHLCV data via yfinance. Returns {ticker: DataFrame}."""
    import yfinance as yf

    logger.info("Downloading data for %s from %s to %s", tickers, start, end)
    data = {}
    # Add buffer for indicator warm-up (250 trading days ≈ 1 year)
    buffer_start = (pd.Timestamp(start) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")

    for ticker in tickers:
        df = yf.download(ticker, start=buffer_start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            logger.warning("No data for %s", ticker)
            continue
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[ticker] = df
        logger.info("  %s: %d rows (%s to %s)", ticker, len(df),
                     df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d"))

    return data


def download_benchmark(start: str, end: str) -> pd.DataFrame:
    """Download SPY as benchmark."""
    import yfinance as yf

    buffer_start = (pd.Timestamp(start) - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    df = yf.download("SPY", start=buffer_start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ─── Cost model ───────────────────────────────────────────────────────────────

def transaction_cost(price: float, shares: float, is_large_cap: bool = True) -> float:
    """Calculate total transaction cost (commission + slippage + spread)."""
    notional = abs(price * shares)
    slippage = notional * (SLIPPAGE_BPS / 10_000)
    spread = notional * (SPREAD_BPS / 10_000)
    commission = notional * COMMISSION_PCT
    return slippage + spread + commission


# ─── Core backtest engine ─────────────────────────────────────────────────────

class MinimalBacktester:
    """Event-driven backtester with no look-ahead bias."""

    def __init__(
        self,
        tickers: list[str],
        data: dict[str, pd.DataFrame],
        benchmark: pd.DataFrame,
        capital: float = INITIAL_CAPITAL,
    ):
        self.tickers = tickers
        self.data = data
        self.benchmark = benchmark
        self.initial_capital = capital
        self.cash = capital
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.daily_snapshots: list[DailySnapshot] = []

    def _get_history_up_to(self, ticker: str, date: pd.Timestamp) -> pd.DataFrame:
        """Return historical data up to (inclusive) the given date — no future leak."""
        df = self.data[ticker]
        return df.loc[df.index <= date].copy()

    def _current_price(self, ticker: str, date: pd.Timestamp) -> float | None:
        """Get closing price on given date."""
        df = self.data[ticker]
        if date in df.index:
            return float(df.loc[date, "Close"])
        # Find the last available price
        mask = df.index <= date
        if mask.any():
            return float(df.loc[mask, "Close"].iloc[-1])
        return None

    def _portfolio_value(self, date: pd.Timestamp) -> float:
        """Total NAV = cash + sum of position market values."""
        nav = self.cash
        for ticker, pos in self.positions.items():
            price = self._current_price(ticker, date)
            if price:
                nav += pos.shares * price
        return nav

    def _execute_buy(self, ticker: str, date: pd.Timestamp, signal: float):
        """Buy a position if constraints allow."""
        if ticker in self.positions:
            return  # already holding
        if len(self.positions) >= MAX_POSITIONS:
            return  # max positions reached

        price = self._current_price(ticker, date)
        if price is None or price <= 0:
            return

        nav = self._portfolio_value(date)
        allocation = nav * POSITION_SIZE_PCT
        cost = transaction_cost(price, 1)  # per-share cost rate
        effective_price = price * (1 + (SLIPPAGE_BPS + SPREAD_BPS) / 10_000)

        shares = int(allocation / effective_price)
        if shares <= 0:
            return

        total_cost = transaction_cost(price, shares)
        total_outlay = shares * price + total_cost

        if total_outlay > self.cash:
            shares = int((self.cash - total_cost) / effective_price)
            if shares <= 0:
                return
            total_cost = transaction_cost(price, shares)
            total_outlay = shares * price + total_cost

        self.cash -= total_outlay
        self.positions[ticker] = Position(
            ticker=ticker,
            shares=shares,
            entry_price=price,
            entry_date=date.strftime("%Y-%m-%d"),
        )
        self.trades.append(Trade(
            ticker=ticker,
            action="BUY",
            shares=shares,
            price=price,
            cost=total_cost,
            date=date.strftime("%Y-%m-%d"),
            signal=signal,
        ))

    def _execute_sell(self, ticker: str, date: pd.Timestamp, signal: float):
        """Sell an existing position."""
        if ticker not in self.positions:
            return

        pos = self.positions[ticker]
        price = self._current_price(ticker, date)
        if price is None or price <= 0:
            return

        effective_price = price * (1 - (SLIPPAGE_BPS + SPREAD_BPS) / 10_000)
        total_cost = transaction_cost(price, pos.shares)
        proceeds = pos.shares * effective_price - total_cost

        self.cash += pos.shares * price - total_cost  # actual cash: market price minus cost
        del self.positions[ticker]

        self.trades.append(Trade(
            ticker=ticker,
            action="SELL",
            shares=pos.shares,
            price=price,
            cost=total_cost,
            date=date.strftime("%Y-%m-%d"),
            signal=signal,
        ))

    def run(self, start_date: str, end_date: str) -> BacktestResult:
        """Run backtest over the specified period."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.daily_snapshots.clear()
        clear_fund_cache()

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        # Get all trading days from benchmark
        trading_days = self.benchmark.loc[
            (self.benchmark.index >= start) & (self.benchmark.index <= end)
        ].index

        if len(trading_days) == 0:
            raise ValueError(f"No trading days between {start_date} and {end_date}")

        # Benchmark tracking
        bench_start_price = float(self.benchmark.loc[
            self.benchmark.index >= start, "Close"
        ].iloc[0])

        logger.info("Running backtest: %s to %s (%d trading days)",
                     start_date, end_date, len(trading_days))

        for day in trading_days:
            # Compute signals for each ticker
            signals = {}
            for ticker in self.tickers:
                if ticker not in self.data:
                    continue

                hist = self._get_history_up_to(ticker, day)
                if len(hist) < 200:
                    continue

                # Technical signal (daily)
                t_sig = tech_signal(hist)

                # Fundamental signal (quarterly refresh — use month for cache)
                f_sig = fund_signal(ticker, day.strftime("%Y-%m-%d"))

                # Fused signal
                fused = TECH_WEIGHT * t_sig + FUND_WEIGHT * f_sig
                signals[ticker] = fused

            # Generate orders based on signals
            # First: check sells (free up capital)
            for ticker in list(self.positions.keys()):
                sig = signals.get(ticker, 0.0)
                if sig < SELL_THRESHOLD:
                    self._execute_sell(ticker, day, sig)

            # Then: check buys (ranked by signal strength)
            buy_candidates = [
                (ticker, sig) for ticker, sig in signals.items()
                if sig > BUY_THRESHOLD and ticker not in self.positions
            ]
            buy_candidates.sort(key=lambda x: x[1], reverse=True)

            for ticker, sig in buy_candidates:
                if len(self.positions) >= MAX_POSITIONS:
                    break
                self._execute_buy(ticker, day, sig)

            # Record daily snapshot
            nav = self._portfolio_value(day)
            if day in self.benchmark.index:
                bench_price = float(self.benchmark.loc[day, "Close"])
            else:
                mask = self.benchmark.index <= day
                bench_price = float(self.benchmark.loc[mask, "Close"].iloc[-1]) if mask.any() else bench_start_price

            bench_nav = self.initial_capital * (bench_price / bench_start_price)

            pos_values = {}
            for t, p in self.positions.items():
                px = self._current_price(t, day)
                if px:
                    pos_values[t] = p.shares * px

            self.daily_snapshots.append(DailySnapshot(
                date=day.strftime("%Y-%m-%d"),
                nav=nav,
                cash=self.cash,
                positions=pos_values,
                benchmark_nav=bench_nav,
            ))

        # Close all remaining positions at end
        last_day = trading_days[-1]
        for ticker in list(self.positions.keys()):
            self._execute_sell(ticker, last_day, -999)

        return self._compute_results(start_date, end_date)

    def _compute_results(self, start_date: str, end_date: str) -> BacktestResult:
        """Compute performance metrics from daily snapshots and trades."""
        if not self.daily_snapshots:
            raise ValueError("No snapshots recorded")

        navs = pd.Series(
            [s.nav for s in self.daily_snapshots],
            index=pd.to_datetime([s.date for s in self.daily_snapshots]),
        )
        bench_navs = pd.Series(
            [s.benchmark_nav for s in self.daily_snapshots],
            index=pd.to_datetime([s.date for s in self.daily_snapshots]),
        )

        # Daily returns
        daily_ret = navs.pct_change().dropna()
        bench_daily_ret = bench_navs.pct_change().dropna()

        # Total return
        total_return = (navs.iloc[-1] / self.initial_capital) - 1
        bench_return = (bench_navs.iloc[-1] / self.initial_capital) - 1

        # Annualised return
        n_years = len(daily_ret) / 252
        ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

        # Sharpe ratio (annualised, risk-free = 0 for simplicity)
        if daily_ret.std() > 0:
            sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        if bench_daily_ret.std() > 0:
            bench_sharpe = (bench_daily_ret.mean() / bench_daily_ret.std()) * np.sqrt(252)
        else:
            bench_sharpe = 0.0

        # Maximum drawdown
        cummax = navs.cummax()
        drawdown = (navs - cummax) / cummax
        max_dd = abs(float(drawdown.min()))

        # Calmar ratio
        calmar = ann_return / max_dd if max_dd > 0 else 0.0

        # Trade analysis
        buy_trades = [t for t in self.trades if t.action == "BUY"]
        sell_trades = [t for t in self.trades if t.action == "SELL"]

        # Match buy/sell pairs
        pnl_list = []
        holding_days_list = []
        trade_pairs = []

        for sell in sell_trades:
            matching_buy = next(
                (b for b in buy_trades if b.ticker == sell.ticker
                 and b.date <= sell.date and b not in [p[0] for p in trade_pairs]),
                None,
            )
            if matching_buy:
                trade_pairs.append((matching_buy, sell))
                pnl = (sell.price - matching_buy.price) * sell.shares - sell.cost - matching_buy.cost
                pnl_list.append(pnl)
                days = (pd.Timestamp(sell.date) - pd.Timestamp(matching_buy.date)).days
                holding_days_list.append(days)

        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p <= 0]

        win_rate = len(wins) / len(pnl_list) * 100 if pnl_list else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 1
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        avg_holding = np.mean(holding_days_list) if holding_days_list else 0

        # Monthly returns and win rate
        monthly_ret = navs.resample("ME").last().pct_change().dropna()
        monthly_win_rate = (monthly_ret > 0).mean() * 100 if len(monthly_ret) > 0 else 0

        return BacktestResult(
            period=f"{start_date} to {end_date}",
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_nav=float(navs.iloc[-1]),
            total_return_pct=total_return * 100,
            annualised_return_pct=ann_return * 100,
            sharpe_ratio=float(sharpe),
            max_drawdown_pct=max_dd * 100,
            calmar_ratio=float(calmar),
            win_rate_pct=float(win_rate),
            avg_win_loss_ratio=float(win_loss_ratio),
            total_trades=len(pnl_list),
            avg_holding_days=float(avg_holding),
            monthly_win_rate_pct=float(monthly_win_rate),
            benchmark_return_pct=bench_return * 100,
            benchmark_sharpe=float(bench_sharpe),
            excess_return_pct=(total_return - bench_return) * 100,
            daily_returns=daily_ret.tolist(),
            monthly_returns=monthly_ret.tolist(),
            trades=[asdict(t) for t in self.trades],
            nav_series=[(s.date, s.nav, s.benchmark_nav) for s in self.daily_snapshots],
        )


# ─── Randomisation test ──────────────────────────────────────────────────────

def randomisation_test(
    backtester: MinimalBacktester,
    actual_result: BacktestResult,
    start_date: str,
    end_date: str,
    n_shuffles: int = N_RANDOM_SHUFFLES,
) -> dict:
    """
    Shuffle trade signals randomly N times and compare to actual Sharpe.
    Returns p-value: probability that random signals produce equal/better Sharpe.
    """
    logger.info("Running randomisation test with %d shuffles...", n_shuffles)
    actual_sharpe = actual_result.sharpe_ratio

    random_sharpes = []

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    trading_days = backtester.benchmark.loc[
        (backtester.benchmark.index >= start) & (backtester.benchmark.index <= end)
    ].index

    # Pre-compute all signals
    all_signals: dict[str, dict[str, float]] = {}  # date -> {ticker: signal}
    for day in trading_days:
        day_signals = {}
        for ticker in backtester.tickers:
            if ticker not in backtester.data:
                continue
            hist = backtester._get_history_up_to(ticker, day)
            if len(hist) < 200:
                continue
            t_sig = tech_signal(hist)
            f_sig = fund_signal(ticker, day.strftime("%Y-%m-%d"))
            fused = TECH_WEIGHT * t_sig + FUND_WEIGHT * f_sig
            day_signals[ticker] = fused
        all_signals[day.strftime("%Y-%m-%d")] = day_signals

    # Collect all signal values
    all_signal_values = []
    for day_sigs in all_signals.values():
        all_signal_values.extend(day_sigs.values())
    all_signal_values = np.array(all_signal_values)

    rng = np.random.default_rng(42)

    for i in range(n_shuffles):
        if (i + 1) % 100 == 0:
            logger.info("  Shuffle %d/%d", i + 1, n_shuffles)

        # Shuffle all signals
        shuffled = rng.permutation(all_signal_values).tolist()

        # Reconstruct shuffled signals
        idx = 0
        shuffled_signals = {}
        for date_str, day_sigs in all_signals.items():
            shuffled_day = {}
            for ticker in day_sigs:
                shuffled_day[ticker] = shuffled[idx]
                idx += 1
            shuffled_signals[date_str] = shuffled_day

        # Run simplified backtest with shuffled signals
        cash = backtester.initial_capital
        positions: dict[str, Position] = {}
        navs_list = []

        bench_start_price = float(backtester.benchmark.loc[
            backtester.benchmark.index >= start, "Close"
        ].iloc[0])

        for day in trading_days:
            date_str = day.strftime("%Y-%m-%d")
            day_sigs = shuffled_signals.get(date_str, {})

            # Sell
            for ticker in list(positions.keys()):
                sig = day_sigs.get(ticker, 0.0)
                if sig < SELL_THRESHOLD:
                    pos = positions.pop(ticker)
                    price = backtester._current_price(ticker, day)
                    if price:
                        cost = transaction_cost(price, pos.shares)
                        cash += pos.shares * price - cost

            # Buy
            candidates = [
                (t, s) for t, s in day_sigs.items()
                if s > BUY_THRESHOLD and t not in positions
            ]
            candidates.sort(key=lambda x: x[1], reverse=True)

            for ticker, sig in candidates:
                if len(positions) >= MAX_POSITIONS:
                    break
                price = backtester._current_price(ticker, day)
                if price is None or price <= 0:
                    continue
                nav_est = cash + sum(
                    (backtester._current_price(t, day) or 0) * p.shares
                    for t, p in positions.items()
                )
                alloc = nav_est * POSITION_SIZE_PCT
                eff_price = price * (1 + (SLIPPAGE_BPS + SPREAD_BPS) / 10_000)
                shares = int(alloc / eff_price)
                if shares <= 0:
                    continue
                cost = transaction_cost(price, shares)
                outlay = shares * price + cost
                if outlay > cash:
                    continue
                cash -= outlay
                positions[ticker] = Position(ticker, shares, price, date_str)

            # NAV
            nav = cash
            for t, p in positions.items():
                px = backtester._current_price(t, day)
                if px:
                    nav += p.shares * px
            navs_list.append(nav)

        # Close remaining
        for ticker, pos in positions.items():
            price = backtester._current_price(ticker, trading_days[-1])
            if price:
                cost = transaction_cost(price, pos.shares)
                cash += pos.shares * price - cost

        if len(navs_list) > 1:
            nav_series = pd.Series(navs_list)
            dr = nav_series.pct_change().dropna()
            if dr.std() > 0:
                s = (dr.mean() / dr.std()) * np.sqrt(252)
            else:
                s = 0.0
            random_sharpes.append(float(s))

    random_sharpes = np.array(random_sharpes)
    p_value = float(np.mean(random_sharpes >= actual_sharpe))

    return {
        "actual_sharpe": actual_sharpe,
        "random_sharpe_mean": float(random_sharpes.mean()),
        "random_sharpe_std": float(random_sharpes.std()),
        "random_sharpe_p95": float(np.percentile(random_sharpes, 95)),
        "p_value": p_value,
        "n_shuffles": n_shuffles,
        "significant": p_value < 0.05,
    }


# ─── Report generation ────────────────────────────────────────────────────────

def print_report(result: BacktestResult, randomisation: dict | None = None):
    """Print a formatted performance report."""
    print("\n" + "=" * 70)
    print(f"  BACKTEST REPORT — {result.period}")
    print("=" * 70)

    print(f"\n{'Metric':<35} {'Strategy':>15} {'SPY B&H':>15}")
    print("-" * 65)
    print(f"{'Total Return':<35} {result.total_return_pct:>14.2f}% {result.benchmark_return_pct:>14.2f}%")
    print(f"{'Annualised Return':<35} {result.annualised_return_pct:>14.2f}%")
    print(f"{'Sharpe Ratio':<35} {result.sharpe_ratio:>15.3f} {result.benchmark_sharpe:>15.3f}")
    print(f"{'Max Drawdown':<35} {result.max_drawdown_pct:>14.2f}%")
    print(f"{'Calmar Ratio':<35} {result.calmar_ratio:>15.3f}")
    print(f"{'Excess Return vs SPY':<35} {result.excess_return_pct:>14.2f}%")

    print(f"\n{'Trade Statistics':<35}")
    print("-" * 65)
    print(f"{'Total Round-Trip Trades':<35} {result.total_trades:>15}")
    print(f"{'Win Rate':<35} {result.win_rate_pct:>14.1f}%")
    print(f"{'Avg Win/Loss Ratio':<35} {result.avg_win_loss_ratio:>15.2f}")
    print(f"{'Avg Holding Period (days)':<35} {result.avg_holding_days:>15.1f}")
    print(f"{'Monthly Win Rate':<35} {result.monthly_win_rate_pct:>14.1f}%")

    if randomisation:
        print(f"\n{'Randomisation Test':<35}")
        print("-" * 65)
        print(f"{'Actual Sharpe':<35} {randomisation['actual_sharpe']:>15.3f}")
        print(f"{'Random Mean Sharpe':<35} {randomisation['random_sharpe_mean']:>15.3f}")
        print(f"{'Random 95th Percentile':<35} {randomisation['random_sharpe_p95']:>15.3f}")
        print(f"{'p-value':<35} {randomisation['p_value']:>15.4f}")
        print(f"{'Significant (p < 0.05)':<35} {'✓ YES' if randomisation['significant'] else '✗ NO':>15}")

    print("\n" + "=" * 70)


def save_results(
    results: dict[str, BacktestResult],
    randomisation: dict | None,
    output_dir: str,
):
    """Save results to JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    for period_name, result in results.items():
        data = asdict(result)
        # Convert numpy types
        filepath = os.path.join(output_dir, f"backtest_{period_name}.json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Saved %s results to %s", period_name, filepath)

    if randomisation:
        filepath = os.path.join(output_dir, "randomisation_test.json")
        with open(filepath, "w") as f:
            json.dump(randomisation, f, indent=2)
        logger.info("Saved randomisation results to %s", filepath)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Minimal Alpha Verification Backtest")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL)
    parser.add_argument("--skip-randomisation", action="store_true",
                        help="Skip the 1000-shuffle randomisation test (faster)")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--test-only", action="store_true",
                        help="Only run test set (assumes parameters are final)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    tickers = args.tickers
    print(f"\n🔬 Phase 1 Alpha Verification Backtest")
    print(f"   Tickers: {tickers}")
    print(f"   Capital: ${args.capital:,.0f}")

    # Download all data
    all_data = download_data(tickers, TRAIN_START, TEST_END)
    benchmark = download_benchmark(TRAIN_START, TEST_END)

    if not all_data:
        print("ERROR: No data downloaded. Check network/yfinance.")
        sys.exit(1)

    bt = MinimalBacktester(tickers, all_data, benchmark, args.capital)
    results = {}

    if not args.test_only:
        # --- Training set ---
        print(f"\n{'─' * 50}")
        print(f"  TRAINING SET: {TRAIN_START} to {TRAIN_END}")
        print(f"{'─' * 50}")
        train_result = bt.run(TRAIN_START, TRAIN_END)
        results["train"] = train_result
        print_report(train_result)

        # --- Validation set ---
        print(f"\n{'─' * 50}")
        print(f"  VALIDATION SET: {VALID_START} to {VALID_END}")
        print(f"{'─' * 50}")
        valid_result = bt.run(VALID_START, VALID_END)
        results["validation"] = valid_result
        print_report(valid_result)

    # --- Test set ---
    print(f"\n{'─' * 50}")
    print(f"  TEST SET: {TEST_START} to {TEST_END}")
    print(f"{'─' * 50}")
    test_result = bt.run(TEST_START, TEST_END)
    results["test"] = test_result

    # Randomisation test on test set
    randomisation = None
    if not args.skip_randomisation:
        randomisation = randomisation_test(bt, test_result, TEST_START, TEST_END)

    print_report(test_result, randomisation)

    # Save results
    save_results(results, randomisation, args.output_dir)

    # Gate check summary
    print("\n" + "=" * 70)
    print("  PHASE 1 GATE CHECK (Test Set)")
    print("=" * 70)
    gates = {
        "Sharpe > 1.0": test_result.sharpe_ratio > 1.0,
        "Max DD < 20%": test_result.max_drawdown_pct < 20,
        "Return > SPY": test_result.total_return_pct > test_result.benchmark_return_pct,
        "p-value < 0.05": randomisation["significant"] if randomisation else None,
        "Monthly WR > 50%": test_result.monthly_win_rate_pct > 50,
        "Win/Loss > 1.2": test_result.avg_win_loss_ratio > 1.2,
    }

    all_pass = True
    for gate, passed in gates.items():
        if passed is None:
            status = "⏭ SKIPPED"
        elif passed:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
            all_pass = False
        print(f"  {status}  {gate}")

    if all_pass:
        print("\n  ✅ ALL GATES PASSED — Ready for Phase 2")
    else:
        print("\n  ❌ GATE CHECK FAILED — Do NOT proceed to Phase 2")
        print("     Fix signal logic first. Do NOT re-tune on test set.")

    print("=" * 70)

    return results, randomisation


if __name__ == "__main__":
    main()
