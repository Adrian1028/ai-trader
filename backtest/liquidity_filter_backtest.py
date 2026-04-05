"""
Step 2 — Liquidity Filter Backtest
====================================

Compares:
  A) Equal-weight all 20 stocks (with macro overlay)
  B) Equal-weight filtered stocks (crowding risk filter + macro overlay)

Both strategies go to cash when macro signal is RISK_OFF.

Usage:
    python -m backtest.liquidity_filter_backtest
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.signals.macro_data import get_macro_data
from src.signals.vix_term_structure import MacroRiskSwitch
from src.signals.liquidity_filter import compute_crowding_risk, filter_stocks
from src.universe.stock_pool import TICKERS, SECTOR_MAP

logger = logging.getLogger(__name__)

TRAIN_START = "2014-01-01"
TEST_START = "2022-01-01"
TEST_END = "2024-12-31"
INITIAL_CAPITAL = 100_000.0
SLIPPAGE_BPS = 5
REBALANCE_FREQUENCY = 21  # trading days (~monthly)


def _download_all(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Download OHLCV for all tickers."""
    import yfinance as yf

    buffer_start = (pd.Timestamp(start) - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=buffer_start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            logger.warning("No data for %s", ticker)
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[ticker] = df
    logger.info("Downloaded %d/%d tickers", len(data), len(tickers))
    return data


def _run_equal_weight_backtest(
    tickers_fn,  # callable(date, all_data) -> list of tickers to hold
    all_data: dict[str, pd.DataFrame],
    macro_signals: pd.DataFrame,
    spy: pd.Series,
    start: str,
    end: str,
    capital: float = INITIAL_CAPITAL,
) -> tuple[pd.Series, pd.Series]:
    """
    Run equal-weight portfolio backtest with macro overlay.

    tickers_fn: function(date, all_data) -> list[str] of tickers to hold

    Returns (nav_series, benchmark_nav_series)
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    # Use SPY trading days
    trading_days = spy.loc[(spy.index >= start_ts) & (spy.index <= end_ts)].index

    nav = capital
    holdings: dict[str, float] = {}  # ticker -> shares
    cash = capital
    nav_list = []
    bench_list = []

    spy_start = float(spy.loc[spy.index >= start_ts].iloc[0])
    days_since_rebalance = REBALANCE_FREQUENCY  # force rebalance on first day

    for day in trading_days:
        # Get macro signal
        if day in macro_signals.index:
            signal = float(macro_signals.loc[day, "signal"])
        else:
            # Forward fill
            prev = macro_signals.loc[macro_signals.index <= day]
            signal = float(prev["signal"].iloc[-1]) if not prev.empty else 1.0

        days_since_rebalance += 1

        # Rebalance check
        should_rebalance = days_since_rebalance >= REBALANCE_FREQUENCY

        if signal == 0.0:
            # RISK_OFF: sell everything
            if holdings:
                for ticker, shares in holdings.items():
                    if ticker in all_data:
                        mask = all_data[ticker].index <= day
                        if mask.any():
                            price = float(all_data[ticker].loc[mask, "Close"].iloc[-1])
                            cash += shares * price * (1 - SLIPPAGE_BPS / 10_000)
                holdings.clear()
                days_since_rebalance = 0
        elif should_rebalance:
            # Get target tickers
            target_tickers = tickers_fn(day, all_data)
            if not target_tickers:
                target_tickers = list(all_data.keys())

            # Compute current NAV
            current_nav = cash
            for ticker, shares in holdings.items():
                if ticker in all_data:
                    mask = all_data[ticker].index <= day
                    if mask.any():
                        current_nav += shares * float(all_data[ticker].loc[mask, "Close"].iloc[-1])

            # Target allocation (equal weight, scaled by macro signal)
            target_weight = signal / len(target_tickers)
            target_value = current_nav * target_weight

            # Sell holdings not in target
            for ticker in list(holdings.keys()):
                if ticker not in target_tickers:
                    if ticker in all_data:
                        mask = all_data[ticker].index <= day
                        if mask.any():
                            price = float(all_data[ticker].loc[mask, "Close"].iloc[-1])
                            cash += holdings[ticker] * price * (1 - SLIPPAGE_BPS / 10_000)
                    del holdings[ticker]

            # Rebalance to equal weight
            for ticker in target_tickers:
                if ticker not in all_data:
                    continue
                mask = all_data[ticker].index <= day
                if not mask.any():
                    continue
                price = float(all_data[ticker].loc[mask, "Close"].iloc[-1])
                if price <= 0:
                    continue

                current_shares = holdings.get(ticker, 0)
                current_value = current_shares * price
                target_shares = int(target_value / price)
                diff = target_shares - current_shares

                if abs(diff) > 0:
                    cost = abs(diff) * price * SLIPPAGE_BPS / 10_000
                    if diff > 0:  # buying
                        outlay = diff * price + cost
                        if outlay <= cash:
                            cash -= outlay
                            holdings[ticker] = current_shares + diff
                    else:  # selling
                        proceeds = abs(diff) * price - cost
                        cash += proceeds
                        holdings[ticker] = current_shares + diff
                        if holdings[ticker] <= 0:
                            del holdings[ticker]

            days_since_rebalance = 0

        # Compute NAV
        nav = cash
        for ticker, shares in holdings.items():
            if ticker in all_data:
                mask = all_data[ticker].index <= day
                if mask.any():
                    nav += shares * float(all_data[ticker].loc[mask, "Close"].iloc[-1])
        nav_list.append(nav)

        # Benchmark
        if day in spy.index:
            bench_list.append(capital * float(spy.loc[day]) / spy_start)
        else:
            bench_list.append(bench_list[-1] if bench_list else capital)

    nav_series = pd.Series(nav_list, index=trading_days[:len(nav_list)])
    bench_series = pd.Series(bench_list, index=trading_days[:len(bench_list)])
    return nav_series, bench_series


def _compute_metrics(nav: pd.Series, bench: pd.Series, capital: float) -> dict:
    """Compute standard performance metrics."""
    daily_ret = nav.pct_change().dropna()
    bench_ret = bench.pct_change().dropna()

    total_return = (nav.iloc[-1] / capital - 1) * 100
    bench_return = (bench.iloc[-1] / capital - 1) * 100

    sharpe = 0.0
    if daily_ret.std() > 0:
        sharpe = float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252))

    bench_sharpe = 0.0
    if bench_ret.std() > 0:
        bench_sharpe = float((bench_ret.mean() / bench_ret.std()) * np.sqrt(252))

    dd = (nav / nav.cummax() - 1)
    max_dd = abs(float(dd.min())) * 100

    bench_dd = (bench / bench.cummax() - 1)
    bench_max_dd = abs(float(bench_dd.min())) * 100

    # 2022 specific
    mask_2022 = (daily_ret.index >= "2022-01-01") & (daily_ret.index <= "2022-12-31")
    y2022 = None
    if mask_2022.any():
        y2022 = float((1 + daily_ret[mask_2022]).prod() - 1) * 100

    return {
        "total_return_pct": total_return,
        "benchmark_return_pct": bench_return,
        "sharpe": sharpe,
        "benchmark_sharpe": bench_sharpe,
        "max_dd_pct": max_dd,
        "benchmark_max_dd_pct": bench_max_dd,
        "year_2022_return": y2022,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    output_dir = str(PROJECT_ROOT / "results")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  STEP 2: Liquidity Filter Backtest")
    print("=" * 70)

    # Load macro data & compute signals
    print("\n  Loading macro data...")
    macro_data = get_macro_data(TRAIN_START, TEST_END)
    macro_df = pd.DataFrame(macro_data).sort_index().ffill()

    switch = MacroRiskSwitch(
        rv_trigger=0.98, rv_warning=0.92, rv_recover=0.90,
        oas_trigger=0.15, oas_recover=0.08, oas_lookback=21, min_hold_days=5,
    )
    macro_signals = switch.compute_signals(macro_df)

    # Download stock data
    print("  Downloading stock data for 20 tickers...")
    all_data = _download_all(TICKERS, TRAIN_START, TEST_END)

    print("  Downloading SPY...")
    from src.signals.macro_data import fetch_from_yfinance
    spy = fetch_from_yfinance("SPY", TRAIN_START, TEST_END)

    # Strategy A: All stocks (no filter)
    def all_tickers_fn(date, data):
        return [t for t in TICKERS if t in data]

    # Strategy B: Filtered stocks
    def filtered_tickers_fn(date, data):
        scores = compute_crowding_risk(data, date)
        retained, excluded = filter_stocks(scores, threshold=1.5)
        if excluded:
            logger.debug("Day %s: excluded %s", date.strftime("%Y-%m-%d"), excluded)
        return retained if retained else list(data.keys())

    print(f"\n  Running Strategy A (all 20 stocks) on test period...")
    nav_a, bench = _run_equal_weight_backtest(
        all_tickers_fn, all_data, macro_signals, spy, TEST_START, TEST_END,
    )
    metrics_a = _compute_metrics(nav_a, bench, INITIAL_CAPITAL)
    print(f"    Return: {metrics_a['total_return_pct']:.2f}%, Sharpe: {metrics_a['sharpe']:.3f}, "
          f"DD: {metrics_a['max_dd_pct']:.2f}%")

    print(f"\n  Running Strategy B (filtered stocks) on test period...")
    nav_b, bench = _run_equal_weight_backtest(
        filtered_tickers_fn, all_data, macro_signals, spy, TEST_START, TEST_END,
    )
    metrics_b = _compute_metrics(nav_b, bench, INITIAL_CAPITAL)
    print(f"    Return: {metrics_b['total_return_pct']:.2f}%, Sharpe: {metrics_b['sharpe']:.3f}, "
          f"DD: {metrics_b['max_dd_pct']:.2f}%")

    # Forward return analysis: are filtered stocks actually worse?
    print("\n  Computing forward return analysis...")
    start_ts = pd.Timestamp(TEST_START)
    end_ts = pd.Timestamp(TEST_END)
    sample_dates = spy.loc[(spy.index >= start_ts) & (spy.index <= end_ts)].index[::REBALANCE_FREQUENCY]

    excluded_fwd_returns = []
    retained_fwd_returns = []

    for date in sample_dates:
        scores = compute_crowding_risk(all_data, date)
        retained, excluded = filter_stocks(scores, threshold=1.5)

        fwd_date = date + pd.DateOffset(days=30)
        for ticker in excluded:
            if ticker in all_data:
                df = all_data[ticker]
                mask_now = df.index <= date
                mask_fwd = df.index <= fwd_date
                if mask_now.any() and mask_fwd.any():
                    p0 = float(df.loc[mask_now, "Close"].iloc[-1])
                    p1 = float(df.loc[mask_fwd, "Close"].iloc[-1])
                    if p0 > 0:
                        excluded_fwd_returns.append((p1 / p0 - 1))

        for ticker in retained:
            if ticker in all_data:
                df = all_data[ticker]
                mask_now = df.index <= date
                mask_fwd = df.index <= fwd_date
                if mask_now.any() and mask_fwd.any():
                    p0 = float(df.loc[mask_now, "Close"].iloc[-1])
                    p1 = float(df.loc[mask_fwd, "Close"].iloc[-1])
                    if p0 > 0:
                        retained_fwd_returns.append((p1 / p0 - 1))

    avg_excluded_fwd = np.mean(excluded_fwd_returns) if excluded_fwd_returns else 0
    avg_retained_fwd = np.mean(retained_fwd_returns) if retained_fwd_returns else 0

    # Gate check
    print(f"\n{'=' * 70}")
    print("  STEP 2 GATE CHECK")
    print(f"{'=' * 70}")

    gates = {
        "Filter DD < Baseline DD": metrics_b["max_dd_pct"] < metrics_a["max_dd_pct"],
        "Filter Sharpe > Baseline Sharpe": metrics_b["sharpe"] > metrics_a["sharpe"],
        "Filter 2022 > Baseline 2022": (
            metrics_b["year_2022_return"] is not None
            and metrics_a["year_2022_return"] is not None
            and metrics_b["year_2022_return"] > metrics_a["year_2022_return"]
        ),
        "Excluded avg 30d return < Retained": avg_excluded_fwd < avg_retained_fwd,
    }

    all_pass = True
    for gate, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        symbol = "v" if passed else "x"
        if not passed:
            all_pass = False
        print(f"  {symbol} {status}  {gate}")

    print(f"\n  Forward return analysis:")
    print(f"    Excluded stocks avg 30d return: {avg_excluded_fwd*100:.2f}%")
    print(f"    Retained stocks avg 30d return: {avg_retained_fwd*100:.2f}%")
    print(f"    N excluded samples: {len(excluded_fwd_returns)}")
    print(f"    N retained samples: {len(retained_fwd_returns)}")

    if all_pass:
        print(f"\n  ALL GATES PASSED -> Proceed to Signal Correlation Check")
    else:
        failed = [g for g, p in gates.items() if not p]
        print(f"\n  {len(failed)} GATE(S) FAILED")

    print(f"{'=' * 70}")

    # Save
    output = {
        "baseline_metrics": metrics_a,
        "filtered_metrics": metrics_b,
        "forward_return_analysis": {
            "excluded_avg_30d_return": avg_excluded_fwd,
            "retained_avg_30d_return": avg_retained_fwd,
            "n_excluded_samples": len(excluded_fwd_returns),
            "n_retained_samples": len(retained_fwd_returns),
        },
        "gates": gates,
        "all_gates_passed": all_pass,
    }
    filepath = os.path.join(output_dir, "liquidity_filter_backtest.json")
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {filepath}")


if __name__ == "__main__":
    main()
