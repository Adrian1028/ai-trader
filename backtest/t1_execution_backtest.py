"""
T+1 Execution Bias Test — The Most Important Backtest
======================================================

Compares three execution assumptions:
  A) T+0: Signal at close, execute at close (UNREALISTIC — current baseline)
  B) T+1 Open: Signal at T close, execute at T+1 Open (+0.05% slippage)
  C) T+1 VWAP: Signal at T close, execute at T+1 VWAP proxy (+0.10% slippage)

If Version B Sharpe < 0.70, the signal's alpha was mostly look-ahead bias.

Usage:
    python -m backtest.t1_execution_backtest
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

from src.signals.macro_data import get_macro_data, fetch_from_yfinance
from src.signals.vix_term_structure import MacroRiskSwitch

logger = logging.getLogger(__name__)

# Periods
TRAIN_START = "2010-01-01"
TEST_START = "2022-01-01"
TEST_END = "2024-12-31"

INITIAL_CAPITAL = 100_000.0

# Macro switch parameters (validated in Step 1)
SWITCH_PARAMS = dict(
    rv_trigger=0.98, rv_warning=0.92, rv_recover=0.90,
    oas_trigger=0.15, oas_recover=0.08, oas_lookback=21, min_hold_days=5,
)


@dataclass
class VersionResult:
    version: str
    description: str
    total_return_pct: float
    annualised_return_pct: float
    sharpe: float
    max_dd_pct: float
    year_2022_return_pct: float | None
    spy_total_return_pct: float
    spy_sharpe: float
    spy_max_dd_pct: float
    spy_2022_return_pct: float | None
    excess_return_pct: float
    n_trades: int


def _download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data (need Open, High, Low, Close)."""
    import yfinance as yf
    buf = (pd.Timestamp(start) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=buf, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def run_version(
    spy_ohlcv: pd.DataFrame,
    macro_signals: pd.DataFrame,
    start: str,
    end: str,
    version: str,
    capital: float = INITIAL_CAPITAL,
) -> VersionResult:
    """
    Run a single execution version.

    version:
      "A" = T+0 close-to-close (unrealistic baseline)
      "B" = T+1 Open execution (+0.05% slippage)
      "C" = T+1 VWAP proxy execution (+0.10% slippage)
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    # Get trading days
    spy_period = spy_ohlcv.loc[(spy_ohlcv.index >= start_ts) & (spy_ohlcv.index <= end_ts)]
    trading_days = spy_period.index

    if len(trading_days) < 10:
        raise ValueError(f"Not enough trading days: {len(trading_days)}")

    # Slippage
    if version == "A":
        slippage_bps = 3  # minimal (just spread)
    elif version == "B":
        slippage_bps = 8  # 0.05% opening slippage + 0.03% spread
    else:  # C
        slippage_bps = 13  # 0.10% VWAP slippage + 0.03% spread

    # Build daily execution prices
    exec_prices = pd.Series(index=trading_days, dtype=float)
    for i, day in enumerate(trading_days):
        if version == "A":
            # Execute at today's close
            exec_prices[day] = float(spy_ohlcv.loc[day, "Close"])
        elif version == "B":
            # Execute at NEXT day's open
            if i + 1 < len(trading_days):
                next_day = trading_days[i + 1]
                exec_prices[day] = float(spy_ohlcv.loc[next_day, "Open"])
            else:
                exec_prices[day] = float(spy_ohlcv.loc[day, "Close"])
        else:  # C
            # Execute at NEXT day's VWAP proxy
            if i + 1 < len(trading_days):
                next_day = trading_days[i + 1]
                h = float(spy_ohlcv.loc[next_day, "High"])
                l = float(spy_ohlcv.loc[next_day, "Low"])
                c = float(spy_ohlcv.loc[next_day, "Close"])
                exec_prices[day] = (h + l + c) / 3
            else:
                exec_prices[day] = float(spy_ohlcv.loc[day, "Close"])

    # Run backtest
    cash = capital
    shares = 0.0
    nav_list = []
    n_trades = 0

    # Signal alignment: for versions B and C, the signal from day T
    # determines the trade on day T+1. So we need to shift signals.
    signal_series = pd.Series(index=trading_days, dtype=float)
    for day in trading_days:
        if day in macro_signals.index:
            signal_series[day] = float(macro_signals.loc[day, "signal"])
        else:
            prev = macro_signals.loc[macro_signals.index <= day]
            signal_series[day] = float(prev["signal"].iloc[-1]) if not prev.empty else 1.0

    if version in ("B", "C"):
        # Shift signal by 1 day: today's signal drives tomorrow's trade
        # This means the signal available on day T applies to day T+1
        signal_series = signal_series.shift(1).fillna(1.0)

    prev_signal = 1.0
    for day in trading_days:
        signal = signal_series[day]
        price = exec_prices[day]

        if pd.isna(signal) or pd.isna(price) or price <= 0:
            nav = cash + shares * float(spy_ohlcv.loc[day, "Close"])
            nav_list.append(nav)
            continue

        # Determine target allocation
        target_pct = signal  # 1.0, 0.5, or 0.0

        # Current equity value at execution price
        current_nav = cash + shares * price

        # Target shares
        target_value = current_nav * target_pct
        target_shares = target_value / price if price > 0 else 0

        # Only trade if allocation changed
        if abs(target_pct - prev_signal) > 0.01:
            diff_shares = target_shares - shares
            trade_value = abs(diff_shares * price)
            cost = trade_value * (slippage_bps / 10_000)

            if diff_shares > 0:  # buying
                actual_cost = diff_shares * price + cost
                if actual_cost <= cash:
                    cash -= actual_cost
                    shares += diff_shares
                    n_trades += 1
            elif diff_shares < 0:  # selling
                proceeds = abs(diff_shares) * price - cost
                cash += proceeds
                shares += diff_shares  # diff_shares is negative
                n_trades += 1

            prev_signal = target_pct

        # NAV at close price (for performance measurement)
        close_price = float(spy_ohlcv.loc[day, "Close"])
        nav = cash + shares * close_price
        nav_list.append(nav)

    nav_series = pd.Series(nav_list, index=trading_days[:len(nav_list)])

    # Metrics
    daily_ret = nav_series.pct_change().dropna()
    total_return = (nav_series.iloc[-1] / capital - 1)
    n_years = len(daily_ret) / 252
    ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    sharpe = 0.0
    if daily_ret.std() > 0:
        sharpe = float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252))

    dd = (nav_series / nav_series.cummax() - 1)
    max_dd = abs(float(dd.min()))

    # 2022
    mask_2022 = (daily_ret.index >= "2022-01-01") & (daily_ret.index <= "2022-12-31")
    y2022 = float((1 + daily_ret[mask_2022]).prod() - 1) * 100 if mask_2022.any() else None

    # SPY benchmark
    spy_close = spy_ohlcv.loc[trading_days, "Close"]
    spy_ret = spy_close.pct_change().dropna()
    spy_total = (spy_close.iloc[-1] / spy_close.iloc[0] - 1)
    spy_sharpe = float((spy_ret.mean() / spy_ret.std()) * np.sqrt(252)) if spy_ret.std() > 0 else 0
    spy_nav = spy_close / spy_close.iloc[0] * capital
    spy_dd = abs(float((spy_nav / spy_nav.cummax() - 1).min()))
    spy_2022 = None
    mask_spy_2022 = (spy_ret.index >= "2022-01-01") & (spy_ret.index <= "2022-12-31")
    if mask_spy_2022.any():
        spy_2022 = float((1 + spy_ret[mask_spy_2022]).prod() - 1) * 100

    descriptions = {
        "A": "T+0 Close (unrealistic baseline)",
        "B": "T+1 Open (+0.05% slip)",
        "C": "T+1 VWAP (+0.10% slip)",
    }

    return VersionResult(
        version=version,
        description=descriptions[version],
        total_return_pct=total_return * 100,
        annualised_return_pct=ann_return * 100,
        sharpe=sharpe,
        max_dd_pct=max_dd * 100,
        year_2022_return_pct=y2022,
        spy_total_return_pct=spy_total * 100,
        spy_sharpe=spy_sharpe,
        spy_max_dd_pct=spy_dd * 100,
        spy_2022_return_pct=spy_2022,
        excess_return_pct=(total_return - spy_total) * 100,
        n_trades=n_trades,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    output_dir = str(PROJECT_ROOT / "results")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("  T+1 EXECUTION BIAS TEST")
    print("  Can the signal survive realistic execution delays?")
    print("=" * 80)

    # Load macro data
    print("\n  Loading macro data...")
    macro_data = get_macro_data(TRAIN_START, TEST_END)
    macro_df = pd.DataFrame(macro_data).sort_index().ffill()
    switch = MacroRiskSwitch(**SWITCH_PARAMS)
    macro_signals = switch.compute_signals(macro_df)

    # Load SPY OHLCV
    print("  Loading SPY OHLCV...")
    spy_ohlcv = _download_ohlcv("SPY", TRAIN_START, TEST_END)
    print(f"  SPY: {len(spy_ohlcv)} rows, columns: {list(spy_ohlcv.columns)}")

    results = []

    for version in ["A", "B", "C"]:
        print(f"\n  Running Version {version}...")
        # Full test period
        r = run_version(spy_ohlcv, macro_signals, TEST_START, TEST_END, version)
        results.append(r)
        print(f"    Sharpe: {r.sharpe:.3f}, Return: {r.total_return_pct:.2f}%, "
              f"DD: {r.max_dd_pct:.2f}%, Trades: {r.n_trades}")

    # Also run on training + validation for reference
    print("\n  Running Version B on full period (2010-2024) for reference...")
    r_full = run_version(spy_ohlcv, macro_signals, TRAIN_START, TEST_END, "B")
    print(f"    Full period B: Sharpe {r_full.sharpe:.3f}, Return {r_full.total_return_pct:.2f}%")

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"  T+1 EXECUTION BIAS — COMPARISON TABLE (Test: {TEST_START} to {TEST_END})")
    print(f"{'=' * 80}")
    print(f"\n  {'Version':<30} {'Sharpe':>8} {'Ann Ret':>10} {'Max DD':>9} {'2022':>9} {'vs SPY':>9} {'Trades':>8}")
    print(f"  {'-' * 84}")
    for r in results:
        y2022 = f"{r.year_2022_return_pct:.1f}%" if r.year_2022_return_pct is not None else "N/A"
        print(f"  {r.description:<30} {r.sharpe:>8.3f} {r.annualised_return_pct:>9.2f}% "
              f"{r.max_dd_pct:>8.2f}% {y2022:>9} {r.excess_return_pct:>+8.2f}% {r.n_trades:>8}")

    spy_r = results[0]  # all have same SPY baseline
    print(f"  {'SPY Buy & Hold':<30} {spy_r.spy_sharpe:>8.3f} "
          f"{'':>10} {spy_r.spy_max_dd_pct:>8.2f}% "
          f"{str(spy_r.spy_2022_return_pct)+'%' if spy_r.spy_2022_return_pct else 'N/A':>9}")

    # Sharpe decay analysis
    a_sharpe = results[0].sharpe
    b_sharpe = results[1].sharpe
    c_sharpe = results[2].sharpe

    if a_sharpe > 0:
        decay_b = (a_sharpe - b_sharpe) / a_sharpe * 100
        decay_c = (a_sharpe - c_sharpe) / a_sharpe * 100
    else:
        decay_b = 0
        decay_c = 0

    print(f"\n  Sharpe Decay Analysis:")
    print(f"    A -> B decay: {decay_b:.1f}%")
    print(f"    A -> C decay: {decay_c:.1f}%")
    if decay_b > 50:
        print(f"    WARNING: >50% decay suggests alpha was mostly look-ahead bias")
    elif decay_b > 20:
        print(f"    MODERATE: 20-50% decay, signal still valuable but expectations reduced")
    else:
        print(f"    GOOD: <20% decay, signal is robust to execution delay")

    # Gate check (Version B)
    b = results[1]
    print(f"\n{'=' * 80}")
    print(f"  VERSION B (T+1 Open) GATE CHECK")
    print(f"{'=' * 80}")

    gates = {
        "Sharpe > 0.70": b.sharpe > 0.70,
        "2022 Return > SPY 2022": (
            b.year_2022_return_pct is not None
            and b.spy_2022_return_pct is not None
            and b.year_2022_return_pct > b.spy_2022_return_pct
        ),
        "Max DD < 70% of SPY DD": b.max_dd_pct < b.spy_max_dd_pct * 0.70,
        "Ann Return > SPY Ann Return": b.annualised_return_pct > (
            ((1 + b.spy_total_return_pct / 100) ** (1 / 3) - 1) * 100  # 3-year annualised
        ),
    }

    all_pass = True
    for gate, passed in gates.items():
        status = "v PASS" if passed else "x FAIL"
        if not passed:
            all_pass = False
        print(f"  {status}  {gate}")

    if all_pass:
        print(f"\n  ALL GATES PASSED -> Signal survives T+1 execution")
        print(f"  -> Proceed to Step 2 (multi-index test)")
    else:
        failed = [g for g, p in gates.items() if not p]
        print(f"\n  {len(failed)} GATE(S) FAILED")
        if decay_b > 50:
            print(f"  -> Signal alpha was mostly look-ahead bias. Return to research.")
        else:
            print(f"  -> Signal weakened but may still be viable. Analyse specific failures.")

    print(f"{'=' * 80}")

    # Save
    output = {
        "results": [asdict(r) for r in results],
        "full_period_b": asdict(r_full),
        "sharpe_decay_b_pct": decay_b,
        "sharpe_decay_c_pct": decay_c,
        "gates": {g: p for g, p in gates.items()},
        "all_gates_passed": all_pass,
    }
    filepath = os.path.join(output_dir, "t1_execution_backtest.json")
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {filepath}")

    return results


if __name__ == "__main__":
    main()
