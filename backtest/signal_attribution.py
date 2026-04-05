"""
Signal Attribution Analysis
============================

Decomposes each signal's contribution:
  A) Macro switch only + equal weight
  B) [Liquidity filter — REMOVED, documented as D007]
  C) Macro + insider weighting
  D) Full strategy with risk constraints

Records delta Sharpe for each addition.

Usage:
    python -m backtest.signal_attribution
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.signals.macro_data import get_macro_data, fetch_from_yfinance
from src.signals.vix_term_structure import MacroRiskSwitch
from src.signals.sec_insider import load_insider_data, compute_not_sold_scores, apply_insider_weights
from src.universe.stock_pool import TICKERS, SECTOR_MAP

logger = logging.getLogger(__name__)

TEST_START = "2022-01-01"
TEST_END = "2024-12-31"
INITIAL_CAPITAL = 100_000.0
SLIPPAGE_BPS = 5
REBALANCE_DAYS = 21

MAX_SINGLE_POSITION_PCT = 0.15
MAX_SECTOR_PCT = 0.35


def _download_all(tickers, start, end):
    import yfinance as yf
    buf = (pd.Timestamp(start) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    data = {}
    for t in tickers:
        df = yf.download(t, start=buf, end=end, progress=False, auto_adjust=True)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[t] = df
    return data


def _get_price(data, ticker, date):
    if ticker not in data:
        return 0.0
    df = data[ticker]
    mask = df.index <= date
    return float(df.loc[mask, "Close"].iloc[-1]) if mask.any() else 0.0


def _run_variant(
    all_data, spy, trading_days,
    macro_signals=None,
    insider_data=None,
    use_insider=False,
    use_risk_limits=False,
):
    """Run a strategy variant and return Sharpe + max DD."""
    cash = INITIAL_CAPITAL
    holdings = {}
    nav_list = []
    days_since = REBALANCE_DAYS

    for day in trading_days:
        signal = 1.0
        if macro_signals is not None:
            if day in macro_signals.index:
                signal = float(macro_signals.loc[day, "signal"])
            else:
                prev = macro_signals.loc[macro_signals.index <= day]
                signal = float(prev["signal"].iloc[-1]) if not prev.empty else 1.0

        days_since += 1

        if signal == 0.0 and holdings:
            for t, s in holdings.items():
                p = _get_price(all_data, t, day)
                if p > 0:
                    cash += s * p * (1 - SLIPPAGE_BPS / 10_000)
            holdings.clear()
            days_since = 0
        elif days_since >= REBALANCE_DAYS:
            available = [t for t in TICKERS if t in all_data]
            if not available:
                nav = cash + sum(_get_price(all_data, t, day) * s for t, s in holdings.items())
                nav_list.append(nav)
                continue

            weights = {t: 1.0 / len(available) for t in available}

            if use_insider and insider_data:
                scores = compute_not_sold_scores(
                    insider_data, available, day.strftime("%Y-%m-%d"))
                weights = apply_insider_weights(weights, scores)

            if use_risk_limits:
                for t in weights:
                    if weights[t] > MAX_SINGLE_POSITION_PCT:
                        weights[t] = MAX_SINGLE_POSITION_PCT
                sector_totals = {}
                for t, w in weights.items():
                    sec = SECTOR_MAP.get(t, "Other")
                    sector_totals[sec] = sector_totals.get(sec, 0) + w
                for sec, total in sector_totals.items():
                    if total > MAX_SECTOR_PCT:
                        scale = MAX_SECTOR_PCT / total
                        for t in weights:
                            if SECTOR_MAP.get(t, "Other") == sec:
                                weights[t] *= scale
                total_w = sum(weights.values())
                if total_w > 0:
                    weights = {t: w / total_w for t, w in weights.items()}

            weights = {t: w * signal for t, w in weights.items()}

            current_nav = cash
            for t, s in holdings.items():
                current_nav += _get_price(all_data, t, day) * s

            # Sell non-targets
            for t in list(holdings.keys()):
                if t not in weights or weights.get(t, 0) == 0:
                    p = _get_price(all_data, t, day)
                    if p > 0:
                        cash += holdings[t] * p * (1 - SLIPPAGE_BPS / 10_000)
                    del holdings[t]

            for t, w in weights.items():
                p = _get_price(all_data, t, day)
                if p <= 0:
                    continue
                target_shares = int(current_nav * w / p)
                current_shares = holdings.get(t, 0)
                diff = target_shares - current_shares
                if abs(diff) > 0:
                    cost = abs(diff) * p * SLIPPAGE_BPS / 10_000
                    if diff > 0:
                        outlay = diff * p + cost
                        if outlay <= cash:
                            cash -= outlay
                            holdings[t] = current_shares + diff
                    else:
                        proceeds = abs(diff) * p - cost
                        cash += proceeds
                        new = current_shares + diff
                        if new > 0:
                            holdings[t] = new
                        elif t in holdings:
                            del holdings[t]

            days_since = 0

        nav = cash
        for t, s in holdings.items():
            nav += _get_price(all_data, t, day) * s
        nav_list.append(nav)

    nav_series = pd.Series(nav_list, index=trading_days[:len(nav_list)])
    dr = nav_series.pct_change().dropna()
    sharpe = float((dr.mean() / dr.std()) * np.sqrt(252)) if dr.std() > 0 else 0
    dd = abs(float((nav_series / nav_series.cummax() - 1).min())) * 100
    ret = (nav_series.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    return {"sharpe": sharpe, "max_dd_pct": dd, "total_return_pct": ret}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    output_dir = str(PROJECT_ROOT / "results")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  SIGNAL ATTRIBUTION ANALYSIS")
    print("=" * 70)

    # Load data
    print("  Loading data...")
    macro_data = get_macro_data("2014-01-01", TEST_END)
    macro_df = pd.DataFrame(macro_data).sort_index().ffill()
    switch = MacroRiskSwitch(
        rv_trigger=0.98, rv_warning=0.92, rv_recover=0.90,
        oas_trigger=0.15, oas_recover=0.08, oas_lookback=21, min_hold_days=5,
    )
    macro_signals = switch.compute_signals(macro_df)

    all_data = _download_all(TICKERS, "2014-01-01", TEST_END)
    spy = fetch_from_yfinance("SPY", TEST_START, TEST_END)
    trading_days = spy.loc[(spy.index >= TEST_START) & (spy.index <= TEST_END)].index

    insider_data = load_insider_data(list(all_data.keys()))

    # Variant 0: No macro (pure equal weight buy-and-hold of 20 stocks)
    print("\n  V0: Equal weight buy-and-hold (no macro)...")
    v0 = _run_variant(all_data, spy, trading_days,
                      macro_signals=None, use_insider=False, use_risk_limits=False)
    print(f"    Sharpe: {v0['sharpe']:.3f}, DD: {v0['max_dd_pct']:.2f}%, Return: {v0['total_return_pct']:.2f}%")

    # Variant A: Macro switch + equal weight
    print("  VA: Macro switch + equal weight...")
    va = _run_variant(all_data, spy, trading_days,
                      macro_signals=macro_signals, use_insider=False, use_risk_limits=False)
    print(f"    Sharpe: {va['sharpe']:.3f}, DD: {va['max_dd_pct']:.2f}%, Return: {va['total_return_pct']:.2f}%")

    # Variant B: Macro + insider
    print("  VB: Macro + insider weighted...")
    vb = _run_variant(all_data, spy, trading_days,
                      macro_signals=macro_signals, insider_data=insider_data,
                      use_insider=True, use_risk_limits=False)
    print(f"    Sharpe: {vb['sharpe']:.3f}, DD: {vb['max_dd_pct']:.2f}%, Return: {vb['total_return_pct']:.2f}%")

    # Variant C: Macro + insider + risk limits
    print("  VC: Macro + insider + risk limits...")
    vc = _run_variant(all_data, spy, trading_days,
                      macro_signals=macro_signals, insider_data=insider_data,
                      use_insider=True, use_risk_limits=True)
    print(f"    Sharpe: {vc['sharpe']:.3f}, DD: {vc['max_dd_pct']:.2f}%, Return: {vc['total_return_pct']:.2f}%")

    # Attribution
    print(f"\n{'=' * 70}")
    print("  SIGNAL CONTRIBUTION ATTRIBUTION")
    print(f"{'=' * 70}")

    delta_macro = va["sharpe"] - v0["sharpe"]
    delta_insider = vb["sharpe"] - va["sharpe"]
    delta_risk = vc["sharpe"] - vb["sharpe"]

    print(f"\n  {'Component':<35} {'Delta Sharpe':>15} {'Verdict':>12}")
    print(f"  {'-' * 62}")
    print(f"  {'+ Macro switch':<35} {delta_macro:>+14.3f} {'KEEP' if delta_macro > 0 else 'REMOVE':>12}")
    print(f"  {'+ Insider weighting':<35} {delta_insider:>+14.3f} {'KEEP' if delta_insider > 0 else 'REMOVE':>12}")
    print(f"  {'+ Risk limits':<35} {delta_risk:>+14.3f} {'KEEP' if delta_risk > -0.1 else 'REVIEW':>12}")
    print(f"\n  {'Cumulative Sharpe':<35} {vc['sharpe']:>15.3f}")

    # Save
    output = {
        "v0_no_macro": v0,
        "va_macro_only": va,
        "vb_macro_insider": vb,
        "vc_full": vc,
        "attribution": {
            "macro_delta_sharpe": delta_macro,
            "insider_delta_sharpe": delta_insider,
            "risk_limits_delta_sharpe": delta_risk,
        },
        "recommendations": {
            "macro": "KEEP" if delta_macro > 0 else "REMOVE",
            "insider": "KEEP" if delta_insider > 0 else "REMOVE",
            "risk_limits": "KEEP" if delta_risk > -0.1 else "REVIEW",
        },
    }
    filepath = os.path.join(output_dir, "signal_attribution.json")
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {filepath}")


if __name__ == "__main__":
    main()
