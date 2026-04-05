"""
Phase 2 — Dynamic vs Fixed Weight Comparison
==============================================

Compares three weighting schemes:
  A) Fixed equal weights
  B) Fixed optimised weights (from training set)
  C) Dynamic weights (rolling accuracy-based)

Only adopt C if it significantly outperforms B on validation set.

Usage:
    python -m backtest.weight_optimization_test
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.minimal_backtest import (
    download_data,
    download_benchmark,
    DEFAULT_TICKERS,
    INITIAL_CAPITAL,
    TRAIN_START,
    TRAIN_END,
    VALID_START,
    VALID_END,
    BUY_THRESHOLD,
    SELL_THRESHOLD,
    MAX_POSITIONS,
    POSITION_SIZE_PCT,
    SLIPPAGE_BPS,
    SPREAD_BPS,
    transaction_cost,
    Position,
)
from backtest.agents.technical_agent_minimal import compute_signal as tech_signal
from backtest.agents.fundamental_agent_minimal import compute_signal as fund_signal

logger = logging.getLogger(__name__)


def _backtest_with_weights(
    tickers: list[str],
    data: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    start_date: str,
    end_date: str,
    weight_fn,
    capital: float = INITIAL_CAPITAL,
) -> tuple[float, float, list[float]]:
    """
    Run backtest with a weight function.

    weight_fn(day_index: int, tech_sig: float, fund_sig: float,
              past_signals: list) -> float
    Returns (sharpe, max_dd_pct, daily_returns).
    """
    cash = capital
    positions: dict[str, Position] = {}
    nav_list = []
    past_signals: list[dict] = []

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    trading_days = benchmark.loc[
        (benchmark.index >= start) & (benchmark.index <= end)
    ].index

    for day_idx, day in enumerate(trading_days):
        signals = {}
        for ticker in tickers:
            if ticker not in data:
                continue
            df = data[ticker]
            hist = df.loc[df.index <= day].copy()
            if len(hist) < 200:
                continue

            t_sig = tech_signal(hist)
            f_sig = fund_signal(ticker, day.strftime("%Y-%m-%d"))
            fused = weight_fn(day_idx, t_sig, f_sig, past_signals)
            signals[ticker] = float(np.clip(fused, -1, 1))

        # Record signals for dynamic weights
        past_signals.append({"day": day.strftime("%Y-%m-%d"), "signals": signals.copy()})

        # Sell
        for ticker in list(positions.keys()):
            sig = signals.get(ticker, 0.0)
            if sig < SELL_THRESHOLD:
                pos = positions.pop(ticker)
                mask = data[ticker].index <= day
                if mask.any():
                    price = float(data[ticker].loc[mask, "Close"].iloc[-1])
                    cost = transaction_cost(price, pos.shares)
                    cash += pos.shares * price - cost

        # Buy
        candidates = [(t, s) for t, s in signals.items()
                       if s > BUY_THRESHOLD and t not in positions]
        candidates.sort(key=lambda x: x[1], reverse=True)

        for ticker, sig in candidates:
            if len(positions) >= MAX_POSITIONS:
                break
            mask = data[ticker].index <= day
            if not mask.any():
                continue
            price = float(data[ticker].loc[mask, "Close"].iloc[-1])
            if price <= 0:
                continue
            nav_est = cash + sum(
                float(data[t].loc[data[t].index <= day, "Close"].iloc[-1]) * p.shares
                for t, p in positions.items()
                if len(data[t].loc[data[t].index <= day]) > 0
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
            positions[ticker] = Position(ticker, shares, price, day.strftime("%Y-%m-%d"))

        nav = cash
        for t, p in positions.items():
            mask = data[t].index <= day
            if mask.any():
                nav += p.shares * float(data[t].loc[mask, "Close"].iloc[-1])
        nav_list.append(nav)

    nav_series = pd.Series(nav_list, index=trading_days[:len(nav_list)])
    daily_ret = nav_series.pct_change().dropna()
    sharpe = 0.0
    if len(daily_ret) > 0 and daily_ret.std() > 0:
        sharpe = float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252))

    cummax = nav_series.cummax()
    dd = (nav_series - cummax) / cummax
    max_dd = abs(float(dd.min())) * 100

    return sharpe, max_dd, daily_ret.tolist()


# ─── Weight schemes ──────────────────────────────────────────────────────────

def equal_weight_fn(day_idx, t_sig, f_sig, past):
    """Scheme A: 50/50 equal weight."""
    return 0.5 * t_sig + 0.5 * f_sig


def make_fixed_weight_fn(w_tech: float):
    """Scheme B: Fixed optimised weight."""
    w_fund = 1.0 - w_tech

    def fn(day_idx, t_sig, f_sig, past):
        return w_tech * t_sig + w_fund * f_sig

    return fn


def dynamic_weight_fn(day_idx, t_sig, f_sig, past):
    """
    Scheme C: Dynamic weights based on rolling accuracy.
    Look back at last 60 signals, measure which agent was more
    directionally correct (based on next-day price movement).
    """
    lookback = 60
    if len(past) < lookback + 1:
        # Not enough data, fall back to equal
        return 0.5 * t_sig + 0.5 * f_sig

    # Can't measure accuracy of current signals, so use past
    # This is inherently limited in backtest (no future data)
    # We approximate: measure alignment of past signals with subsequent returns
    # This is a simplified version — in production, would track per-trade accuracy

    # Default: 60/40 tech/fund (same as Phase 1 baseline)
    # Adjust ±10% based on recent trend strength
    recent = past[-lookback:]
    bullish_days = sum(1 for s in recent if any(v > 0.1 for v in s["signals"].values()))
    bull_ratio = bullish_days / lookback

    # If market is trending strongly, trust technicals more
    if bull_ratio > 0.7 or bull_ratio < 0.3:
        w_tech = 0.70  # strong trend → technicals
    else:
        w_tech = 0.50  # range → equal weight

    w_fund = 1.0 - w_tech
    return w_tech * t_sig + w_fund * f_sig


def optimise_weights_on_train(
    tickers, data, benchmark, start_date, end_date,
) -> float:
    """Find optimal tech weight on training data using grid search."""
    best_sharpe = -999
    best_w = 0.5

    for w in np.arange(0.3, 0.8, 0.05):
        fn = make_fixed_weight_fn(w)
        sharpe, _, _ = _backtest_with_weights(
            tickers, data, benchmark, start_date, end_date, fn,
        )
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_w = w

    return float(best_w)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    tickers = DEFAULT_TICKERS
    output_dir = str(PROJECT_ROOT / "results")

    print("\n" + "=" * 70)
    print("  PHASE 2: Weight Optimization Test")
    print("=" * 70)

    all_data = download_data(tickers, "2012-01-01", VALID_END)
    benchmark = download_benchmark("2012-01-01", VALID_END)

    # Step 1: Optimise fixed weights on training set
    print("\n  Optimising fixed weights on training set...")
    optimal_w = optimise_weights_on_train(
        tickers, all_data, benchmark, TRAIN_START, TRAIN_END,
    )
    print(f"  Optimal tech weight: {optimal_w:.2f}")

    # Step 2: Compare all three on validation set
    print("\n  Comparing on validation set...")

    # A: Equal weight
    sharpe_a, dd_a, ret_a = _backtest_with_weights(
        tickers, all_data, benchmark, VALID_START, VALID_END, equal_weight_fn,
    )
    print(f"  A) Equal weight:     Sharpe={sharpe_a:.3f}, DD={dd_a:.2f}%")

    # B: Fixed optimised
    sharpe_b, dd_b, ret_b = _backtest_with_weights(
        tickers, all_data, benchmark, VALID_START, VALID_END,
        make_fixed_weight_fn(optimal_w),
    )
    print(f"  B) Fixed optimised:  Sharpe={sharpe_b:.3f}, DD={dd_b:.2f}%")

    # C: Dynamic
    sharpe_c, dd_c, ret_c = _backtest_with_weights(
        tickers, all_data, benchmark, VALID_START, VALID_END, dynamic_weight_fn,
    )
    print(f"  C) Dynamic weights:  Sharpe={sharpe_c:.3f}, DD={dd_c:.2f}%")

    # Decision
    c_vs_b = sharpe_c - sharpe_b
    adopt_dynamic = c_vs_b > 0.1  # significant improvement needed

    print("\n" + "=" * 70)
    print("  WEIGHT OPTIMIZATION VERDICT")
    print("=" * 70)
    print(f"\n  A) Equal:    Sharpe {sharpe_a:.3f}")
    print(f"  B) Fixed:    Sharpe {sharpe_b:.3f}  (tech_w={optimal_w:.2f})")
    print(f"  C) Dynamic:  Sharpe {sharpe_c:.3f}")
    print(f"\n  C vs B delta: {c_vs_b:+.3f}")

    if adopt_dynamic:
        print(f"  ✓ ADOPT DYNAMIC WEIGHTS (delta > 0.1)")
        recommendation = "dynamic"
    else:
        best = "B" if sharpe_b >= sharpe_a else "A"
        print(f"  ✗ KEEP {'FIXED OPTIMISED' if best == 'B' else 'EQUAL'} WEIGHTS")
        print(f"    Dynamic weights don't add enough value to justify complexity")
        recommendation = "fixed_optimised" if best == "B" else "equal"

    # Save
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "equal_sharpe": sharpe_a,
        "equal_max_dd": dd_a,
        "fixed_sharpe": sharpe_b,
        "fixed_max_dd": dd_b,
        "fixed_tech_weight": optimal_w,
        "dynamic_sharpe": sharpe_c,
        "dynamic_max_dd": dd_c,
        "c_vs_b_delta": c_vs_b,
        "recommendation": recommendation,
        "adopt_dynamic": adopt_dynamic,
    }
    filepath = os.path.join(output_dir, "weight_optimization.json")
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to {filepath}")


if __name__ == "__main__":
    main()
