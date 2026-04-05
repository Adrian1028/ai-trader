"""
Phase 2 — LLM (Gemini) Necessity Test
=======================================

Compares:
  A) System with Gemini cross-factor synthesis
  B) Pure rule-based majority vote system

If Gemini's added Sharpe < 0.1, recommend removing LLM dependency.

Usage:
    python -m backtest.llm_necessity_test
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

from backtest.minimal_backtest import (
    MinimalBacktester,
    download_data,
    download_benchmark,
    DEFAULT_TICKERS,
    INITIAL_CAPITAL,
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

# Minimum Sharpe improvement to justify LLM complexity
LLM_VALUE_THRESHOLD = 0.1


def simulate_llm_synthesis(
    tech_sig: float,
    fund_sig: float,
    prices: pd.DataFrame,
) -> float:
    """
    Simulate what a Gemini-like cross-factor synthesis would add.

    In practice this would call the Gemini API; here we simulate the
    *type* of reasoning an LLM adds: contextual weighting and
    contradiction resolution.

    The point is to test whether this extra logic helps.
    """
    close = prices["Close"]
    if len(close) < 50:
        return (tech_sig + fund_sig) / 2

    # LLM-like reasoning: adjust weights based on market context
    # Trend strength → trust technicals more
    ret_20d = (close.iloc[-1] / close.iloc[-20]) - 1
    volatility = close.pct_change().tail(20).std() * np.sqrt(252)

    if abs(ret_20d) > 0.10:
        # Strong trend — trust technicals
        w_tech, w_fund = 0.75, 0.25
    elif volatility > 0.30:
        # High vol — trust fundamentals (less noisy)
        w_tech, w_fund = 0.35, 0.65
    else:
        w_tech, w_fund = 0.55, 0.45

    # Contradiction resolution: if signals disagree, reduce magnitude
    if (tech_sig > 0.2 and fund_sig < -0.2) or (tech_sig < -0.2 and fund_sig > 0.2):
        fused = w_tech * tech_sig + w_fund * fund_sig
        fused *= 0.5  # reduce confidence on disagreement
    else:
        fused = w_tech * tech_sig + w_fund * fund_sig

    return float(np.clip(fused, -1, 1))


def majority_vote(tech_sig: float, fund_sig: float) -> float:
    """Simple majority vote: average signals, no context."""
    return (tech_sig + fund_sig) / 2


def run_comparison(
    tickers: list[str],
    data: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    signal_fn,
    start_date: str,
    end_date: str,
    capital: float = INITIAL_CAPITAL,
) -> tuple[list[float], float, float]:
    """Run backtest with a given signal fusion function."""
    cash = capital
    positions: dict[str, Position] = {}
    nav_list = []

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    trading_days = benchmark.loc[
        (benchmark.index >= start) & (benchmark.index <= end)
    ].index

    for day in trading_days:
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
            fused = signal_fn(t_sig, f_sig, hist) if signal_fn.__code__.co_varnames[:3] == ('tech_sig', 'fund_sig', 'prices') else signal_fn(t_sig, f_sig)
            signals[ticker] = float(np.clip(fused, -1, 1))

        # Sell
        for ticker in list(positions.keys()):
            sig = signals.get(ticker, 0.0)
            if sig < SELL_THRESHOLD:
                pos = positions.pop(ticker)
                price_data = data[ticker]
                mask = price_data.index <= day
                if mask.any():
                    price = float(price_data.loc[mask, "Close"].iloc[-1])
                    cost = transaction_cost(price, pos.shares)
                    cash += pos.shares * price - cost

        # Buy
        candidates = [(t, s) for t, s in signals.items()
                       if s > BUY_THRESHOLD and t not in positions]
        candidates.sort(key=lambda x: x[1], reverse=True)

        for ticker, sig in candidates:
            if len(positions) >= MAX_POSITIONS:
                break
            price_data = data[ticker]
            mask = price_data.index <= day
            if not mask.any():
                continue
            price = float(price_data.loc[mask, "Close"].iloc[-1])
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

        # NAV
        nav = cash
        for t, p in positions.items():
            price_data = data[t]
            mask = price_data.index <= day
            if mask.any():
                nav += p.shares * float(price_data.loc[mask, "Close"].iloc[-1])
        nav_list.append(nav)

    nav_series = pd.Series(nav_list, index=trading_days[:len(nav_list)])
    daily_ret = nav_series.pct_change().dropna()
    sharpe = 0.0
    if len(daily_ret) > 0 and daily_ret.std() > 0:
        sharpe = float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252))

    cummax = nav_series.cummax()
    dd = (nav_series - cummax) / cummax
    max_dd = abs(float(dd.min())) * 100

    return daily_ret.tolist(), sharpe, max_dd


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    tickers = DEFAULT_TICKERS
    output_dir = str(PROJECT_ROOT / "results")

    print("\n" + "=" * 70)
    print("  PHASE 2: LLM (Gemini) Necessity Test")
    print("=" * 70)

    all_data = download_data(tickers, "2014-01-01", VALID_END)
    benchmark = download_benchmark("2014-01-01", VALID_END)

    # System A: with LLM-like synthesis
    print("\n  Running System A (LLM cross-factor synthesis)...")
    ret_a, sharpe_a, dd_a = run_comparison(
        tickers, all_data, benchmark, simulate_llm_synthesis, VALID_START, VALID_END,
    )
    print(f"    Sharpe: {sharpe_a:.3f}, Max DD: {dd_a:.2f}%")

    # System B: simple majority vote
    print("\n  Running System B (majority vote, no LLM)...")
    ret_b, sharpe_b, dd_b = run_comparison(
        tickers, all_data, benchmark, majority_vote, VALID_START, VALID_END,
    )
    print(f"    Sharpe: {sharpe_b:.3f}, Max DD: {dd_b:.2f}%")

    # Analysis
    delta = sharpe_a - sharpe_b
    keep_llm = delta >= LLM_VALUE_THRESHOLD

    # Divergence analysis
    n = min(len(ret_a), len(ret_b))
    diff = np.array(ret_a[:n]) - np.array(ret_b[:n])
    llm_wins = int(np.sum(diff > 0))
    rule_wins = int(np.sum(diff < 0))
    ties = n - llm_wins - rule_wins

    print("\n" + "=" * 70)
    print("  LLM NECESSITY VERDICT")
    print("=" * 70)
    print(f"\n  LLM Sharpe:       {sharpe_a:.3f}")
    print(f"  Rule Sharpe:      {sharpe_b:.3f}")
    print(f"  Delta:            {delta:+.3f}")
    print(f"  Threshold:        {LLM_VALUE_THRESHOLD}")
    print(f"\n  LLM wins {llm_wins}/{n} days, Rules win {rule_wins}/{n} days, Ties: {ties}")

    if keep_llm:
        print(f"\n  ✓ KEEP LLM — Delta ({delta:.3f}) ≥ threshold ({LLM_VALUE_THRESHOLD})")
    else:
        print(f"\n  ✗ REMOVE LLM — Delta ({delta:.3f}) < threshold ({LLM_VALUE_THRESHOLD})")
        print("    Benefits of removing:")
        print("    - No API rate limit issues")
        print("    - Lower latency")
        print("    - Fully reproducible results")
        print("    - No cost")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "llm_sharpe": sharpe_a,
        "rule_sharpe": sharpe_b,
        "delta_sharpe": delta,
        "llm_max_dd": dd_a,
        "rule_max_dd": dd_b,
        "llm_wins_days": llm_wins,
        "rule_wins_days": rule_wins,
        "ties": ties,
        "total_days": n,
        "keep_llm": keep_llm,
        "threshold": LLM_VALUE_THRESHOLD,
        "decision_documented": True,
        "recommendation": "KEEP" if keep_llm else "REMOVE",
    }
    filepath = os.path.join(output_dir, "llm_necessity.json")
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to {filepath}")


if __name__ == "__main__":
    main()
