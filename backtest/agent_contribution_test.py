"""
Phase 2 — Agent Marginal Contribution Test
============================================

Methodology: Start with the Phase 1 baseline (technical + fundamental),
then add one agent at a time and measure the marginal improvement.

Retention rule:
  - ΔSharpe > 0.1  AND  p < 0.05  → RETAIN
  - ΔSharpe > 0    AND  p > 0.05  → WATCH (tentatively retain)
  - ΔSharpe ≤ 0                    → REMOVE

Usage:
    python -m backtest.agent_contribution_test
"""

from __future__ import annotations

import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

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
    TEST_START,
    TEST_END,
    VALID_START,
    VALID_END,
    TECH_WEIGHT,
    FUND_WEIGHT,
    BUY_THRESHOLD,
    SELL_THRESHOLD,
    MAX_POSITIONS,
    POSITION_SIZE_PCT,
    SLIPPAGE_BPS,
    SPREAD_BPS,
    transaction_cost,
)
from backtest.agents.technical_agent_minimal import compute_signal as tech_signal
from backtest.agents.fundamental_agent_minimal import compute_signal as fund_signal

logger = logging.getLogger(__name__)


# ─── Additional Agent Interfaces ──────────────────────────────────────────────

class AdditionalAgent(ABC):
    """Interface for agents being tested for marginal contribution."""

    name: str
    weight: float = 0.15  # default weight when added

    @abstractmethod
    def compute_signal(self, ticker: str, date: pd.Timestamp,
                       prices: pd.DataFrame) -> float:
        """Return signal in [-1, +1]."""
        ...


class SentimentAgent(AdditionalAgent):
    """Simplified news sentiment agent using price-volume divergence as proxy."""
    name = "sentiment"
    weight = 0.15

    def compute_signal(self, ticker: str, date: pd.Timestamp,
                       prices: pd.DataFrame) -> float:
        if len(prices) < 20:
            return 0.0
        recent = prices.tail(20)
        # Price-volume divergence: rising volume + falling price = negative sentiment proxy
        price_ret = (recent["Close"].iloc[-1] / recent["Close"].iloc[0]) - 1
        vol_ret = (recent["Volume"].iloc[-5:].mean() /
                   recent["Volume"].iloc[:5].mean()) - 1

        if price_ret < -0.02 and vol_ret > 0.3:
            return -0.5  # panic selling
        elif price_ret > 0.02 and vol_ret > 0.3:
            return 0.5   # strong buying interest
        elif price_ret > 0.02 and vol_ret < -0.2:
            return -0.3  # rising on thin volume (suspicious)
        return 0.0


class MacroAgent(AdditionalAgent):
    """Simplified macro agent using SPY as market regime proxy."""
    name = "macro"
    weight = 0.10

    def compute_signal(self, ticker: str, date: pd.Timestamp,
                       prices: pd.DataFrame) -> float:
        # Use the stock's own 200-day trend as macro proxy
        if len(prices) < 200:
            return 0.0
        close = prices["Close"]
        sma200 = close.rolling(200).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        current = close.iloc[-1]

        if current > sma200 and sma50 > sma200:
            return 0.4   # bull regime
        elif current < sma200 and sma50 < sma200:
            return -0.4  # bear regime
        return 0.0


class InsiderAgent(AdditionalAgent):
    """Simplified insider trading agent using price momentum divergence."""
    name = "insider"
    weight = 0.08

    def compute_signal(self, ticker: str, date: pd.Timestamp,
                       prices: pd.DataFrame) -> float:
        # Proxy: unusual volume spikes before price moves
        if len(prices) < 60:
            return 0.0
        recent_vol = prices["Volume"].tail(5).mean()
        avg_vol = prices["Volume"].tail(60).mean()
        if avg_vol == 0:
            return 0.0

        vol_ratio = recent_vol / avg_vol
        price_5d = (prices["Close"].iloc[-1] / prices["Close"].iloc[-6]) - 1

        if vol_ratio > 2.0 and abs(price_5d) < 0.02:
            # Big volume, small price move — someone knows something
            return 0.3 if recent_vol > avg_vol else -0.3
        return 0.0


class SocialSentimentAgent(AdditionalAgent):
    """Simplified social sentiment using short-term momentum reversal."""
    name = "social_sentiment"
    weight = 0.07

    def compute_signal(self, ticker: str, date: pd.Timestamp,
                       prices: pd.DataFrame) -> float:
        # Proxy: extreme short-term moves suggest retail FOMO
        if len(prices) < 10:
            return 0.0
        ret_5d = (prices["Close"].iloc[-1] / prices["Close"].iloc[-6]) - 1

        if ret_5d > 0.10:
            return -0.4  # retail FOMO, likely to reverse
        elif ret_5d < -0.10:
            return 0.4   # retail panic, likely to bounce
        return 0.0


class OptionsFlowAgent(AdditionalAgent):
    """Simplified options flow using volatility expansion detection."""
    name = "options_flow"
    weight = 0.08

    def compute_signal(self, ticker: str, date: pd.Timestamp,
                       prices: pd.DataFrame) -> float:
        if len(prices) < 30:
            return 0.0
        # Use realized volatility as proxy for implied volatility
        returns = prices["Close"].pct_change().dropna()
        short_vol = returns.tail(5).std() * np.sqrt(252)
        long_vol = returns.tail(30).std() * np.sqrt(252)

        if long_vol == 0:
            return 0.0

        vol_ratio = short_vol / long_vol

        if vol_ratio > 1.5:
            return -0.3  # volatility expansion → caution
        elif vol_ratio < 0.6:
            return 0.3   # volatility compression → potential breakout
        return 0.0


# ─── Contribution testing engine ──────────────────────────────────────────────

ADDITIONAL_AGENTS = [
    SentimentAgent(),
    MacroAgent(),
    InsiderAgent(),
    SocialSentimentAgent(),
    OptionsFlowAgent(),
]


@dataclass
class ContributionResult:
    agent_name: str
    baseline_sharpe: float
    new_sharpe: float
    delta_sharpe: float
    baseline_max_dd: float
    new_max_dd: float
    signal_disagreement_rate: float
    p_value: float
    decision: Literal["RETAIN", "WATCH", "REMOVE"]
    retained: bool


def run_backtest_with_agents(
    tickers: list[str],
    data: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    additional_agents: list[AdditionalAgent],
    start_date: str,
    end_date: str,
    capital: float = INITIAL_CAPITAL,
) -> tuple[list[float], float, float, int]:
    """
    Run backtest with baseline + additional agents.
    Returns (daily_returns_list, sharpe, max_dd_pct, n_trades).
    """
    from backtest.minimal_backtest import Position

    cash = capital
    positions: dict[str, Position] = {}
    nav_list = []

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    trading_days = benchmark.loc[
        (benchmark.index >= start) & (benchmark.index <= end)
    ].index

    # Compute weights: scale baseline weights down to make room for new agents
    total_additional_weight = sum(a.weight for a in additional_agents)
    scale = 1.0 - total_additional_weight
    scaled_tech = TECH_WEIGHT * scale
    scaled_fund = FUND_WEIGHT * scale

    for day in trading_days:
        signals = {}
        for ticker in tickers:
            if ticker not in data:
                continue

            df = data[ticker]
            hist = df.loc[df.index <= day].copy()
            if len(hist) < 200:
                continue

            # Baseline signals
            t_sig = tech_signal(hist)
            f_sig = fund_signal(ticker, day.strftime("%Y-%m-%d"))
            fused = scaled_tech * t_sig + scaled_fund * f_sig

            # Additional agent signals
            for agent in additional_agents:
                a_sig = agent.compute_signal(ticker, day, hist)
                fused += agent.weight * a_sig

            signals[ticker] = float(np.clip(fused, -1, 1))

        # Sell
        for ticker in list(positions.keys()):
            sig = signals.get(ticker, 0.0)
            if sig < SELL_THRESHOLD:
                pos = positions.pop(ticker)
                price_data = data[ticker]
                if day in price_data.index:
                    price = float(price_data.loc[day, "Close"])
                else:
                    mask = price_data.index <= day
                    price = float(price_data.loc[mask, "Close"].iloc[-1]) if mask.any() else 0
                if price > 0:
                    cost = transaction_cost(price, pos.shares)
                    cash += pos.shares * price - cost

        # Buy
        candidates = [
            (t, s) for t, s in signals.items()
            if s > BUY_THRESHOLD and t not in positions
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        for ticker, sig in candidates:
            if len(positions) >= MAX_POSITIONS:
                break
            price_data = data[ticker]
            if day in price_data.index:
                price = float(price_data.loc[day, "Close"])
            else:
                mask = price_data.index <= day
                price = float(price_data.loc[mask, "Close"].iloc[-1]) if mask.any() else 0
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

    # Close remaining
    for ticker, pos in list(positions.items()):
        price_data = data[ticker]
        mask = price_data.index <= trading_days[-1]
        if mask.any():
            price = float(price_data.loc[mask, "Close"].iloc[-1])
            cost = transaction_cost(price, pos.shares)
            cash += pos.shares * price - cost

    nav_series = pd.Series(nav_list, index=trading_days[:len(nav_list)])
    daily_ret = nav_series.pct_change().dropna()

    sharpe = 0.0
    if len(daily_ret) > 0 and daily_ret.std() > 0:
        sharpe = float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252))

    cummax = nav_series.cummax()
    dd = (nav_series - cummax) / cummax
    max_dd = abs(float(dd.min())) * 100

    return daily_ret.tolist(), sharpe, max_dd, len(nav_list)


def paired_t_test(returns_a: list[float], returns_b: list[float]) -> float:
    """Paired t-test on daily returns. Returns p-value."""
    from scipy import stats

    n = min(len(returns_a), len(returns_b))
    if n < 30:
        return 1.0

    a = np.array(returns_a[:n])
    b = np.array(returns_b[:n])
    diff = a - b

    t_stat = diff.mean() / (diff.std() / np.sqrt(n))
    p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
    return p_value


def test_all_agents(
    tickers: list[str] | None = None,
    output_dir: str | None = None,
):
    """Run the full agent contribution test."""
    tickers = tickers or DEFAULT_TICKERS
    output_dir = output_dir or str(PROJECT_ROOT / "results")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("\n" + "=" * 70)
    print("  PHASE 2: Agent Marginal Contribution Test")
    print("=" * 70)

    # Download data
    all_data = download_data(tickers, "2014-01-01", TEST_END)
    benchmark = download_benchmark("2014-01-01", TEST_END)

    # Baseline: tech + fundamental only (on validation set)
    print("\n  Running baseline (Technical + Fundamental)...")
    baseline_returns, baseline_sharpe, baseline_dd, _ = run_backtest_with_agents(
        tickers, all_data, benchmark, [], VALID_START, VALID_END,
    )
    print(f"  Baseline Sharpe: {baseline_sharpe:.3f}, Max DD: {baseline_dd:.2f}%")

    results: list[ContributionResult] = []

    for agent in ADDITIONAL_AGENTS:
        print(f"\n  Testing + {agent.name} (weight={agent.weight})...")

        new_returns, new_sharpe, new_dd, _ = run_backtest_with_agents(
            tickers, all_data, benchmark, [agent], VALID_START, VALID_END,
        )

        delta_sharpe = new_sharpe - baseline_sharpe

        # p-value via paired t-test
        try:
            p_val = paired_t_test(new_returns, baseline_returns)
        except Exception:
            p_val = 1.0

        # Signal disagreement rate
        disagreements = 0
        total = 0
        start = pd.Timestamp(VALID_START)
        end = pd.Timestamp(VALID_END)
        trading_days = benchmark.loc[
            (benchmark.index >= start) & (benchmark.index <= end)
        ].index

        for day in trading_days[::5]:  # sample every 5 days
            for ticker in tickers:
                if ticker not in all_data:
                    continue
                df = all_data[ticker]
                hist = df.loc[df.index <= day]
                if len(hist) < 200:
                    continue

                t_sig = tech_signal(hist)
                f_sig = fund_signal(ticker, day.strftime("%Y-%m-%d"))
                baseline_fused = TECH_WEIGHT * t_sig + FUND_WEIGHT * f_sig
                agent_sig = agent.compute_signal(ticker, day, hist)

                if (baseline_fused > 0 and agent_sig < -0.2) or \
                   (baseline_fused < 0 and agent_sig > 0.2):
                    disagreements += 1
                total += 1

        disagree_rate = disagreements / max(total, 1)

        # Decision
        if delta_sharpe > 0.1 and p_val < 0.05:
            decision = "RETAIN"
        elif delta_sharpe > 0:
            decision = "WATCH"
        else:
            decision = "REMOVE"

        result = ContributionResult(
            agent_name=agent.name,
            baseline_sharpe=baseline_sharpe,
            new_sharpe=new_sharpe,
            delta_sharpe=delta_sharpe,
            baseline_max_dd=baseline_dd,
            new_max_dd=new_dd,
            signal_disagreement_rate=disagree_rate,
            p_value=p_val,
            decision=decision,
            retained=decision in ("RETAIN", "WATCH"),
        )
        results.append(result)

        status = {"RETAIN": "✓ RETAIN", "WATCH": "⚠ WATCH", "REMOVE": "✗ REMOVE"}
        print(f"    ΔSharpe: {delta_sharpe:+.3f}  p={p_val:.4f}  disagree={disagree_rate:.1%}")
        print(f"    Decision: {status[decision]}")

    # Summary
    print("\n" + "=" * 70)
    print("  AGENT CONTRIBUTION SUMMARY")
    print("=" * 70)
    print(f"\n  {'Agent':<20} {'ΔSharpe':>10} {'p-value':>10} {'Decision':>10}")
    print("  " + "-" * 52)
    for r in results:
        print(f"  {r.agent_name:<20} {r.delta_sharpe:>+10.3f} {r.p_value:>10.4f} {r.decision:>10}")

    retained = [r for r in results if r.retained]
    removed = [r for r in results if not r.retained]

    print(f"\n  Retained: {len(retained)} agents")
    print(f"  Removed: {len(removed)} agents")
    print(f"  Final system: Technical + Fundamental + {', '.join(r.agent_name for r in retained)}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output = {
        "baseline_sharpe": baseline_sharpe,
        "baseline_max_dd": baseline_dd,
        "agents": [asdict(r) for r in results],
        "retained_agents": [r.agent_name for r in retained],
        "removed_agents": [r.agent_name for r in removed],
    }
    filepath = os.path.join(output_dir, "agent_contribution.json")
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {filepath}")

    return results


if __name__ == "__main__":
    test_all_agents()
