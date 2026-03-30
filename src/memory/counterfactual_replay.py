"""
Counterfactual Replay Environment
=================================
Replays historical trades with altered parameters to answer:
  "What would have happened if we had used different settings?"

Integrates with:
  - Adaptive-OPRO (evaluating candidate parameters on past data)
  - Failure Attribution (testing if a corrective action would fix a loss)
  - Episodic Memory (storing counterfactual outcomes alongside actuals)

Friction model includes:
  - UK Stamp Duty (0.5% on purchases of UK equities)
  - PTM Levy (£1 on transactions > £10,000)
  - Bid-ask spread estimation
  - Slippage model (based on historical slippage distribution)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrictionModel:
    """Physical friction costs for UK equity trading."""
    stamp_duty_rate: float = 0.005        # 0.5% on UK equity purchases
    ptm_levy_threshold: float = 10_000.0  # PTM levy applies above this
    ptm_levy_amount: float = 1.0          # flat £1
    estimated_spread_bps: float = 5.0     # basis points half-spread
    slippage_bps: float = 3.0             # additional execution slippage

    def total_cost(self, value: float, is_buy: bool, is_uk_equity: bool = True) -> float:
        """Calculate total friction cost for a trade."""
        cost = 0.0

        # Stamp duty (only on purchases of UK equities)
        if is_buy and is_uk_equity:
            cost += value * self.stamp_duty_rate

        # PTM levy
        if value > self.ptm_levy_threshold:
            cost += self.ptm_levy_amount

        # Spread cost (half-spread on entry and exit)
        cost += value * (self.estimated_spread_bps / 10_000)

        # Slippage
        cost += value * (self.slippage_bps / 10_000)

        return cost


@dataclass
class ReplayTrade:
    """One simulated trade in a counterfactual replay."""
    ticker: str = ""
    action: str = ""           # "BUY" or "SELL"
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    entry_idx: int = 0         # bar index in the price series
    exit_idx: int = 0

    # Outcomes
    gross_pnl: float = 0.0
    friction_cost: float = 0.0
    net_pnl: float = 0.0
    roi: float = 0.0

    # Stop/TP triggers
    stopped_out: bool = False
    took_profit: bool = False
    timed_out: bool = False


@dataclass
class ReplayResult:
    """Aggregate result of a counterfactual replay session."""
    session_id: str = ""
    parameter_label: str = ""
    timestamp: float = field(default_factory=time.time)

    trades: list[ReplayTrade] = field(default_factory=list)
    total_gross_pnl: float = 0.0
    total_friction: float = 0.0
    total_net_pnl: float = 0.0
    total_roi: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    trade_count: int = 0

    @property
    def summary(self) -> str:
        return (
            f"Replay '{self.parameter_label}': "
            f"{self.trade_count} trades, "
            f"net PnL={self.total_net_pnl:.2f}, "
            f"ROI={self.total_roi:.2%}, "
            f"win={self.win_rate:.0%}, "
            f"Sharpe={self.sharpe_ratio:.2f}, "
            f"MaxDD={self.max_drawdown:.2%}, "
            f"friction={self.total_friction:.2f}"
        )


class CounterfactualReplayEngine:
    """
    Replays a price series with configurable parameters and full
    friction modelling to evaluate "what if" scenarios.
    """

    def __init__(
        self,
        friction: FrictionModel | None = None,
        initial_capital: float = 100_000.0,
    ) -> None:
        self._friction = friction or FrictionModel()
        self._initial_capital = initial_capital
        self._counter = 0

    def replay(
        self,
        price_series: np.ndarray,
        signals: np.ndarray,
        parameters: dict[str, float],
        ticker: str = "",
        is_uk_equity: bool = True,
        max_holding_bars: int = 50,
    ) -> ReplayResult:
        """
        Run a counterfactual replay.

        Parameters
        ----------
        price_series : array of close prices
        signals : array of signal scores (same length as prices),
                  positive = buy signal, negative = sell signal
        parameters : dict with keys from Adaptive-OPRO parameter space
        ticker : instrument identifier (for logging)
        is_uk_equity : whether UK stamp duty applies
        max_holding_bars : max bars to hold before forced exit
        """
        self._counter += 1
        result = ReplayResult(
            session_id=f"REPLAY-{self._counter:06d}",
            parameter_label=f"params-{self._counter}",
        )

        if len(price_series) < 20 or len(signals) != len(price_series):
            logger.warning("Invalid replay input: prices=%d, signals=%d",
                           len(price_series), len(signals))
            return result

        min_conf = parameters.get("min_confidence_to_trade", 0.3)
        min_buy = parameters.get("min_buy_score", 0.3)
        max_sell = parameters.get("max_sell_score", -0.3)
        atr_stop = parameters.get("atr_stop_multiplier", 2.0)
        atr_tp = parameters.get("atr_tp_multiplier", 3.0)
        kelly_scale = parameters.get("half_kelly_scaling", 0.5)

        capital = self._initial_capital
        position: ReplayTrade | None = None

        # Compute rolling ATR(14) for the series
        atrs = self._rolling_atr(price_series, period=14)

        for i in range(20, len(price_series)):
            price = price_series[i]
            signal = signals[i]
            atr = atrs[i] if atrs[i] > 0 else price * 0.02

            # ── If in position: check stop/TP/timeout ─────────────────
            if position is not None:
                bars_held = i - position.entry_idx

                if position.action == "BUY":
                    stop_price = position.entry_price - atr_stop * atr
                    tp_price = position.entry_price + atr_tp * atr

                    if price <= stop_price:
                        position = self._close_position(
                            position, price, i, capital, is_uk_equity,
                        )
                        position.stopped_out = True
                        result.trades.append(position)
                        capital += position.net_pnl
                        position = None
                        continue

                    if price >= tp_price:
                        position = self._close_position(
                            position, price, i, capital, is_uk_equity,
                        )
                        position.took_profit = True
                        result.trades.append(position)
                        capital += position.net_pnl
                        position = None
                        continue

                    if bars_held >= max_holding_bars:
                        position = self._close_position(
                            position, price, i, capital, is_uk_equity,
                        )
                        position.timed_out = True
                        result.trades.append(position)
                        capital += position.net_pnl
                        position = None
                        continue

                # Exit on opposite signal
                if signal <= max_sell and position.action == "BUY":
                    position = self._close_position(
                        position, price, i, capital, is_uk_equity,
                    )
                    result.trades.append(position)
                    capital += position.net_pnl
                    position = None
                    continue

            # ── If flat: check for entry ──────────────────────────────
            if position is None and signal >= min_buy:
                # Position size: kelly-scaled fraction of capital
                pos_size = capital * kelly_scale * 0.1  # conservative
                if pos_size > capital * 0.05:
                    pos_size = capital * 0.05

                quantity = pos_size / price if price > 0 else 0

                entry_friction = self._friction.total_cost(
                    pos_size, is_buy=True, is_uk_equity=is_uk_equity,
                )

                position = ReplayTrade(
                    ticker=ticker,
                    action="BUY",
                    entry_price=price,
                    quantity=quantity,
                    entry_idx=i,
                    friction_cost=entry_friction,
                )

        # Close any open position at end
        if position is not None:
            position = self._close_position(
                position, price_series[-1], len(price_series) - 1,
                capital, is_uk_equity,
            )
            position.timed_out = True
            result.trades.append(position)

        # ── Aggregate results ─────────────────────────────────────────
        self._compute_aggregate(result)
        logger.info("Counterfactual: %s", result.summary)
        return result

    def compare_parameters(
        self,
        price_series: np.ndarray,
        signals: np.ndarray,
        param_sets: list[dict[str, float]],
        labels: list[str] | None = None,
        **kwargs: Any,
    ) -> list[ReplayResult]:
        """
        Run multiple replays with different parameter sets for comparison.
        Used by Adaptive-OPRO to evaluate candidate configurations.
        """
        labels = labels or [f"set-{i}" for i in range(len(param_sets))]
        results = []
        for params, label in zip(param_sets, labels):
            result = self.replay(price_series, signals, params, **kwargs)
            result.parameter_label = label
            results.append(result)

        # Log comparison
        results.sort(key=lambda r: r.total_net_pnl, reverse=True)
        logger.info("Parameter comparison results:")
        for r in results:
            logger.info("  %s", r.summary)

        return results

    # ── internal helpers ──────────────────────────────────────────────

    def _close_position(
        self,
        trade: ReplayTrade,
        exit_price: float,
        exit_idx: int,
        capital: float,
        is_uk_equity: bool,
    ) -> ReplayTrade:
        trade.exit_price = exit_price
        trade.exit_idx = exit_idx

        trade_value = trade.quantity * exit_price
        exit_friction = self._friction.total_cost(
            trade_value, is_buy=False, is_uk_equity=is_uk_equity,
        )
        trade.friction_cost += exit_friction

        if trade.action == "BUY":
            trade.gross_pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            trade.gross_pnl = (trade.entry_price - exit_price) * trade.quantity

        trade.net_pnl = trade.gross_pnl - trade.friction_cost

        entry_value = trade.entry_price * trade.quantity
        trade.roi = trade.net_pnl / entry_value if entry_value > 0 else 0

        return trade

    def _compute_aggregate(self, result: ReplayResult) -> None:
        if not result.trades:
            return

        result.trade_count = len(result.trades)
        result.total_gross_pnl = sum(t.gross_pnl for t in result.trades)
        result.total_friction = sum(t.friction_cost for t in result.trades)
        result.total_net_pnl = sum(t.net_pnl for t in result.trades)

        entry_values = [t.entry_price * t.quantity for t in result.trades]
        total_invested = sum(entry_values) if entry_values else 1
        result.total_roi = result.total_net_pnl / total_invested if total_invested > 0 else 0

        wins = [t for t in result.trades if t.net_pnl > 0]
        result.win_rate = len(wins) / len(result.trades) if result.trades else 0

        # Sharpe ratio (annualised from trade returns)
        rois = np.array([t.roi for t in result.trades])
        if len(rois) > 1 and np.std(rois) > 0:
            result.sharpe_ratio = float(np.mean(rois) / np.std(rois) * np.sqrt(252))
        else:
            result.sharpe_ratio = 0.0

        # Max drawdown
        equity = [self._initial_capital]
        for t in result.trades:
            equity.append(equity[-1] + t.net_pnl)
        equity_arr = np.array(equity)
        running_max = np.maximum.accumulate(equity_arr)
        drawdowns = (equity_arr - running_max) / running_max
        result.max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0

    @staticmethod
    def _rolling_atr(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Simplified ATR from close prices only (TR ≈ |close[i] - close[i-1]|)."""
        atrs = np.zeros_like(prices)
        for i in range(1, len(prices)):
            start = max(0, i - period)
            true_ranges = np.abs(np.diff(prices[start:i + 1]))
            if len(true_ranges) > 0:
                atrs[i] = np.mean(true_ranges)
        return atrs
