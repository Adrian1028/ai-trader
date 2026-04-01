"""
Decision Fusion Agent
=====================
Consumes a MarketView (from intelligence layer) and a PortfolioRiskState,
runs the Risk Agent, and produces a concrete TradeProposal — or decides
to hold.

This is the single decision point that bridges analysis → execution.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from typing import TYPE_CHECKING

from src.agents.decision.correlation_filter import CorrelationFilter
from src.agents.decision.portfolio_optimizer import (
    BlackLittermanOptimizer,
    RiskParityAllocator,
)
from src.agents.decision.risk import (
    PortfolioRiskState,
    RiskAgent,
    RiskEnvelope,
    RiskVerdict,
)
from src.agents.decision.stress_tester import StressTester
from src.agents.intelligence.orchestrator import MarketView
from src.core.base_agent import SignalDirection

if TYPE_CHECKING:
    from src.core.virtual_account import VirtualSubAccount

logger = logging.getLogger(__name__)


class TradeAction(Enum):
    BUY = auto()
    SELL = auto()
    HOLD = auto()


@dataclass
class TradeProposal:
    """Concrete, actionable output from the Decision layer."""
    action: TradeAction = TradeAction.HOLD
    isin: str = ""
    ticker: str = ""           # standard market ticker (AAPL)
    t212_ticker: str = ""      # Trading 212 internal code (AAPL_US_EQ)

    # Quantities (positive for buys; caller converts to negative for sells)
    quantity: float = 0.0
    estimated_value: float = 0.0
    current_price: float = 0.0

    # Risk envelope attached
    risk: RiskEnvelope = field(default_factory=RiskEnvelope)

    # Decision reasoning chain
    reasoning: str = ""
    intelligence_summary: str = ""

    # Metadata
    timestamp: float = field(default_factory=time.time)

    @property
    def is_actionable(self) -> bool:
        return self.action in (TradeAction.BUY, TradeAction.SELL) and self.quantity > 0


# Minimum thresholds to act
_MIN_CONFIDENCE_TO_TRADE = 0.05
_MIN_SCORE_TO_BUY = 0.10
_MAX_SCORE_TO_SELL = -0.10


class DecisionFusionAgent:
    """
    Deterministic decision gate.
    Combines intelligence signals with risk constraints to produce
    trade proposals.
    """

    def __init__(
        self,
        risk_agent: RiskAgent,
        correlation_filter: CorrelationFilter | None = None,
        bl_optimizer: BlackLittermanOptimizer | None = None,
        rp_allocator: RiskParityAllocator | None = None,
        stress_tester: StressTester | None = None,
        min_confidence: float = _MIN_CONFIDENCE_TO_TRADE,
        min_buy_score: float = _MIN_SCORE_TO_BUY,
        max_sell_score: float = _MAX_SCORE_TO_SELL,
    ) -> None:
        self._risk = risk_agent
        self._corr_filter = correlation_filter or CorrelationFilter()
        self._bl = bl_optimizer or BlackLittermanOptimizer()
        self._rp = rp_allocator or RiskParityAllocator()
        self._stress = stress_tester or StressTester()
        self._min_conf = min_confidence
        self._min_buy = min_buy_score
        self._max_sell = max_sell_score

    def decide(
        self,
        view: MarketView,
        portfolio: PortfolioRiskState,
        current_price: float,
        returns: np.ndarray,
        atr: float | None = None,
        existing_position_value: float = 0.0,
        virtual_account: "VirtualSubAccount | None" = None,
    ) -> TradeProposal:
        """
        Produce a trade proposal from a fused MarketView.

        Parameters
        ----------
        view : MarketView from the intelligence orchestrator
        portfolio : current portfolio risk snapshot
        current_price : latest price of the instrument
        returns : historical daily returns array
        atr : average true range (optional, from TechnicalAgent)
        existing_position_value : current value of any existing position in this ISIN
        """
        proposal = TradeProposal(
            isin=view.isin,
            ticker=view.ticker,
            t212_ticker=view.t212_ticker,
            current_price=current_price,
            intelligence_summary=view.summary,
        )

        # ── Gate 1: confidence threshold ──────────────────────────────
        if view.fused_confidence < self._min_conf:
            proposal.action = TradeAction.HOLD
            proposal.reasoning = (
                f"Confidence {view.fused_confidence:.2f} below threshold "
                f"{self._min_conf:.2f} — holding."
            )
            logger.info("Decision HOLD for %s: low confidence", view.ticker)
            return proposal

        # ── Gate 2: directional threshold ─────────────────────────────
        direction: int
        if view.fused_score >= self._min_buy:
            direction = 1
            proposal.action = TradeAction.BUY
        elif view.fused_score <= self._max_sell:
            direction = -1
            if existing_position_value > 0:
                proposal.action = TradeAction.SELL
            else:
                proposal.action = TradeAction.HOLD
                proposal.reasoning = (
                    f"Bearish signal (score={view.fused_score:+.2f}) but no existing "
                    f"position to sell — holding."
                )
                return proposal
        else:
            proposal.action = TradeAction.HOLD
            proposal.reasoning = (
                f"Score {view.fused_score:+.2f} in neutral zone "
                f"[{self._max_sell}, {self._min_buy}] — holding."
            )
            return proposal

        # ── Extract MTF confluence score from technical agent ─────────
        confluence_score = 0.0
        for sig in view.signals:
            if sig.source == "technical" and sig.data:
                confluence_score = sig.data.get("confluence_score", 0.0)
                break

        # ── Extract regime from MarketView (Phase 5) ────────────────────
        regime = view.regime  # injected by system orchestrator, may be None

        # ── Gate 3: Risk evaluation ───────────────────────────────────
        if virtual_account is not None:
            # 帳戶感知模式：含現金緩衝、持倉上限、自動降階
            risk_envelope = self._risk.evaluate_with_account(
                direction=direction,
                current_price=current_price,
                returns=returns,
                atr=atr,
                confidence=view.fused_confidence,
                virtual_account=virtual_account,
                symbol=view.ticker,
                confluence_score=confluence_score,
                regime=regime,
            )
        else:
            risk_envelope = self._risk.evaluate(
                direction=direction,
                current_price=current_price,
                returns=returns,
                atr=atr,
                confidence=view.fused_confidence,
                portfolio=portfolio,
                confluence_score=confluence_score,
                regime=regime,
            )
        proposal.risk = risk_envelope

        if risk_envelope.verdict == RiskVerdict.REJECTED:
            proposal.action = TradeAction.HOLD
            proposal.reasoning = f"Risk rejected: {risk_envelope.reason}"
            logger.info("Decision HOLD for %s: risk rejected", view.ticker)
            return proposal

        # ── Compute final quantity ────────────────────────────────────
        if proposal.action == TradeAction.BUY:
            proposal.quantity = risk_envelope.suggested_quantity
            proposal.estimated_value = proposal.quantity * current_price
        elif proposal.action == TradeAction.SELL:
            # Sell existing position (up to full position)
            max_sell_qty = existing_position_value / current_price if current_price > 0 else 0
            # Risk agent may suggest partial sell
            proposal.quantity = min(risk_envelope.suggested_quantity, max_sell_qty)
            proposal.estimated_value = proposal.quantity * current_price

        proposal.reasoning = (
            f"Action={proposal.action.name} | "
            f"Score={view.fused_score:+.2f} Conf={view.fused_confidence:.2f} | "
            f"Qty={proposal.quantity:.4f} Value={proposal.estimated_value:.2f} | "
            f"SL={risk_envelope.stop_loss_price:.2f} "
            f"TP={risk_envelope.take_profit_price:.2f} "
            f"RR={risk_envelope.risk_reward_ratio:.2f} | "
            f"VaR95={risk_envelope.var_95:.2f} | "
            f"Risk: {risk_envelope.reason}"
        )

        logger.info("Decision %s for %s: %s", proposal.action.name, view.ticker, proposal.reasoning)
        return proposal

    def decide_batch(
        self,
        views: list[MarketView],
        portfolio: PortfolioRiskState,
        price_map: dict[str, float],
        returns_map: dict[str, np.ndarray],
        atr_map: dict[str, float | None] | None = None,
        position_map: dict[str, float] | None = None,
        virtual_account: "VirtualSubAccount | None" = None,
    ) -> list[TradeProposal]:
        """Evaluate multiple instruments and return ranked proposals."""
        atr_map = atr_map or {}
        position_map = position_map or {}
        proposals = []

        for view in views:
            price = price_map.get(view.ticker, 0.0)
            returns = returns_map.get(view.ticker, np.array([]))
            atr = atr_map.get(view.ticker)
            existing = position_map.get(view.isin, 0.0)

            if price <= 0:
                continue

            proposal = self.decide(
                view=view,
                portfolio=portfolio,
                current_price=price,
                returns=returns,
                atr=atr,
                existing_position_value=existing,
                virtual_account=virtual_account,
            )
            proposals.append(proposal)

        # Sort: actionable proposals first, then by absolute score
        proposals.sort(
            key=lambda p: (p.is_actionable, abs(p.risk.risk_reward_ratio)),
            reverse=True,
        )

        # ── Phase 6: Correlation filter ──────────────────────────
        # Get existing holding tickers from portfolio positions
        holding_tickers = [
            t for t in portfolio.positions.keys()
            if portfolio.positions[t] > 0
        ]
        self._corr_filter.filter_proposals(
            proposals=proposals,
            holding_tickers=holding_tickers,
            returns_map=returns_map,
        )

        # ── Phase 7: Portfolio optimization ──────────────────────
        buy_proposals = [p for p in proposals if p.action == TradeAction.BUY and p.quantity > 0]
        if len(buy_proposals) >= 2:
            self._apply_portfolio_optimization(
                buy_proposals, views, returns_map, portfolio,
            )

        # ── Phase 7b: Stress test gate ───────────────────────────
        actionable = [p for p in proposals if p.action in (TradeAction.BUY, TradeAction.SELL) and p.quantity > 0]
        if actionable:
            self._apply_stress_test_gate(actionable, returns_map, portfolio)

        return proposals

    # ── Portfolio optimization (Phase 7) ────────────────────────────

    def _apply_portfolio_optimization(
        self,
        buy_proposals: list[TradeProposal],
        views: list[MarketView],
        returns_map: dict[str, np.ndarray],
        portfolio: PortfolioRiskState,
    ) -> None:
        """
        Apply Black-Litterman optimization to redistribute capital
        among BUY proposals. Falls back to Risk Parity when average
        confidence is low.
        """
        tickers = [p.ticker for p in buy_proposals]
        valid_tickers = [t for t in tickers if t in returns_map and len(returns_map[t]) >= 30]

        if len(valid_tickers) < 2:
            return

        # Build covariance matrix
        cov_tickers, cov_matrix = self._corr_filter.get_covariance_matrix(
            returns_map, valid_tickers,
        )
        if len(cov_tickers) < 2 or cov_matrix.size == 0:
            return

        # Map views to AI return expectations
        view_map = {v.ticker: v for v in views}
        avg_confidence = 0.0
        view_returns_list: list[float] = []
        view_conf_list: list[float] = []

        for t in cov_tickers:
            v = view_map.get(t)
            if v is not None:
                view_returns_list.append(v.fused_score * 0.01)  # score → expected return
                view_conf_list.append(v.fused_confidence)
                avg_confidence += v.fused_confidence
            else:
                view_returns_list.append(0.0)
                view_conf_list.append(0.3)

        avg_confidence = avg_confidence / len(cov_tickers) if cov_tickers else 0.0

        n = len(cov_tickers)
        market_weights = np.full(n, 1.0 / n)  # equal weight as prior

        # Choose optimizer: BL if good confidence, Risk Parity as fallback
        if avg_confidence >= 0.4:
            optimal_weights = self._bl.optimize(
                market_weights=market_weights,
                cov_matrix=cov_matrix,
                view_returns=np.array(view_returns_list),
                view_confidence=np.array(view_conf_list),
            )
            method = "Black-Litterman"
        else:
            optimal_weights = self._rp.optimize(cov_matrix)
            method = "Risk Parity"

        # Redistribute total BUY capital according to optimal weights
        total_buy_value = sum(p.estimated_value for p in buy_proposals if p.ticker in cov_tickers)
        if total_buy_value <= 0:
            return

        ticker_weight = dict(zip(cov_tickers, optimal_weights))

        for proposal in buy_proposals:
            if proposal.ticker not in ticker_weight:
                continue
            w = ticker_weight[proposal.ticker]
            new_value = total_buy_value * w
            if proposal.current_price > 0:
                old_qty = proposal.quantity
                proposal.quantity = round(new_value / proposal.current_price, 6)
                proposal.estimated_value = proposal.quantity * proposal.current_price
                proposal.reasoning = (
                    f"[{method} w={w:.2%}] {proposal.reasoning}"
                )
                logger.debug(
                    "[PortfolioOpt] %s: qty %.4f → %.4f (weight=%.2%%)",
                    proposal.ticker, old_qty, proposal.quantity, w * 100,
                )

        logger.info(
            "[PortfolioOpt] Applied %s to %d proposals (avg_conf=%.2f)",
            method, len(cov_tickers), avg_confidence,
        )

    # ── Stress test gate (Phase 7b) ──────────────────────────────────

    def _apply_stress_test_gate(
        self,
        proposals: list[TradeProposal],
        returns_map: dict[str, np.ndarray],
        portfolio: PortfolioRiskState,
    ) -> None:
        """
        Run stress test on the proposed portfolio. If it fails,
        reduce all proposal quantities by the computed reduction factor.
        """
        # Build proposed portfolio weights (current + new)
        proposed_weights: dict[str, float] = {}
        nav = portfolio.total_nav if portfolio.total_nav > 0 else 1.0

        # Current holdings
        for ticker, value in portfolio.positions.items():
            proposed_weights[ticker] = value / nav

        # Add proposed trades
        for p in proposals:
            if p.action == TradeAction.BUY:
                proposed_weights[p.ticker] = proposed_weights.get(p.ticker, 0.0) + p.estimated_value / nav
            elif p.action == TradeAction.SELL:
                proposed_weights[p.ticker] = max(
                    0.0, proposed_weights.get(p.ticker, 0.0) - p.estimated_value / nav,
                )

        # Run stress test
        stress_result = self._stress.full_stress_test(proposed_weights, returns_map)

        if stress_result.passes_stress_test:
            return

        # Failed — compute reduction factor and scale down proposals
        factor = self._stress.compute_reduction_factor(stress_result)

        for p in proposals:
            old_qty = p.quantity
            p.quantity = round(p.quantity * factor, 6)
            p.estimated_value = p.quantity * p.current_price
            p.reasoning = (
                f"[STRESS {factor:.0%}] {stress_result.summary} | {p.reasoning}"
            )

        logger.warning(
            "[StressGate] Stress test FAILED: %s — reducing all proposals by %.0f%%",
            stress_result.summary, (1 - factor) * 100,
        )
