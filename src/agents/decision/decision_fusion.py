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
from src.agents.decision.risk import (
    PortfolioRiskState,
    RiskAgent,
    RiskEnvelope,
    RiskVerdict,
)
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
_MIN_CONFIDENCE_TO_TRADE = 0.30
_MIN_SCORE_TO_BUY = 0.3
_MAX_SCORE_TO_SELL = -0.3


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
        min_confidence: float = _MIN_CONFIDENCE_TO_TRADE,
        min_buy_score: float = _MIN_SCORE_TO_BUY,
        max_sell_score: float = _MAX_SCORE_TO_SELL,
    ) -> None:
        self._risk = risk_agent
        self._corr_filter = correlation_filter or CorrelationFilter()
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

        return proposals
