"""
Execution Agent
===============
Translates TradeProposals into Trading 212 API JSON payloads.

Responsibilities:
  - Validate-before-actuate (pre-flight checks)
  - Negative quantity enforcement on sells
  - Compliance guard integration (absolute veto)
  - Order lifecycle tracking
  - Slippage estimation & post-fill reporting
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from src.agents.decision.decision_fusion import TradeAction, TradeProposal
from src.agents.decision.risk import PortfolioRiskState
from src.compliance.guard import ComplianceGuard, VetoResult
from src.core.client import Trading212Client, Trading212APIError
from src.core.virtual_account import VirtualSubAccount, VirtualAccountManager

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING_VALIDATION = auto()
    VALIDATED = auto()
    COMPLIANCE_VETOED = auto()
    SUBMITTED = auto()
    FILLED = auto()
    PARTIALLY_FILLED = auto()
    REJECTED_BY_API = auto()
    CANCELLED = auto()
    FAILED = auto()


@dataclass
class OrderTicket:
    """Full lifecycle record of a single order."""
    ticket_id: str = ""
    proposal: TradeProposal = field(default_factory=TradeProposal)
    status: OrderStatus = OrderStatus.PENDING_VALIDATION

    # API payload sent
    payload: dict[str, Any] = field(default_factory=dict)

    # API response
    api_order_id: int | None = None
    api_response: dict[str, Any] = field(default_factory=dict)

    # Validation / veto details
    validation_errors: list[str] = field(default_factory=list)
    veto: VetoResult | None = None

    # Post-fill
    fill_price: float | None = None
    fill_quantity: float | None = None
    slippage: float = 0.0          # fill_price - expected_price

    # Timestamps
    created_at: float = field(default_factory=time.time)
    submitted_at: float | None = None
    filled_at: float | None = None


class ExecutionAgent:
    """
    Converts trade proposals into live API orders with full pre-flight
    validation and compliance gating.
    """

    def __init__(
        self,
        client: Trading212Client,
        compliance: ComplianceGuard,
        virtual_account: VirtualSubAccount | None = None,
        account_manager: VirtualAccountManager | None = None,
    ) -> None:
        self._client = client
        self._compliance = compliance
        self._virtual_account = virtual_account
        self._account_manager = account_manager
        self._order_counter = 0
        self._active_tickets: dict[str, OrderTicket] = {}

    # ── main entry point ──────────────────────────────────────────────

    async def execute(
        self,
        proposal: TradeProposal,
        portfolio: PortfolioRiskState,
        pending_count_for_instrument: int = 0,
        is_uk_equity: bool = True,
    ) -> OrderTicket:
        """
        Full execution pipeline:
          1. Pre-flight validation
          2. Compliance gate (FCA-grade, absolute veto)
          3. Payload construction
          4. API submission
          5. Post-fill recording
        """
        self._order_counter += 1
        ticket = OrderTicket(
            ticket_id=f"ORD-{self._order_counter:06d}",
            proposal=proposal,
        )

        # ── Step 1: Pre-flight validation ─────────────────────────────
        errors = self._validate_before_actuate(proposal)
        if errors:
            ticket.status = OrderStatus.FAILED
            ticket.validation_errors = errors
            logger.warning(
                "Ticket %s FAILED pre-flight: %s", ticket.ticket_id, errors,
            )
            self._active_tickets[ticket.ticket_id] = ticket
            return ticket

        ticket.status = OrderStatus.VALIDATED

        # ── Step 2: Compliance gate (absolute veto) ───────────────────
        # 新版：使用 pre_trade_check 整合虛擬帳戶驗證
        # 舊版：使用 validate_order 作為降級方案
        side = "buy" if proposal.action == TradeAction.BUY else "sell"

        if self._virtual_account is not None:
            veto = self._compliance.pre_trade_check(
                ticker=proposal.ticker,
                quantity=proposal.quantity,
                estimated_price=proposal.current_price,
                account=self._virtual_account,
                side=side,
                reference_price=proposal.current_price,
                expected_roi=(
                    proposal.risk.risk_reward_ratio * 0.01
                    if proposal.risk.risk_reward_ratio > 0 else 0
                ),
                is_uk_equity=is_uk_equity,
                pending_count_for_instrument=pending_count_for_instrument,
            )
        else:
            veto = self._compliance.validate_order(
                order_value=proposal.estimated_value,
                nav=portfolio.total_nav,
                pending_count_for_instrument=pending_count_for_instrument,
                ticker=proposal.ticker,
                side=side,
                reference_price=proposal.current_price,
                order_price=proposal.current_price,
                expected_roi=(
                    proposal.risk.risk_reward_ratio * 0.01
                    if proposal.risk.risk_reward_ratio > 0 else 0
                ),
                is_uk_equity=is_uk_equity,
            )
        ticket.veto = veto

        if not veto.is_approved:
            ticket.status = OrderStatus.COMPLIANCE_VETOED
            logger.warning(
                "Ticket %s VETOED by compliance: %s", ticket.ticket_id, veto.detail,
            )
            self._active_tickets[ticket.ticket_id] = ticket
            return ticket

        # ── Step 3: Build API payload ─────────────────────────────────
        payload = self._build_payload(proposal)
        ticket.payload = payload

        # ── Step 4: Submit to API ─────────────────────────────────────
        try:
            ticket.submitted_at = time.time()

            if proposal.risk.stop_loss_price > 0 and proposal.risk.take_profit_price > 0:
                response = await self._submit_with_brackets(payload, proposal)
            else:
                response = await self._client.place_order(payload)

            ticket.api_response = response
            ticket.api_order_id = response.get("id")
            ticket.status = OrderStatus.SUBMITTED

            # 通知合規模組記錄已下單（更新日成交量和速率追蹤）
            self._compliance.record_order_placed(proposal.estimated_value)

            logger.info(
                "Ticket %s SUBMITTED — API order ID: %s",
                ticket.ticket_id, ticket.api_order_id,
            )

        except Trading212APIError as e:
            ticket.status = OrderStatus.REJECTED_BY_API
            ticket.api_response = {"error": e.body, "status": e.status}
            logger.error(
                "Ticket %s REJECTED by API: HTTP %d — %s",
                ticket.ticket_id, e.status, e.body,
            )

        except Exception as e:
            ticket.status = OrderStatus.FAILED
            ticket.validation_errors.append(f"Unexpected error: {e}")
            logger.exception("Ticket %s FAILED unexpectedly", ticket.ticket_id)

        self._active_tickets[ticket.ticket_id] = ticket
        return ticket

    # ── post-fill tracking ────────────────────────────────────────────

    async def check_fill(self, ticket: OrderTicket) -> OrderTicket:
        """Poll the order status and record fill details."""
        if ticket.api_order_id is None:
            return ticket

        side = "buy" if ticket.proposal.action == TradeAction.BUY else "sell"

        try:
            # Check order history for fill
            history = await self._client.order_history()
            items = history.get("items", [])
            for item in items:
                if item.get("id") == ticket.api_order_id:
                    status = item.get("status", "").lower()
                    if status in ("filled", "completed"):
                        ticket.status = OrderStatus.FILLED
                        ticket.fill_price = float(item.get("fillPrice", 0))
                        ticket.fill_quantity = float(item.get("filledQuantity", 0))
                        ticket.filled_at = time.time()

                        # Calculate slippage
                        if ticket.fill_price and ticket.proposal.current_price > 0:
                            ticket.slippage = (
                                ticket.fill_price - ticket.proposal.current_price
                            )

                        # ── Virtual account bookkeeping ──────
                        if self._virtual_account is not None and ticket.fill_price:
                            qty = ticket.fill_quantity or ticket.proposal.quantity
                            if ticket.proposal.action == TradeAction.SELL:
                                qty = -abs(qty)
                            self._virtual_account.record_trade(
                                ticker=ticket.proposal.ticker,
                                quantity=qty,
                                execution_price=ticket.fill_price,
                            )
                            # Persist to disk
                            if self._account_manager is not None:
                                self._account_manager.save_state()

                        # 通知合規模組成交完成（釋放掛單槽位）
                        self._compliance.record_fill(
                            ticker=ticket.proposal.ticker,
                            side=side,
                        )

                        logger.info(
                            "Ticket %s FILLED @ %.4f (slippage: %.4f)",
                            ticket.ticket_id, ticket.fill_price, ticket.slippage,
                        )
                    elif status == "cancelled":
                        ticket.status = OrderStatus.CANCELLED
                        self._compliance.record_cancellation(
                            ticker=ticket.proposal.ticker,
                        )
                    break
        except Exception:
            logger.exception("Failed to check fill for ticket %s", ticket.ticket_id)

        return ticket

    # ── cancel ────────────────────────────────────────────────────────

    async def cancel(self, ticket: OrderTicket) -> OrderTicket:
        if ticket.api_order_id is not None:
            try:
                await self._client.cancel_order(ticket.api_order_id)
                ticket.status = OrderStatus.CANCELLED
                self._compliance.record_cancellation(
                    ticker=ticket.proposal.ticker,
                )
                logger.info("Ticket %s CANCELLED", ticket.ticket_id)
            except Trading212APIError as e:
                logger.error("Failed to cancel %s: %s", ticket.ticket_id, e)
        return ticket

    # ── batch execution ────────────────────────────────────────────

    async def execute_batch(
        self,
        proposals: list[TradeProposal],
        portfolio: PortfolioRiskState,
        pending_counts: dict[str, int] | None = None,
        is_uk_equity: bool = True,
    ) -> list[OrderTicket]:
        """
        依序執行多個交易提案，跳過不可行的提案。

        Parameters
        ----------
        proposals : 經 DecisionFusionAgent 排序後的提案清單
        portfolio : 當前投資組合風險快照
        pending_counts : ticker → 現有掛單數
        is_uk_equity : 是否為英國股票（影響印花稅）
        """
        pending_counts = pending_counts or {}
        tickets: list[OrderTicket] = []

        for proposal in proposals:
            if not proposal.is_actionable:
                continue

            pending = pending_counts.get(proposal.ticker, 0)
            ticket = await self.execute(
                proposal=proposal,
                portfolio=portfolio,
                pending_count_for_instrument=pending,
                is_uk_equity=is_uk_equity,
            )
            tickets.append(ticket)

            # 若合規觸發 Kill Switch，中斷批次執行
            if self._compliance.kill_switch_triggered:
                logger.critical(
                    "[ExecutionAgent] Kill switch 已觸發，中斷批次執行。"
                )
                break

        submitted = sum(1 for t in tickets if t.status == OrderStatus.SUBMITTED)
        vetoed = sum(1 for t in tickets if t.status == OrderStatus.COMPLIANCE_VETOED)
        failed = sum(1 for t in tickets if t.status in (OrderStatus.FAILED, OrderStatus.REJECTED_BY_API))
        logger.info(
            "[ExecutionAgent] 批次執行完成: %d 提案 → %d 已送出, %d 被否決, %d 失敗",
            len(proposals), submitted, vetoed, failed,
        )
        return tickets

    # ── pre-flight validation ─────────────────────────────────────────

    def _validate_before_actuate(self, proposal: TradeProposal) -> list[str]:
        """
        Validate-before-actuate checks. Returns list of errors (empty = valid).
        These are sanity checks that catch obvious mistakes before the order
        even reaches the compliance layer.
        """
        errors: list[str] = []

        if not proposal.is_actionable:
            errors.append("Proposal is not actionable (HOLD or zero quantity)")
            return errors

        if not proposal.ticker:
            errors.append("Missing ticker/ISIN")

        if proposal.current_price <= 0:
            errors.append(f"Invalid current price: {proposal.current_price}")

        if proposal.quantity <= 0:
            errors.append(f"Invalid quantity: {proposal.quantity}")

        if proposal.estimated_value <= 0:
            errors.append(f"Invalid estimated value: {proposal.estimated_value}")

        # Sanity: estimated value should roughly match price * qty
        expected_value = proposal.quantity * proposal.current_price
        if expected_value > 0:
            deviation = abs(proposal.estimated_value - expected_value) / expected_value
            if deviation > 0.05:  # 5% tolerance
                errors.append(
                    f"Value mismatch: estimated={proposal.estimated_value:.2f} "
                    f"vs computed={expected_value:.2f} (deviation={deviation:.1%})"
                )

        # Stop-loss / take-profit sanity (if set)
        sl = proposal.risk.stop_loss_price
        tp = proposal.risk.take_profit_price
        if proposal.action == TradeAction.BUY:
            if sl > 0 and sl >= proposal.current_price:
                errors.append(
                    f"Stop-loss {sl:.2f} >= current price {proposal.current_price:.2f} "
                    f"for a BUY order"
                )
            if tp > 0 and tp <= proposal.current_price:
                errors.append(
                    f"Take-profit {tp:.2f} <= current price {proposal.current_price:.2f} "
                    f"for a BUY order"
                )
        elif proposal.action == TradeAction.SELL:
            if sl > 0 and sl <= proposal.current_price:
                errors.append(
                    f"Stop-loss {sl:.2f} <= current price {proposal.current_price:.2f} "
                    f"for a SELL order"
                )
            if tp > 0 and tp >= proposal.current_price:
                errors.append(
                    f"Take-profit {tp:.2f} >= current price {proposal.current_price:.2f} "
                    f"for a SELL order"
                )

        return errors

    # ── payload construction ──────────────────────────────────────────

    def _build_payload(self, proposal: TradeProposal) -> dict[str, Any]:
        """
        Build Trading 212 API JSON payload.
        CRITICAL: sells require negative quantity.
        """
        quantity = proposal.quantity
        if proposal.action == TradeAction.SELL:
            quantity = -abs(quantity)   # T212 convention: negative = sell

        # T212 API requires its internal ticker code, NOT standard market ticker
        api_ticker = proposal.t212_ticker or proposal.ticker

        payload: dict[str, Any] = {
            "ticker": api_ticker,
            "quantity": quantity,
        }

        return payload

    async def _submit_with_brackets(
        self, base_payload: dict[str, Any], proposal: TradeProposal,
    ) -> dict[str, Any]:
        """
        Submit market order, then attach stop-loss and take-profit
        as separate limit/stop orders.
        """
        api_ticker = proposal.t212_ticker or proposal.ticker

        # Main order
        result = await self._client.place_order(base_payload)

        # Attach stop-loss
        if proposal.risk.stop_loss_price > 0:
            sl_qty = -abs(proposal.quantity)  # stop = sell direction
            try:
                await self._client.place_stop_order({
                    "ticker": api_ticker,
                    "quantity": sl_qty,
                    "stopPrice": proposal.risk.stop_loss_price,
                })
            except Trading212APIError:
                logger.warning("Failed to attach stop-loss for %s", proposal.ticker)

        # Attach take-profit
        if proposal.risk.take_profit_price > 0:
            tp_qty = -abs(proposal.quantity)
            try:
                await self._client.place_limit_order({
                    "ticker": api_ticker,
                    "quantity": tp_qty,
                    "limitPrice": proposal.risk.take_profit_price,
                })
            except Trading212APIError:
                logger.warning("Failed to attach take-profit for %s", proposal.ticker)

        return result

    # ── accessors ─────────────────────────────────────────────────────

    @property
    def active_tickets(self) -> dict[str, OrderTicket]:
        return self._active_tickets

    def get_ticket(self, ticket_id: str) -> OrderTicket | None:
        return self._active_tickets.get(ticket_id)
