"""Unit tests for Execution Agent — full pipeline coverage."""
from __future__ import annotations

import pytest

from src.agents.decision.decision_fusion import TradeAction, TradeProposal
from src.agents.decision.risk import RiskEnvelope, RiskVerdict, PortfolioRiskState
from src.agents.execution.executor import ExecutionAgent, OrderTicket, OrderStatus
from src.compliance.guard import VetoResult, VetoReason
from src.core.virtual_account import VirtualSubAccount, VirtualPosition


# ═══════════════════════════════════════════════════════════════════
# Stub / Fake objects (no network, no file I/O)
# ═══════════════════════════════════════════════════════════════════

class FakeClient:
    """Stub client for unit tests (no network)."""
    def __init__(self, *, fail: bool = False, fill_price: float = 150.0):
        self._fail = fail
        self._fill_price = fill_price
        self.placed_orders: list[dict] = []
        self.placed_stops: list[dict] = []
        self.placed_limits: list[dict] = []
        self.cancelled: list[int] = []

    async def place_order(self, payload):
        self.placed_orders.append(payload)
        if self._fail:
            from src.core.client import Trading212APIError
            raise Trading212APIError(400, "Bad Request", "/equity/orders/market")
        return {"id": 1001}

    async def place_stop_order(self, payload):
        self.placed_stops.append(payload)
        return {"id": 1002}

    async def place_limit_order(self, payload):
        self.placed_limits.append(payload)
        return {"id": 1003}

    async def cancel_order(self, oid):
        self.cancelled.append(oid)
        return {}

    async def cancel_all_orders(self):
        return {}

    async def order_history(self):
        """Simulate a filled order."""
        return {"items": [
            {
                "id": 1001,
                "status": "filled",
                "fillPrice": self._fill_price,
                "filledQuantity": 10.0,
            },
        ]}


class FakeCompliance:
    """Configurable compliance stub."""
    def __init__(self, *, approve: bool = True, reason: str = ""):
        self._approve = approve
        self._reason = reason
        self.kill_switch_triggered = False
        self.placed_values: list[float] = []
        self.fills: list[dict] = []
        self.cancellations: list[str] = []

    def pre_trade_check(self, **kw):
        if not self._approve:
            return VetoResult(
                is_approved=False,
                reason=VetoReason.FAT_FINGER,
                detail=self._reason or "Compliance stub veto",
                severity="critical",
            )
        return VetoResult(is_approved=True)

    def validate_order(self, **kw):
        return self.pre_trade_check(**kw)

    def record_order_placed(self, order_value: float):
        self.placed_values.append(order_value)

    def record_fill(self, ticker: str = "", side: str = "buy"):
        self.fills.append({"ticker": ticker, "side": side})

    def record_cancellation(self, ticker: str = ""):
        self.cancellations.append(ticker)

    def record_order_cancelled(self):
        pass


def _make_proposal(
    action=TradeAction.BUY,
    ticker="AAPL",
    quantity=10.0,
    price=150.0,
    sl=140.0,
    tp=170.0,
) -> TradeProposal:
    risk = RiskEnvelope(
        verdict=RiskVerdict.APPROVED,
        stop_loss_price=sl,
        take_profit_price=tp,
        risk_reward_ratio=1.5,
        suggested_quantity=quantity,
    )
    return TradeProposal(
        action=action,
        ticker=ticker,
        quantity=quantity,
        current_price=price,
        estimated_value=quantity * price,
        risk=risk,
    )


def _make_portfolio(nav=100_000, cash=50_000, invested=50_000):
    return PortfolioRiskState(
        total_nav=nav,
        invested_value=invested,
        cash=cash,
        exposure_pct=invested / nav if nav > 0 else 0,
    )


# ═══════════════════════════════════════════════════════════════════
# Pre-flight validation (synchronous)
# ═══════════════════════════════════════════════════════════════════

class TestPreFlightValidation:
    def test_rejects_hold_action(self):
        ex = ExecutionAgent(FakeClient(), FakeCompliance())
        proposal = TradeProposal(action=TradeAction.HOLD, quantity=10, ticker="AAPL")
        errors = ex._validate_before_actuate(proposal)
        assert errors

    def test_rejects_missing_ticker(self):
        ex = ExecutionAgent(FakeClient(), FakeCompliance())
        proposal = TradeProposal(
            action=TradeAction.BUY, quantity=10, ticker="",
            current_price=150.0, estimated_value=1500.0,
        )
        errors = ex._validate_before_actuate(proposal)
        assert any("ticker" in e.lower() or "isin" in e.lower() for e in errors)

    def test_rejects_zero_price(self):
        ex = ExecutionAgent(FakeClient(), FakeCompliance())
        proposal = TradeProposal(
            action=TradeAction.BUY, quantity=10, ticker="AAPL",
            current_price=0, estimated_value=1500.0,
        )
        errors = ex._validate_before_actuate(proposal)
        assert any("price" in e.lower() for e in errors)

    def test_rejects_value_mismatch(self):
        ex = ExecutionAgent(FakeClient(), FakeCompliance())
        proposal = TradeProposal(
            action=TradeAction.BUY, quantity=10, ticker="AAPL",
            current_price=150.0, estimated_value=3000.0,
        )
        errors = ex._validate_before_actuate(proposal)
        assert any("mismatch" in e.lower() for e in errors)

    def test_rejects_buy_stop_loss_above_price(self):
        ex = ExecutionAgent(FakeClient(), FakeCompliance())
        proposal = _make_proposal(sl=160.0, tp=200.0)  # SL above price
        errors = ex._validate_before_actuate(proposal)
        assert any("stop-loss" in e.lower() for e in errors)

    def test_rejects_buy_take_profit_below_price(self):
        ex = ExecutionAgent(FakeClient(), FakeCompliance())
        proposal = _make_proposal(sl=140.0, tp=130.0)  # TP below price
        errors = ex._validate_before_actuate(proposal)
        assert any("take-profit" in e.lower() for e in errors)

    def test_rejects_sell_stop_loss_below_price(self):
        ex = ExecutionAgent(FakeClient(), FakeCompliance())
        proposal = _make_proposal(
            action=TradeAction.SELL, sl=140.0, tp=130.0,
        )  # SL below price for SELL
        errors = ex._validate_before_actuate(proposal)
        assert any("stop-loss" in e.lower() for e in errors)

    def test_rejects_sell_take_profit_above_price(self):
        ex = ExecutionAgent(FakeClient(), FakeCompliance())
        proposal = _make_proposal(
            action=TradeAction.SELL, sl=160.0, tp=170.0,
        )  # TP above price for SELL
        errors = ex._validate_before_actuate(proposal)
        assert any("take-profit" in e.lower() for e in errors)

    def test_accepts_valid_buy(self):
        ex = ExecutionAgent(FakeClient(), FakeCompliance())
        proposal = _make_proposal()
        errors = ex._validate_before_actuate(proposal)
        assert errors == []

    def test_accepts_valid_sell(self):
        ex = ExecutionAgent(FakeClient(), FakeCompliance())
        proposal = _make_proposal(
            action=TradeAction.SELL, sl=160.0, tp=130.0,
        )
        errors = ex._validate_before_actuate(proposal)
        assert errors == []


# ═══════════════════════════════════════════════════════════════════
# Payload construction (synchronous)
# ═══════════════════════════════════════════════════════════════════

class TestPayloadConstruction:
    def test_sell_has_negative_quantity(self):
        ex = ExecutionAgent(FakeClient(), FakeCompliance())
        proposal = _make_proposal(action=TradeAction.SELL, quantity=10.5)
        payload = ex._build_payload(proposal)
        assert payload["quantity"] == -10.5
        assert payload["ticker"] == "AAPL"

    def test_buy_has_positive_quantity(self):
        ex = ExecutionAgent(FakeClient(), FakeCompliance())
        proposal = _make_proposal(quantity=5.0)
        payload = ex._build_payload(proposal)
        assert payload["quantity"] == 5.0


# ═══════════════════════════════════════════════════════════════════
# Full async execution pipeline
# ═══════════════════════════════════════════════════════════════════

class TestExecutePipeline:
    """Complete execute() flow with async."""

    @pytest.mark.asyncio
    async def test_successful_buy_order(self):
        """Happy path: proposal → validated → compliance OK → submitted."""
        client = FakeClient()
        compliance = FakeCompliance()
        ex = ExecutionAgent(client, compliance)

        proposal = _make_proposal()
        portfolio = _make_portfolio()

        ticket = await ex.execute(proposal, portfolio)

        assert ticket.status == OrderStatus.SUBMITTED
        assert ticket.ticket_id == "ORD-000001"
        assert ticket.api_order_id == 1001
        assert len(client.placed_orders) == 1
        assert client.placed_orders[0]["ticker"] == "AAPL"
        assert client.placed_orders[0]["quantity"] == 10.0
        # record_order_placed 應被呼叫
        assert len(compliance.placed_values) == 1
        assert compliance.placed_values[0] == 1500.0

    @pytest.mark.asyncio
    async def test_compliance_veto_blocks_order(self):
        """Compliance guard rejects → no API call."""
        client = FakeClient()
        compliance = FakeCompliance(approve=False, reason="超過肥手指上限")
        ex = ExecutionAgent(client, compliance)

        proposal = _make_proposal()
        portfolio = _make_portfolio()

        ticket = await ex.execute(proposal, portfolio)

        assert ticket.status == OrderStatus.COMPLIANCE_VETOED
        assert ticket.veto is not None
        assert not ticket.veto.is_approved
        assert len(client.placed_orders) == 0  # 不應呼叫 API
        assert len(compliance.placed_values) == 0

    @pytest.mark.asyncio
    async def test_api_rejection_handled(self):
        """API returns error → status is REJECTED_BY_API."""
        client = FakeClient(fail=True)
        compliance = FakeCompliance()
        ex = ExecutionAgent(client, compliance)

        proposal = _make_proposal()
        portfolio = _make_portfolio()

        ticket = await ex.execute(proposal, portfolio)

        assert ticket.status == OrderStatus.REJECTED_BY_API
        assert "error" in ticket.api_response

    @pytest.mark.asyncio
    async def test_bracket_orders_attached(self):
        """When stop-loss and take-profit are set, bracket orders are placed."""
        client = FakeClient()
        compliance = FakeCompliance()
        ex = ExecutionAgent(client, compliance)

        proposal = _make_proposal(sl=140.0, tp=170.0)
        portfolio = _make_portfolio()

        ticket = await ex.execute(proposal, portfolio)

        assert ticket.status == OrderStatus.SUBMITTED
        # 主單 + 停損 + 停利
        assert len(client.placed_orders) == 1
        assert len(client.placed_stops) == 1
        assert len(client.placed_limits) == 1
        # 停損和停利的方向都是賣出（負數量）
        assert client.placed_stops[0]["quantity"] < 0
        assert client.placed_stops[0]["stopPrice"] == 140.0
        assert client.placed_limits[0]["quantity"] < 0
        assert client.placed_limits[0]["limitPrice"] == 170.0

    @pytest.mark.asyncio
    async def test_no_brackets_when_no_sl_tp(self):
        """Without stop-loss/take-profit, only main order is placed."""
        client = FakeClient()
        compliance = FakeCompliance()
        ex = ExecutionAgent(client, compliance)

        proposal = _make_proposal(sl=0.0, tp=0.0)
        portfolio = _make_portfolio()

        ticket = await ex.execute(proposal, portfolio)

        assert ticket.status == OrderStatus.SUBMITTED
        assert len(client.placed_orders) == 1
        assert len(client.placed_stops) == 0
        assert len(client.placed_limits) == 0

    @pytest.mark.asyncio
    async def test_virtual_account_used_for_compliance(self):
        """When virtual account is provided, use pre_trade_check path."""
        client = FakeClient()
        compliance = FakeCompliance()
        acc = VirtualSubAccount(
            bot_id="test_bot",
            allocated_capital=100_000.0,
            available_cash=50_000.0,
        )
        ex = ExecutionAgent(client, compliance, virtual_account=acc)

        proposal = _make_proposal()
        portfolio = _make_portfolio()

        ticket = await ex.execute(proposal, portfolio)
        assert ticket.status == OrderStatus.SUBMITTED

    @pytest.mark.asyncio
    async def test_sell_payload_has_negative_qty(self):
        """Sell orders should send negative quantity to API."""
        client = FakeClient()
        compliance = FakeCompliance()
        ex = ExecutionAgent(client, compliance)

        proposal = _make_proposal(
            action=TradeAction.SELL, sl=160.0, tp=130.0,
        )
        portfolio = _make_portfolio()

        ticket = await ex.execute(proposal, portfolio)
        assert ticket.status == OrderStatus.SUBMITTED
        assert client.placed_orders[0]["quantity"] == -10.0

    @pytest.mark.asyncio
    async def test_ticket_counter_increments(self):
        """Each execute() increments the ticket counter."""
        client = FakeClient()
        compliance = FakeCompliance()
        ex = ExecutionAgent(client, compliance)

        t1 = await ex.execute(_make_proposal(), _make_portfolio())
        t2 = await ex.execute(_make_proposal(ticker="TSLA"), _make_portfolio())

        assert t1.ticket_id == "ORD-000001"
        assert t2.ticket_id == "ORD-000002"


# ═══════════════════════════════════════════════════════════════════
# Fill tracking (async)
# ═══════════════════════════════════════════════════════════════════

class TestCheckFill:
    @pytest.mark.asyncio
    async def test_fill_detected_and_slippage_computed(self):
        """check_fill picks up filled status and computes slippage."""
        client = FakeClient(fill_price=151.0)
        compliance = FakeCompliance()
        ex = ExecutionAgent(client, compliance)

        proposal = _make_proposal(price=150.0)
        portfolio = _make_portfolio()

        ticket = await ex.execute(proposal, portfolio)
        assert ticket.status == OrderStatus.SUBMITTED

        ticket = await ex.check_fill(ticket)
        assert ticket.status == OrderStatus.FILLED
        assert ticket.fill_price == 151.0
        assert ticket.fill_quantity == 10.0
        assert ticket.slippage == pytest.approx(1.0)
        assert ticket.filled_at is not None
        # record_fill 應被呼叫
        assert len(compliance.fills) == 1
        assert compliance.fills[0]["ticker"] == "AAPL"

    @pytest.mark.asyncio
    async def test_fill_updates_virtual_account(self):
        """After fill, virtual account balance should be updated."""
        # FakeClient.order_history returns filledQuantity=10.0
        client = FakeClient(fill_price=150.0)
        compliance = FakeCompliance()
        acc = VirtualSubAccount(
            bot_id="test_bot",
            allocated_capital=100_000.0,
            available_cash=50_000.0,
        )
        ex = ExecutionAgent(client, compliance, virtual_account=acc)

        proposal = _make_proposal(quantity=10.0, price=150.0)
        portfolio = _make_portfolio()

        ticket = await ex.execute(proposal, portfolio)
        ticket = await ex.check_fill(ticket)

        assert ticket.status == OrderStatus.FILLED
        # 買入 10 股 @ 150 = 1500，可用現金應減少
        assert acc.available_cash == pytest.approx(50_000.0 - 1500.0)
        assert acc.get_position_qty("AAPL") == pytest.approx(10.0)

    @pytest.mark.asyncio
    async def test_check_fill_without_order_id_noop(self):
        """If no api_order_id, check_fill should be a no-op."""
        ex = ExecutionAgent(FakeClient(), FakeCompliance())
        ticket = OrderTicket(ticket_id="ORD-999999")
        result = await ex.check_fill(ticket)
        assert result.status == OrderStatus.PENDING_VALIDATION

    @pytest.mark.asyncio
    async def test_cancelled_order_detected(self):
        """Cancelled orders should be detected and recorded."""
        client = FakeClient()
        # Override order_history to return a cancelled order
        async def cancelled_history():
            return {"items": [{"id": 1001, "status": "cancelled"}]}
        client.order_history = cancelled_history

        compliance = FakeCompliance()
        ex = ExecutionAgent(client, compliance)

        proposal = _make_proposal()
        ticket = await ex.execute(proposal, _make_portfolio())
        ticket = await ex.check_fill(ticket)

        assert ticket.status == OrderStatus.CANCELLED
        assert len(compliance.cancellations) == 1


# ═══════════════════════════════════════════════════════════════════
# Cancellation
# ═══════════════════════════════════════════════════════════════════

class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_submitted_order(self):
        client = FakeClient()
        compliance = FakeCompliance()
        ex = ExecutionAgent(client, compliance)

        proposal = _make_proposal()
        ticket = await ex.execute(proposal, _make_portfolio())
        assert ticket.status == OrderStatus.SUBMITTED

        ticket = await ex.cancel(ticket)
        assert ticket.status == OrderStatus.CANCELLED
        assert 1001 in client.cancelled
        assert len(compliance.cancellations) == 1
        assert compliance.cancellations[0] == "AAPL"


# ═══════════════════════════════════════════════════════════════════
# Batch execution
# ═══════════════════════════════════════════════════════════════════

class TestExecuteBatch:
    @pytest.mark.asyncio
    async def test_batch_executes_actionable_proposals(self):
        """Batch should only execute actionable proposals."""
        client = FakeClient()
        compliance = FakeCompliance()
        ex = ExecutionAgent(client, compliance)

        proposals = [
            _make_proposal(ticker="AAPL"),
            TradeProposal(action=TradeAction.HOLD, ticker="SKIP"),  # not actionable
            _make_proposal(ticker="TSLA", price=200.0, sl=190.0, tp=220.0),
        ]
        portfolio = _make_portfolio()

        tickets = await ex.execute_batch(proposals, portfolio)

        # HOLD 被跳過，只有 2 個被執行
        assert len(tickets) == 2
        assert all(t.status == OrderStatus.SUBMITTED for t in tickets)
        assert len(client.placed_orders) == 2

    @pytest.mark.asyncio
    async def test_batch_stops_on_kill_switch(self):
        """When kill switch triggers, batch should stop immediately."""
        client = FakeClient()
        compliance = FakeCompliance()
        ex = ExecutionAgent(client, compliance)

        proposals = [
            _make_proposal(ticker="A1"),
            _make_proposal(ticker="A2"),
            _make_proposal(ticker="A3"),
        ]
        portfolio = _make_portfolio()

        # 第一筆提交後觸發 kill switch
        original_execute = ex.execute

        async def execute_and_kill(*args, **kwargs):
            result = await original_execute(*args, **kwargs)
            compliance.kill_switch_triggered = True
            return result

        ex.execute = execute_and_kill

        tickets = await ex.execute_batch(proposals, portfolio)

        # 第一筆執行後 kill switch 觸發，後續應中斷
        assert len(tickets) == 1

    @pytest.mark.asyncio
    async def test_batch_empty_proposals(self):
        """Empty proposal list returns empty tickets."""
        ex = ExecutionAgent(FakeClient(), FakeCompliance())
        tickets = await ex.execute_batch([], _make_portfolio())
        assert tickets == []


# ═══════════════════════════════════════════════════════════════════
# Audit trail integration (smoke test)
# ═══════════════════════════════════════════════════════════════════

class TestAuditTrail:
    def test_record_creation(self):
        from src.agents.audit.audit_trail import AuditTrailAgent
        from src.agents.intelligence.orchestrator import MarketView
        from src.core.base_agent import SignalDirection

        audit = AuditTrailAgent(log_dir="logs/test_audit")
        view = MarketView(
            isin="US0378331005", ticker="AAPL",
            fused_score=0.8, fused_direction=SignalDirection.BUY,
            fused_confidence=0.75,
        )
        proposal = TradeProposal(
            action=TradeAction.BUY, ticker="AAPL", isin="US0378331005",
            quantity=5.0, estimated_value=750.0, current_price=150.0,
        )
        ticket = OrderTicket(
            ticket_id="ORD-000001", status=OrderStatus.SUBMITTED,
            api_order_id=12345,
        )

        record = audit.record_trade(view, proposal, ticket)
        assert record.record_id.startswith("AUD-")
        assert record.ticker == "AAPL"
        assert record.action == "BUY"

    def test_summary_stats_empty(self):
        from src.agents.audit.audit_trail import AuditTrailAgent
        audit = AuditTrailAgent(log_dir="logs/test_audit2")
        stats = audit.summary_stats()
        assert stats["total_trades"] == 0


# ═══════════════════════════════════════════════════════════════════
# Ticket accessors
# ═══════════════════════════════════════════════════════════════════

class TestTicketAccessors:
    @pytest.mark.asyncio
    async def test_active_tickets_tracked(self):
        ex = ExecutionAgent(FakeClient(), FakeCompliance())
        proposal = _make_proposal()
        ticket = await ex.execute(proposal, _make_portfolio())

        assert ticket.ticket_id in ex.active_tickets
        assert ex.get_ticket(ticket.ticket_id) is ticket
        assert ex.get_ticket("NONEXISTENT") is None
