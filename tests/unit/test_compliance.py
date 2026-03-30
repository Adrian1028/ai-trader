"""
Comprehensive unit tests for FCA-grade Compliance Guard.
Covers both the NEW per-bot interface (pre_trade_check + VirtualSubAccount)
and the BACKWARD-COMPATIBLE interface (validate_order).
"""
from __future__ import annotations

import time

import pytest

from src.compliance.guard import ComplianceGuard, VetoReason, VetoResult
from src.core.virtual_account import VirtualSubAccount


# ── Helper ────────────────────────────────────────────────────────

def _make_account(
    bot_id: str = "TestBot",
    capital: float = 100_000.0,
    cash: float | None = None,
) -> VirtualSubAccount:
    return VirtualSubAccount(
        bot_id=bot_id,
        allocated_capital=capital,
        available_cash=cash if cash is not None else capital,
    )


# ══════════════════════════════════════════════════════════════════════
# 1. FAT-FINGER PROTECTION (multi-tier)
# ══════════════════════════════════════════════════════════════════════

class TestFatFinger:
    def test_blocks_order_exceeding_hard_cap(self):
        guard = ComplianceGuard("TestBot", max_order_pct_of_nav=0.05, warn_order_pct_of_nav=0.03)
        acc = _make_account(capital=100_000)
        result = guard.pre_trade_check(
            ticker="AAPL", quantity=40, estimated_price=150.0,
            account=acc,   # order_value = 6000 = 6% of 100k
        )
        assert not result.is_approved
        assert result.reason is VetoReason.FAT_FINGER
        assert result.severity == "critical"

    def test_blocks_order_exceeding_warning_tier(self):
        guard = ComplianceGuard("TestBot", max_order_pct_of_nav=0.05, warn_order_pct_of_nav=0.03)
        acc = _make_account(capital=100_000)
        result = guard.pre_trade_check(
            ticker="AAPL", quantity=23.3, estimated_price=150.0,
            account=acc,   # order_value ≈ 3500 = 3.5% of 100k
        )
        assert not result.is_approved
        assert result.reason is VetoReason.FAT_FINGER
        assert result.severity == "warning"

    def test_allows_order_within_limits(self):
        guard = ComplianceGuard("TestBot", max_order_pct_of_nav=0.05, warn_order_pct_of_nav=0.03)
        acc = _make_account(capital=100_000)
        result = guard.pre_trade_check(
            ticker="AAPL", quantity=13, estimated_price=150.0,
            account=acc,   # order_value = 1950 < 3% of 100k
        )
        assert result.is_approved

    def test_negative_quantity_uses_absolute_value(self):
        guard = ComplianceGuard("TestBot", max_order_pct_of_nav=0.05)
        acc = _make_account(capital=100_000)
        acc.positions["AAPL"] = __import__(
            "src.core.virtual_account", fromlist=["VirtualPosition"]
        ).VirtualPosition("AAPL", 100, 150.0)
        result = guard.pre_trade_check(
            ticker="AAPL", quantity=-40, estimated_price=150.0,
            account=acc, side="sell",  # |order_value| = 6000 = 6%
        )
        assert not result.is_approved
        assert result.reason is VetoReason.FAT_FINGER

    # ── backward compat ──────────────────────────────────────────
    def test_backward_compat_validate_order(self):
        guard = ComplianceGuard("TestBot", max_order_pct_of_nav=0.05, warn_order_pct_of_nav=0.03)
        result = guard.validate_order(
            order_value=6_000, nav=100_000,
            pending_count_for_instrument=0, ticker="AAPL",
        )
        assert not result.allowed
        assert result.reason is VetoReason.FAT_FINGER


# ══════════════════════════════════════════════════════════════════════
# 2. VIRTUAL ACCOUNT INTEGRATION
# ══════════════════════════════════════════════════════════════════════

class TestVirtualAccountIntegration:
    def test_blocks_buy_when_insufficient_cash(self):
        guard = ComplianceGuard("TestBot", max_order_pct_of_nav=1.0, warn_order_pct_of_nav=0.9)
        acc = _make_account(capital=10_000, cash=500)
        result = guard.pre_trade_check(
            ticker="AAPL", quantity=10, estimated_price=150.0,
            account=acc, side="buy",  # needs 1500, has 500
        )
        assert not result.is_approved
        assert result.reason is VetoReason.INSUFFICIENT_FUNDS

    def test_blocks_sell_when_insufficient_position(self):
        guard = ComplianceGuard("TestBot", max_order_pct_of_nav=1.0, warn_order_pct_of_nav=0.9)
        acc = _make_account(capital=10_000)
        # Has no AAPL position
        result = guard.pre_trade_check(
            ticker="AAPL", quantity=-5, estimated_price=150.0,
            account=acc, side="sell",
        )
        assert not result.is_approved
        assert result.reason is VetoReason.INSUFFICIENT_POSITION

    def test_allows_sell_with_sufficient_position(self):
        from src.core.virtual_account import VirtualPosition
        guard = ComplianceGuard("TestBot", max_order_pct_of_nav=1.0, warn_order_pct_of_nav=0.9)
        acc = _make_account(capital=10_000)
        acc.positions["AAPL"] = VirtualPosition("AAPL", 20.0, 150.0)
        result = guard.pre_trade_check(
            ticker="AAPL", quantity=-5, estimated_price=150.0,
            account=acc, side="sell",
        )
        assert result.is_approved


# ══════════════════════════════════════════════════════════════════════
# 3. CUMULATIVE DAILY TURNOVER
# ══════════════════════════════════════════════════════════════════════

class TestDailyTurnover:
    def test_blocks_when_daily_turnover_exceeded(self):
        guard = ComplianceGuard(
            "TestBot",
            max_order_pct_of_nav=0.50,
            warn_order_pct_of_nav=0.40,
            max_daily_turnover_pct=0.30,
            kill_switch_max_ops=100,
            max_orders_per_minute=100,
            loop_detection_repeat_threshold=100,
        )
        acc = _make_account(capital=100_000)
        # Place several orders that sum above 30% of NAV
        for _ in range(3):
            guard.pre_trade_check(
                ticker="AAPL", quantity=60, estimated_price=150.0,
                account=acc,  # each = 9000
            )
        # Fourth should trigger daily cap (27k + 9k = 36k > 30k)
        result = guard.pre_trade_check(
            ticker="MSFT", quantity=60, estimated_price=150.0,
            account=acc,
        )
        assert not result.is_approved
        assert result.reason is VetoReason.CUMULATIVE_EXPOSURE


# ══════════════════════════════════════════════════════════════════════
# 4. KILL SWITCH (rate-based + loop detection)
# ══════════════════════════════════════════════════════════════════════

class TestKillSwitch:
    def test_triggers_on_rapid_fire(self):
        guard = ComplianceGuard(
            "TestBot",
            kill_switch_max_ops=3,
            max_order_pct_of_nav=1.0,
            warn_order_pct_of_nav=0.9,
            max_orders_per_minute=100,
            loop_detection_repeat_threshold=100,
        )
        acc = _make_account(capital=100_000)
        for _ in range(3):
            guard.pre_trade_check(
                ticker="AAPL", quantity=1, estimated_price=100.0, account=acc,
            )
        result = guard.pre_trade_check(
            ticker="MSFT", quantity=1, estimated_price=100.0, account=acc,
        )
        assert not result.is_approved
        assert result.reason is VetoReason.KILL_SWITCH_RATE
        assert guard.is_killed
        assert guard.is_api_key_revoked

    def test_loop_detection_same_ticker(self):
        guard = ComplianceGuard(
            "TestBot",
            kill_switch_max_ops=100,
            loop_detection_window=30,
            loop_detection_repeat_threshold=3,
            max_order_pct_of_nav=1.0,
            warn_order_pct_of_nav=0.9,
            max_orders_per_minute=100,
        )
        acc = _make_account(capital=100_000)
        for _ in range(2):
            guard.pre_trade_check(
                ticker="TSLA", quantity=1, estimated_price=100.0, account=acc,
            )
        result = guard.pre_trade_check(
            ticker="TSLA", quantity=1, estimated_price=100.0, account=acc,
        )
        assert not result.is_approved
        assert result.reason is VetoReason.KILL_SWITCH_LOOP
        assert guard.is_killed

    def test_reset_kill_switch(self):
        guard = ComplianceGuard(
            "TestBot",
            kill_switch_max_ops=1,
            max_order_pct_of_nav=1.0,
            warn_order_pct_of_nav=0.9,
            max_orders_per_minute=100,
            loop_detection_repeat_threshold=100,
        )
        acc = _make_account(capital=100_000)
        guard.pre_trade_check(ticker="X", quantity=1, estimated_price=100, account=acc)
        guard.pre_trade_check(ticker="Y", quantity=1, estimated_price=100, account=acc)
        assert guard.is_killed
        assert guard.is_api_key_revoked

        guard.reset_kill_switch()
        assert not guard.is_killed
        assert not guard.is_api_key_revoked


# ══════════════════════════════════════════════════════════════════════
# 5. MAR ANTI-SPOOFING & ANTI-LAYERING
# ══════════════════════════════════════════════════════════════════════

class TestMAR:
    def test_blocks_high_cancel_rate(self):
        guard = ComplianceGuard(
            "TestBot",
            cancel_rate_threshold=0.80,
            kill_switch_max_ops=1000,
            max_order_pct_of_nav=1.0,
            warn_order_pct_of_nav=0.9,
            loop_detection_repeat_threshold=1000,
            max_orders_per_minute=1000,
        )
        acc = _make_account(capital=100_000)
        for i in range(25):
            guard.pre_trade_check(
                ticker=f"T{i}", quantity=1, estimated_price=100.0, account=acc,
            )
        # Directly set cancel count to avoid check_spoofing triggering kill switch
        # during record_cancellation() calls
        guard.total_orders_cancelled = 22  # 22/25 = 88% cancel rate

        result = guard.pre_trade_check(
            ticker="NEW", quantity=1, estimated_price=100.0, account=acc,
        )
        assert not result.is_approved
        assert result.reason is VetoReason.SPOOFING_SUSPICION

    def test_blocks_high_amendment_rate(self):
        guard = ComplianceGuard(
            "TestBot",
            amendment_rate_threshold=0.70,
            cancel_rate_threshold=0.99,
            kill_switch_max_ops=1000,
            max_order_pct_of_nav=1.0,
            warn_order_pct_of_nav=0.9,
            loop_detection_repeat_threshold=1000,
            max_orders_per_minute=1000,
        )
        acc = _make_account(capital=100_000)
        for i in range(25):
            guard.pre_trade_check(
                ticker=f"T{i}", quantity=1, estimated_price=100.0, account=acc,
            )
        for _ in range(20):
            guard.record_amendment()

        result = guard.pre_trade_check(
            ticker="NEW", quantity=1, estimated_price=100.0, account=acc,
        )
        assert not result.is_approved
        assert result.reason is VetoReason.EXCESSIVE_AMENDMENTS

    def test_layering_detection(self):
        guard = ComplianceGuard(
            "TestBot",
            layering_max_one_side=3,
            kill_switch_max_ops=100,
            max_order_pct_of_nav=1.0,
            warn_order_pct_of_nav=0.9,
            loop_detection_repeat_threshold=100,
            max_orders_per_minute=100,
        )
        acc = _make_account(capital=100_000)
        for _ in range(3):
            guard.pre_trade_check(
                ticker="LAYER", quantity=1, estimated_price=100.0,
                account=acc, side="buy",
            )
        result = guard.pre_trade_check(
            ticker="LAYER", quantity=1, estimated_price=100.0,
            account=acc, side="buy",
        )
        assert not result.is_approved
        assert result.reason is VetoReason.LAYERING_SUSPICION

    def test_pending_limit(self):
        guard = ComplianceGuard("TestBot", max_pending_per_instrument=50)
        acc = _make_account(capital=100_000)
        result = guard.pre_trade_check(
            ticker="FULL", quantity=1, estimated_price=100.0,
            account=acc, pending_count_for_instrument=50,
        )
        assert not result.is_approved
        assert result.reason is VetoReason.PENDING_LIMIT_EXCEEDED


# ══════════════════════════════════════════════════════════════════════
# 6. PRICE SANITY
# ══════════════════════════════════════════════════════════════════════

class TestPriceSanity:
    def test_blocks_anomalous_price(self):
        guard = ComplianceGuard(
            "TestBot",
            max_price_deviation_pct=0.10,
            max_order_pct_of_nav=1.0,
            warn_order_pct_of_nav=0.9,
        )
        acc = _make_account(capital=100_000)
        result = guard.pre_trade_check(
            ticker="AAPL", quantity=10, estimated_price=170.0,
            account=acc, reference_price=150.0,  # 13.3% deviation
        )
        assert not result.is_approved
        assert result.reason is VetoReason.PRICE_ANOMALY

    def test_allows_normal_price(self):
        guard = ComplianceGuard(
            "TestBot",
            max_price_deviation_pct=0.10,
            max_order_pct_of_nav=1.0,
            warn_order_pct_of_nav=0.9,
        )
        acc = _make_account(capital=100_000)
        result = guard.pre_trade_check(
            ticker="AAPL", quantity=10, estimated_price=152.0,
            account=acc, reference_price=150.0,  # 1.3% deviation
        )
        assert result.is_approved


# ══════════════════════════════════════════════════════════════════════
# 7. TRANSACTION COST GATE
# ══════════════════════════════════════════════════════════════════════

class TestTransactionCostGate:
    def test_blocks_when_friction_exceeds_edge(self):
        guard = ComplianceGuard(
            "TestBot",
            min_expected_roi_after_costs=0.005,
            stamp_duty_rate=0.005,
            max_order_pct_of_nav=1.0,
            warn_order_pct_of_nav=0.9,
        )
        acc = _make_account(capital=100_000)
        result = guard.pre_trade_check(
            ticker="LSE_STOCK", quantity=10, estimated_price=100.0,
            account=acc, expected_roi=0.004, is_uk_equity=True,
        )
        assert not result.is_approved
        assert result.reason is VetoReason.COST_EXCEEDS_EDGE

    def test_allows_when_edge_covers_costs(self):
        guard = ComplianceGuard(
            "TestBot",
            min_expected_roi_after_costs=0.005,
            max_order_pct_of_nav=1.0,
            warn_order_pct_of_nav=0.9,
        )
        acc = _make_account(capital=100_000)
        result = guard.pre_trade_check(
            ticker="LSE_STOCK", quantity=10, estimated_price=100.0,
            account=acc, expected_roi=0.02, is_uk_equity=True,
        )
        assert result.is_approved

    def test_round_trip_cost_estimation(self):
        guard = ComplianceGuard("TestBot", stamp_duty_rate=0.005)
        cost = guard.estimate_round_trip_cost(10_000, is_uk=True)
        assert cost > 50   # at minimum stamp duty
        assert cost < 200  # sanity upper bound


# ══════════════════════════════════════════════════════════════════════
# 8. OPRO REWARD PENALTY INTEGRATION
# ══════════════════════════════════════════════════════════════════════

class TestMARPenalty:
    def test_zero_penalty_when_clean(self):
        guard = ComplianceGuard("TestBot")
        assert guard.compute_mar_penalty() == 0.0

    def test_penalty_for_high_cancel_rate(self):
        guard = ComplianceGuard("TestBot")
        guard.total_orders_placed = 100
        guard.total_orders_cancelled = 80
        penalty = guard.compute_mar_penalty()
        assert penalty < -20

    def test_extreme_penalty_for_kill_switch(self):
        guard = ComplianceGuard("TestBot")
        guard.kill_switch_triggered = True
        penalty = guard.compute_mar_penalty()
        assert penalty == -100.0


# ══════════════════════════════════════════════════════════════════════
# 9. COMPLIANCE REPORT
# ══════════════════════════════════════════════════════════════════════

class TestComplianceReport:
    def test_report_structure(self):
        guard = ComplianceGuard("TestBot")
        report = guard.compliance_report()
        assert "bot_id" in report
        assert "killed" in report
        assert "api_key_revoked" in report
        assert "cancel_rate" in report
        assert "amendment_rate" in report
        assert "daily_turnover" in report
        assert "mar_penalty" in report
        assert "pending_buys" in report
        assert "pending_sells" in report


# ══════════════════════════════════════════════════════════════════════
# 10. EVENT LOGGING
# ══════════════════════════════════════════════════════════════════════

class TestEventLogging:
    def test_veto_events_logged(self):
        guard = ComplianceGuard(
            "TestBot",
            max_order_pct_of_nav=0.05,
            warn_order_pct_of_nav=0.03,
            log_dir="logs/test_compliance_events",
        )
        acc = _make_account(capital=100_000)
        # Use warning-tier veto (not critical) so events aren't persisted & cleared
        guard.pre_trade_check(
            ticker="AAPL", quantity=23, estimated_price=150.0, account=acc,
            # order_value ≈ 3450 = 3.45% → warning tier (between 3% and 5%)
        )
        assert len(guard._events) >= 1
        assert guard._events[0].event_type == "veto"

    def test_allow_events_logged(self):
        guard = ComplianceGuard("TestBot", log_dir="logs/test_compliance_events2")
        acc = _make_account(capital=100_000)
        guard.pre_trade_check(
            ticker="AAPL", quantity=1, estimated_price=100.0, account=acc,
        )
        assert any(e.event_type == "allow" for e in guard._events)


# ══════════════════════════════════════════════════════════════════════
# 11. SPOOFING DETECTION (check_spoofing method)
# ══════════════════════════════════════════════════════════════════════

class TestSpoofingDetection:
    def test_check_spoofing_triggers_kill_switch(self):
        guard = ComplianceGuard("TestBot", cancel_rate_threshold=0.85)
        guard.total_orders_placed = 20
        guard.total_orders_cancelled = 18  # 90% cancel rate
        guard.check_spoofing()
        assert guard.kill_switch_triggered

    def test_check_spoofing_safe_with_low_samples(self):
        guard = ComplianceGuard("TestBot", cancel_rate_threshold=0.85)
        guard.total_orders_placed = 5
        guard.total_orders_cancelled = 5  # 100% but too few samples
        guard.check_spoofing()
        assert not guard.kill_switch_triggered

    def test_daily_counter_reset(self):
        guard = ComplianceGuard("TestBot")
        guard.daily_traded_volume = 50_000
        guard.total_orders_placed = 100
        guard.total_orders_cancelled = 30
        guard.reset_daily_counters()
        assert guard.daily_traded_volume == 0.0
        assert guard.total_orders_placed == 0
        assert guard.total_orders_cancelled == 0


# ══════════════════════════════════════════════════════════════════════
# 12. VetoResult BACKWARD COMPAT
# ══════════════════════════════════════════════════════════════════════

class TestVetoResultCompat:
    def test_allowed_alias(self):
        v = VetoResult(is_approved=True)
        assert v.allowed is True
        assert v.is_approved is True

    def test_not_allowed_alias(self):
        v = VetoResult(is_approved=False, reason=VetoReason.FAT_FINGER, detail="test")
        assert v.allowed is False
        assert v.is_approved is False
        assert v.reason is VetoReason.FAT_FINGER
