"""
Tests for Virtual Sub-Account Manager
======================================
Validates:
  1. Account allocation and persistence
  2. Buy/sell bookkeeping with average cost calculation
  3. Affordability and position checks
  4. Multi-bot isolation (防踩踏)
  5. JSON save/load round-trip
  6. Virtual portfolio state construction
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from src.core.virtual_account import (
    VirtualAccountManager,
    VirtualPosition,
    VirtualSubAccount,
)


# ═══════════════════════════════════════════════════════════════════
# VirtualPosition
# ═══════════════════════════════════════════════════════════════════

class TestVirtualPosition:
    def test_market_value(self):
        pos = VirtualPosition(ticker="AAPL", quantity=10.0, average_price=150.0)
        assert pos.market_value == 1500.0

    def test_to_dict(self):
        pos = VirtualPosition(ticker="TSLA", quantity=5.0, average_price=200.0)
        d = pos.to_dict()
        assert d["ticker"] == "TSLA"
        assert d["quantity"] == 5.0
        assert d["average_price"] == 200.0


# ═══════════════════════════════════════════════════════════════════
# VirtualSubAccount
# ═══════════════════════════════════════════════════════════════════

class TestVirtualSubAccount:
    def _make(self, cash: float = 5000.0) -> VirtualSubAccount:
        return VirtualSubAccount(
            bot_id="TestBot",
            allocated_capital=cash,
            available_cash=cash,
        )

    # ── can_afford ────────────────────────────────────────────────

    def test_can_afford_yes(self):
        acc = self._make(5000)
        assert acc.can_afford(3000) is True

    def test_can_afford_no(self):
        acc = self._make(5000)
        assert acc.can_afford(6000) is False

    def test_can_afford_exact(self):
        acc = self._make(5000)
        assert acc.can_afford(5000) is True

    # ── get_position_qty ──────────────────────────────────────────

    def test_get_position_qty_empty(self):
        acc = self._make()
        assert acc.get_position_qty("AAPL") == 0.0

    def test_get_position_qty_after_buy(self):
        acc = self._make()
        acc.record_trade("AAPL", 10.0, 150.0)
        assert acc.get_position_qty("AAPL") == 10.0

    # ── record_trade: buy ─────────────────────────────────────────

    def test_buy_deducts_cash(self):
        acc = self._make(5000)
        acc.record_trade("AAPL", 10.0, 150.0)
        assert acc.available_cash == pytest.approx(3500.0)

    def test_buy_creates_position(self):
        acc = self._make(5000)
        acc.record_trade("AAPL", 10.0, 150.0)
        assert "AAPL" in acc.positions
        assert acc.positions["AAPL"].quantity == 10.0
        assert acc.positions["AAPL"].average_price == 150.0

    def test_buy_averaging_up(self):
        """加碼：買 10 股 @ 100，再買 10 股 @ 200 → 平均成本 150"""
        acc = self._make(10000)
        acc.record_trade("TSLA", 10.0, 100.0)
        acc.record_trade("TSLA", 10.0, 200.0)
        assert acc.positions["TSLA"].quantity == 20.0
        assert acc.positions["TSLA"].average_price == pytest.approx(150.0)
        assert acc.available_cash == pytest.approx(10000 - 1000 - 2000)

    # ── record_trade: sell ────────────────────────────────────────

    def test_sell_adds_cash(self):
        acc = self._make(5000)
        acc.record_trade("AAPL", 10.0, 150.0)   # buy
        acc.record_trade("AAPL", -5.0, 160.0)    # sell 5 @ 160
        assert acc.available_cash == pytest.approx(3500 + 800.0)

    def test_sell_reduces_position(self):
        acc = self._make(5000)
        acc.record_trade("AAPL", 10.0, 150.0)
        acc.record_trade("AAPL", -3.0, 160.0)
        assert acc.positions["AAPL"].quantity == pytest.approx(7.0)

    def test_sell_closes_position(self):
        acc = self._make(5000)
        acc.record_trade("AAPL", 10.0, 150.0)
        acc.record_trade("AAPL", -10.0, 160.0)
        assert "AAPL" not in acc.positions

    def test_sell_tracks_realised_pnl(self):
        """買 10 @ 100, 賣 10 @ 120 → PnL = 200"""
        acc = self._make(5000)
        acc.record_trade("AAPL", 10.0, 100.0)
        acc.record_trade("AAPL", -10.0, 120.0)
        assert acc.realised_pnl == pytest.approx(200.0)

    def test_sell_negative_pnl(self):
        """買 10 @ 100, 賣 10 @ 80 → PnL = -200"""
        acc = self._make(5000)
        acc.record_trade("AAPL", 10.0, 100.0)
        acc.record_trade("AAPL", -10.0, 80.0)
        assert acc.realised_pnl == pytest.approx(-200.0)

    # ── properties ────────────────────────────────────────────────

    def test_total_invested_value(self):
        acc = self._make(10000)
        acc.record_trade("AAPL", 10.0, 150.0)
        acc.record_trade("TSLA", 5.0, 200.0)
        assert acc.total_invested_value == pytest.approx(2500.0)

    def test_total_nav(self):
        acc = self._make(10000)
        acc.record_trade("AAPL", 10.0, 150.0)
        # cash = 10000 - 1500 = 8500, invested = 1500
        assert acc.total_nav == pytest.approx(10000.0)

    def test_exposure_pct(self):
        acc = self._make(10000)
        acc.record_trade("AAPL", 10.0, 100.0)
        # invested = 1000, nav = 10000
        assert acc.exposure_pct == pytest.approx(0.1)

    def test_trade_count_increments(self):
        acc = self._make(5000)
        acc.record_trade("AAPL", 10.0, 100.0)
        acc.record_trade("AAPL", -5.0, 110.0)
        assert acc.trade_count == 2

    def test_summary_string(self):
        acc = self._make(5000)
        acc.record_trade("AAPL", 10.0, 150.0)
        s = acc.summary()
        assert "TestBot" in s
        assert "AAPL" in s


# ═══════════════════════════════════════════════════════════════════
# VirtualAccountManager
# ═══════════════════════════════════════════════════════════════════

class TestVirtualAccountManager:
    def _make_manager(self, tmp_path: str) -> VirtualAccountManager:
        filepath = os.path.join(tmp_path, "test_accounts.json")
        return VirtualAccountManager(storage_file=filepath)

    def test_allocate_new_account(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        acc = mgr.allocate_account("BotA", 5000.0)
        assert acc.bot_id == "BotA"
        assert acc.allocated_capital == 5000.0
        assert acc.available_cash == 5000.0

    def test_allocate_existing_returns_same(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        acc1 = mgr.allocate_account("BotA", 5000.0)
        acc1.record_trade("AAPL", 10.0, 100.0)
        acc2 = mgr.allocate_account("BotA", 9999.0)  # different capital
        # Should return existing, not reset
        assert acc2.available_cash == pytest.approx(4000.0)
        assert acc2 is acc1

    def test_multi_bot_isolation(self, tmp_path):
        """防踩踏：兩個機械人的持倉互不可見"""
        mgr = self._make_manager(str(tmp_path))
        bot_a = mgr.allocate_account("GrowthBot", 5000.0)
        bot_b = mgr.allocate_account("MeanRevBot", 3000.0)

        bot_a.record_trade("TSLA", 10.0, 200.0)
        assert bot_a.get_position_qty("TSLA") == 10.0
        assert bot_b.get_position_qty("TSLA") == 0.0  # B 看不到 A 的股票

    def test_isolation_cash(self, tmp_path):
        """曝險隔離：A 虧光不影響 B"""
        mgr = self._make_manager(str(tmp_path))
        bot_a = mgr.allocate_account("BotA", 1000.0)
        bot_b = mgr.allocate_account("BotB", 3000.0)

        # A 花光所有錢
        bot_a.record_trade("AAPL", 10.0, 100.0)
        assert bot_a.can_afford(100.0) is False  # A 沒錢了
        assert bot_b.can_afford(100.0) is True   # B 不受影響

    def test_total_allocated(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        mgr.allocate_account("BotA", 5000.0)
        mgr.allocate_account("BotB", 3000.0)
        assert mgr.total_allocated == pytest.approx(8000.0)

    def test_validate_against_real_account(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        mgr.allocate_account("BotA", 5000.0)
        mgr.allocate_account("BotB", 3000.0)
        result = mgr.validate_against_real_account(real_nav=10000.0)
        assert result["over_allocated"] is False

        result2 = mgr.validate_against_real_account(real_nav=6000.0)
        assert result2["over_allocated"] is True

    def test_remove_account_empty(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        mgr.allocate_account("BotA", 5000.0)
        assert mgr.remove_account("BotA") is True
        assert mgr.get_account("BotA") is None

    def test_remove_account_with_positions_fails(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        acc = mgr.allocate_account("BotA", 5000.0)
        acc.record_trade("AAPL", 10.0, 100.0)
        assert mgr.remove_account("BotA") is False  # Still has positions


# ═══════════════════════════════════════════════════════════════════
# JSON Persistence Round-Trip
# ═══════════════════════════════════════════════════════════════════

class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        filepath = os.path.join(str(tmp_path), "test_accounts.json")

        # Create and populate
        mgr1 = VirtualAccountManager(storage_file=filepath)
        acc = mgr1.allocate_account("PersistBot", 5000.0)
        acc.record_trade("AAPL", 10.0, 150.0)
        acc.record_trade("TSLA", 5.0, 200.0)
        acc.record_trade("AAPL", -3.0, 160.0)
        mgr1.save_state()

        # Load into new manager
        mgr2 = VirtualAccountManager(storage_file=filepath)
        acc2 = mgr2.get_account("PersistBot")

        assert acc2 is not None
        assert acc2.bot_id == "PersistBot"
        assert acc2.allocated_capital == 5000.0
        assert acc2.available_cash == pytest.approx(acc.available_cash)
        assert acc2.trade_count == 3
        assert acc2.realised_pnl == pytest.approx(acc.realised_pnl)

        # Positions preserved
        assert "AAPL" in acc2.positions
        assert acc2.positions["AAPL"].quantity == pytest.approx(7.0)
        assert "TSLA" in acc2.positions
        assert acc2.positions["TSLA"].quantity == pytest.approx(5.0)

    def test_load_nonexistent_file(self, tmp_path):
        filepath = os.path.join(str(tmp_path), "nonexistent.json")
        mgr = VirtualAccountManager(storage_file=filepath)
        assert len(mgr.accounts) == 0


# ═══════════════════════════════════════════════════════════════════
# Integration: Virtual Portfolio State for RiskAgent
# ═══════════════════════════════════════════════════════════════════

class TestVirtualPortfolioState:
    def test_build_virtual_portfolio_state(self):
        from src.agents.decision.risk import RiskAgent

        acc = VirtualSubAccount(
            bot_id="TestBot",
            allocated_capital=10000.0,
            available_cash=8000.0,
        )
        acc.positions["AAPL"] = VirtualPosition("AAPL", 10.0, 150.0)
        acc.positions["TSLA"] = VirtualPosition("TSLA", 5.0, 200.0)

        state = RiskAgent.build_virtual_portfolio_state(acc)
        assert state.cash == pytest.approx(8000.0)
        assert state.invested_value == pytest.approx(2500.0)
        assert state.total_nav == pytest.approx(10500.0)
        assert len(state.positions) == 2

    def test_build_virtual_portfolio_with_price_map(self):
        from src.agents.decision.risk import RiskAgent

        acc = VirtualSubAccount(
            bot_id="TestBot",
            allocated_capital=10000.0,
            available_cash=8000.0,
        )
        acc.positions["AAPL"] = VirtualPosition("AAPL", 10.0, 150.0)

        # Market price different from average cost
        price_map = {"AAPL": 170.0}
        state = RiskAgent.build_virtual_portfolio_state(acc, price_map=price_map)
        assert state.invested_value == pytest.approx(1700.0)  # 10 * 170
        assert state.total_nav == pytest.approx(9700.0)       # 8000 + 1700
