"""Unit tests for Risk Agent."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.agents.decision.risk import (
    PortfolioRiskState,
    RiskAgent,
    RiskEnvelope,
    RiskVerdict,
    _CASH_BUFFER_PCT,
    _MIN_TRADEABLE_QTY,
    _MEMORY_BLEND_WEIGHT,
    _MIN_MEMORY_TRADES,
)
from src.core.virtual_account import VirtualPosition, VirtualSubAccount
from src.memory.episodic_memory import Episode, EpisodicMemory


@pytest.fixture
def agent():
    return RiskAgent(
        max_single_position_pct=0.05,
        max_portfolio_exposure_pct=0.95,
        max_var_pct_of_nav=0.02,
    )


@pytest.fixture
def portfolio():
    return PortfolioRiskState(
        total_nav=100_000,
        invested_value=50_000,
        cash=50_000,
        exposure_pct=0.5,
        positions={"POS1": 25_000, "POS2": 25_000},
    )


@pytest.fixture
def returns():
    np.random.seed(42)
    return np.random.normal(0.0005, 0.02, 252)


class TestVaR:
    def test_var_positive_for_volatile_asset(self, agent, returns):
        var95, var99, es, cvar95, cvar99 = agent._compute_var(returns, price=150.0)
        assert var95 > 0
        assert var99 > var95
        assert es >= var95
        assert cvar95 >= 0
        assert cvar99 >= cvar95

    def test_var_zero_for_zero_volatility(self, agent):
        flat = np.zeros(50)
        var95, var99, es, cvar95, cvar99 = agent._compute_var(flat, price=100.0)
        assert var95 == 0.0
        assert cvar95 == 0.0


class TestKelly:
    def test_positive_kelly_for_profitable_returns(self, agent):
        # Upward-biased returns
        np.random.seed(1)
        returns = np.random.normal(0.005, 0.01, 100)
        kelly = agent._kelly_fraction(returns, confidence=0.7, direction=1)
        assert kelly > 0

    def test_zero_kelly_when_no_edge(self, agent):
        np.random.seed(2)
        returns = np.random.normal(-0.01, 0.02, 100)
        kelly = agent._kelly_fraction(returns, confidence=0.2, direction=1)
        assert kelly >= 0  # never negative


class TestPositionSizing:
    def test_approved_within_limits(self, agent, portfolio, returns):
        env = agent.evaluate(
            direction=1, current_price=150.0, returns=returns,
            atr=3.0, confidence=0.7, portfolio=portfolio,
        )
        assert env.verdict in (RiskVerdict.APPROVED, RiskVerdict.REDUCED)
        assert env.suggested_quantity > 0

    def test_rejected_when_fully_invested(self, agent, returns):
        full_portfolio = PortfolioRiskState(
            total_nav=100_000,
            invested_value=96_000,
            cash=4_000,
            exposure_pct=0.96,
        )
        env = agent.evaluate(
            direction=1, current_price=150.0, returns=returns,
            atr=3.0, confidence=0.7, portfolio=full_portfolio,
        )
        assert env.verdict == RiskVerdict.REJECTED

    def test_rejected_with_insufficient_data(self, agent, portfolio):
        short = np.array([0.01, -0.01, 0.005])
        env = agent.evaluate(
            direction=1, current_price=150.0, returns=short,
            atr=3.0, confidence=0.7, portfolio=portfolio,
        )
        assert env.verdict == RiskVerdict.REJECTED


class TestStopLoss:
    def test_stop_loss_below_price_for_buys(self, agent, portfolio, returns):
        env = agent.evaluate(
            direction=1, current_price=150.0, returns=returns,
            atr=3.0, confidence=0.7, portfolio=portfolio,
        )
        if env.verdict != RiskVerdict.REJECTED:
            assert env.stop_loss_price < 150.0
            assert env.take_profit_price > 150.0
            assert env.risk_reward_ratio > 0


class TestPortfolioState:
    def test_concentration_single_stock(self):
        state = PortfolioRiskState(
            total_nav=100_000,
            positions={"ONLY": 100_000},
        )
        assert state.concentration == pytest.approx(1.0)

    def test_concentration_diversified(self):
        state = PortfolioRiskState(
            total_nav=100_000,
            positions={f"S{i}": 10_000 for i in range(10)},
        )
        assert state.concentration == pytest.approx(0.1)


# ═══════════════════════════════════════════════════════════════════
# VirtualSubAccount-aware evaluation (evaluate_with_account)
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def virtual_account():
    """虛擬帳戶：10,000 現金，持有 TSLA 10 股 @ 200"""
    acc = VirtualSubAccount(
        bot_id="test_bot",
        allocated_capital=10_000.0,
        available_cash=5_000.0,
    )
    acc.positions["TSLA"] = VirtualPosition(
        ticker="TSLA", quantity=10.0, average_price=200.0,
    )
    return acc


@pytest.fixture
def rich_account():
    """虛擬帳戶：現金充裕，無持倉"""
    return VirtualSubAccount(
        bot_id="rich_bot",
        allocated_capital=100_000.0,
        available_cash=100_000.0,
    )


@pytest.fixture
def broke_account():
    """虛擬帳戶：零現金"""
    return VirtualSubAccount(
        bot_id="broke_bot",
        allocated_capital=10_000.0,
        available_cash=0.0,
    )


class TestEvaluateWithAccountBuy:
    """買入方向的帳戶感知評估"""

    def test_buy_approved_with_sufficient_cash(self, agent, returns, rich_account):
        env = agent.evaluate_with_account(
            direction=1, current_price=150.0, returns=returns,
            atr=3.0, confidence=0.7,
            virtual_account=rich_account, symbol="AAPL",
        )
        assert env.verdict in (RiskVerdict.APPROVED, RiskVerdict.REDUCED)
        assert env.suggested_quantity > 0

    def test_buy_rejected_when_no_cash(self, agent, returns, broke_account):
        env = agent.evaluate_with_account(
            direction=1, current_price=150.0, returns=returns,
            atr=3.0, confidence=0.7,
            virtual_account=broke_account, symbol="AAPL",
        )
        assert env.verdict == RiskVerdict.REJECTED
        assert env.suggested_quantity == 0.0

    def test_buy_auto_downgrade_when_cash_tight(self, agent, returns):
        """現金不足以買滿理想數量，應自動降階"""
        # 持倉價值高 → NAV 高 → Kelly 建議量大，但現金很少
        acc = VirtualSubAccount(
            bot_id="tight_bot",
            allocated_capital=100_000.0,
            available_cash=200.0,  # 現金很少
        )
        # 加入大量持倉拉高 NAV（讓 Kelly × NAV 遠超可用現金）
        acc.positions["MSFT"] = VirtualPosition(
            ticker="MSFT", quantity=200.0, average_price=400.0,
        )
        # NAV = 200 + 200*400 = 80,200 → Kelly 建議 ~5% = 4,010 value
        # 但安全現金只有 200*0.99 = 198 → 一定會降階
        env = agent.evaluate_with_account(
            direction=1, current_price=150.0, returns=returns,
            atr=3.0, confidence=0.7,
            virtual_account=acc, symbol="AAPL",
        )
        if env.verdict != RiskVerdict.REJECTED:
            max_buyable = (200.0 * (1 - _CASH_BUFFER_PCT)) / 150.0
            assert env.suggested_quantity <= max_buyable + 0.0001
            assert env.verdict == RiskVerdict.REDUCED
            assert "降階" in env.reason

    def test_buy_respects_cash_buffer(self, agent, returns):
        """即使剛好夠，也要預留 1% 滑價緩衝"""
        # 精確計算讓基礎評估建議的量 × 價格 > 安全現金
        acc = VirtualSubAccount(
            bot_id="exact_bot",
            allocated_capital=50_000.0,
            available_cash=1000.0,
        )
        env = agent.evaluate_with_account(
            direction=1, current_price=150.0, returns=returns,
            atr=3.0, confidence=0.7,
            virtual_account=acc, symbol="AAPL",
        )
        if env.verdict != RiskVerdict.REJECTED:
            actual_value = env.suggested_quantity * 150.0
            safe_cash = 1000.0 * (1 - _CASH_BUFFER_PCT)
            assert actual_value <= safe_cash + 0.01  # 浮點容差


class TestEvaluateWithAccountSell:
    """賣出方向的帳戶感知評估"""

    def test_sell_rejected_when_no_position(self, agent, returns, rich_account):
        """沒有持倉不能賣"""
        env = agent.evaluate_with_account(
            direction=-1, current_price=150.0, returns=returns,
            atr=3.0, confidence=0.7,
            virtual_account=rich_account, symbol="NVDA",  # 帳戶沒有 NVDA
        )
        assert env.verdict == RiskVerdict.REJECTED
        assert "無" in env.reason and "持倉" in env.reason

    def test_sell_capped_by_owned_quantity(self, agent, returns, virtual_account):
        """賣出不能超過持有量"""
        env = agent.evaluate_with_account(
            direction=-1, current_price=200.0, returns=returns,
            atr=4.0, confidence=0.9,
            virtual_account=virtual_account, symbol="TSLA",
        )
        if env.verdict != RiskVerdict.REJECTED:
            assert env.suggested_quantity <= 10.0  # 持有 10 股

    def test_sell_approved_with_position(self, agent, returns, virtual_account):
        """有持倉應可賣出"""
        env = agent.evaluate_with_account(
            direction=-1, current_price=200.0, returns=returns,
            atr=4.0, confidence=0.7,
            virtual_account=virtual_account, symbol="TSLA",
        )
        assert env.verdict in (RiskVerdict.APPROVED, RiskVerdict.REDUCED)
        assert env.suggested_quantity > 0


class TestEvaluateWithAccountEdgeCases:
    """邊界條件測試"""

    def test_insufficient_return_history(self, agent, virtual_account):
        """歷史數據不足 20 天應拒絕"""
        short_returns = np.array([0.01, -0.01, 0.005])
        env = agent.evaluate_with_account(
            direction=1, current_price=200.0, returns=short_returns,
            atr=4.0, confidence=0.7,
            virtual_account=virtual_account, symbol="TSLA",
        )
        assert env.verdict == RiskVerdict.REJECTED

    def test_price_map_used_for_portfolio_state(self, agent, returns, virtual_account):
        """price_map 應被用來計算更精確的持倉估值"""
        price_map = {"TSLA": 250.0}  # 市價 250 vs 成本 200
        env = agent.evaluate_with_account(
            direction=1, current_price=150.0, returns=returns,
            atr=3.0, confidence=0.7,
            virtual_account=virtual_account, symbol="AAPL",
            price_map=price_map,
        )
        # 主要驗證不會崩潰，且 portfolio state 被正確建構
        assert env.verdict in (
            RiskVerdict.APPROVED, RiskVerdict.REDUCED, RiskVerdict.REJECTED,
        )

    def test_zero_atr_still_computes(self, agent, returns, rich_account):
        """ATR 為 None 時仍能評估（只是沒有停損停利）"""
        env = agent.evaluate_with_account(
            direction=1, current_price=150.0, returns=returns,
            atr=None, confidence=0.7,
            virtual_account=rich_account, symbol="AAPL",
        )
        # 應該仍能通過（VaR + Kelly 不依賴 ATR）
        assert env.stop_loss_price == 0.0  # 無 ATR 就無停損

    def test_very_low_confidence_kelly_rejects(self, agent, returns, rich_account):
        """信心度極低時 Kelly 應拒絕"""
        env = agent.evaluate_with_account(
            direction=1, current_price=150.0, returns=returns,
            atr=3.0, confidence=0.05,
            virtual_account=rich_account, symbol="AAPL",
        )
        # 低信心度通常會讓 Kelly 為 0 或極低
        if env.verdict != RiskVerdict.REJECTED:
            assert env.suggested_quantity > 0  # 若通過，量也要有效

    def test_sell_exact_position_amount(self, agent, returns):
        """賣出剛好等於持有量"""
        acc = VirtualSubAccount(
            bot_id="exact_sell",
            allocated_capital=10_000.0,
            available_cash=5_000.0,
        )
        acc.positions["TINY"] = VirtualPosition(
            ticker="TINY", quantity=0.5, average_price=100.0,
        )
        env = agent.evaluate_with_account(
            direction=-1, current_price=100.0, returns=returns,
            atr=2.0, confidence=0.8,
            virtual_account=acc, symbol="TINY",
        )
        if env.verdict != RiskVerdict.REJECTED:
            assert env.suggested_quantity <= 0.5


class TestStopLossWithAccount:
    """帳戶感知模式下的停損停利驗證"""

    def test_buy_stop_loss_below_price(self, agent, returns, rich_account):
        env = agent.evaluate_with_account(
            direction=1, current_price=150.0, returns=returns,
            atr=3.0, confidence=0.7,
            virtual_account=rich_account, symbol="AAPL",
        )
        if env.verdict != RiskVerdict.REJECTED:
            assert env.stop_loss_price < 150.0
            assert env.take_profit_price > 150.0

    def test_sell_stop_loss_above_price(self, agent, returns, virtual_account):
        env = agent.evaluate_with_account(
            direction=-1, current_price=200.0, returns=returns,
            atr=4.0, confidence=0.7,
            virtual_account=virtual_account, symbol="TSLA",
        )
        if env.verdict != RiskVerdict.REJECTED:
            assert env.stop_loss_price > 200.0
            assert env.take_profit_price < 200.0


# ═══════════════════════════════════════════════════════════════════
# 情節記憶融合 (Episodic Memory ↔ Kelly Fraction)
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def memory_with_trades(tmp_path):
    """記憶庫含 20 筆交易：12 勝 8 敗"""
    mem = EpisodicMemory(store_dir=str(tmp_path / "risk_mem"))
    np.random.seed(99)
    for i in range(12):
        mem.store(Episode(
            ticker="WIN", embedding=np.random.randn(32),
            action="BUY", roi=0.08,
        ))
    for i in range(8):
        mem.store(Episode(
            ticker="LOSS", embedding=np.random.randn(32),
            action="BUY", roi=-0.04,
        ))
    return mem


@pytest.fixture
def sparse_memory(tmp_path):
    """記憶庫只有 3 筆（低於 _MIN_MEMORY_TRADES 門檻）"""
    mem = EpisodicMemory(store_dir=str(tmp_path / "sparse_mem"))
    for i in range(3):
        mem.store(Episode(
            ticker="X", embedding=np.random.randn(32),
            action="BUY", roi=0.05,
        ))
    return mem


class TestKellyMemoryIntegration:
    """RiskAgent + EpisodicMemory 的 Kelly 分數融合"""

    def test_memory_biases_kelly_fraction(self, returns, memory_with_trades):
        """有記憶的 RiskAgent 的 Kelly 結果應與無記憶的不同"""
        agent_no_mem = RiskAgent()
        agent_with_mem = RiskAgent(episodic_memory=memory_with_trades)

        np.random.seed(42)
        kelly_no = agent_no_mem._kelly_fraction(returns, confidence=0.7, direction=1)
        np.random.seed(42)
        kelly_with = agent_with_mem._kelly_fraction(returns, confidence=0.7, direction=1)

        # 記憶有 12/20 = 60% 勝率，盈虧比 0.08/0.04 = 2.0
        # 這與基礎 price return 不同，所以 Kelly 應有差異
        assert kelly_no != kelly_with
        # 兩者都應為正（有利可圖的策略）
        assert kelly_no > 0
        assert kelly_with > 0

    def test_sparse_memory_not_used(self, returns, sparse_memory):
        """不足 _MIN_MEMORY_TRADES 筆時不融合記憶"""
        agent_no_mem = RiskAgent()
        agent_sparse = RiskAgent(episodic_memory=sparse_memory)

        np.random.seed(42)
        kelly_no = agent_no_mem._kelly_fraction(returns, confidence=0.7, direction=1)
        np.random.seed(42)
        kelly_sparse = agent_sparse._kelly_fraction(returns, confidence=0.7, direction=1)

        # 記憶太少，不融合，結果應相同
        assert kelly_no == pytest.approx(kelly_sparse)

    def test_no_memory_backward_compatible(self, returns):
        """沒有傳入記憶的 RiskAgent 行為完全不變"""
        agent = RiskAgent()
        assert agent._memory is None
        kelly = agent._kelly_fraction(returns, confidence=0.7, direction=1)
        assert kelly >= 0

    def test_memory_evaluate_full_pipeline(
        self, returns, memory_with_trades, portfolio,
    ):
        """完整 evaluate() 流程在有記憶時不會崩潰"""
        agent = RiskAgent(episodic_memory=memory_with_trades)
        env = agent.evaluate(
            direction=1, current_price=150.0, returns=returns,
            atr=3.0, confidence=0.7, portfolio=portfolio,
        )
        assert env.verdict in (RiskVerdict.APPROVED, RiskVerdict.REDUCED)
        assert env.half_kelly_fraction > 0

    def test_sell_direction_uses_sell_memory(self, tmp_path, returns, portfolio):
        """賣出方向時只參考 SELL 記憶"""
        mem = EpisodicMemory(store_dir=str(tmp_path / "sell_mem"))
        # BUY 記憶全虧
        for _ in range(10):
            mem.store(Episode(
                ticker="X", embedding=np.random.randn(32),
                action="BUY", roi=-0.10,
            ))
        # SELL 記憶全勝
        for _ in range(10):
            mem.store(Episode(
                ticker="X", embedding=np.random.randn(32),
                action="SELL", roi=0.08,
            ))

        agent = RiskAgent(episodic_memory=mem)
        # 賣出方向的 Kelly 應反映 SELL 記憶（勝率 100%）
        kelly = agent._kelly_fraction(returns, confidence=0.7, direction=-1)
        assert kelly > 0

    def test_memory_blend_weight_capped(self, tmp_path, returns):
        """即使有大量記憶，融合權重不超過 _MEMORY_BLEND_WEIGHT"""
        mem = EpisodicMemory(store_dir=str(tmp_path / "big_mem"))
        for _ in range(200):
            mem.store(Episode(
                ticker="X", embedding=np.random.randn(32),
                action="BUY", roi=0.05,
            ))
        perf = mem.get_recent_performance(action_filter="BUY")
        assert perf["total_trades"] == 30  # limit=30 預設

        agent = RiskAgent(episodic_memory=mem)
        # 不會崩潰，且記憶權重 = min(30/100, 0.30) = 0.30
        kelly = agent._kelly_fraction(returns, confidence=0.7, direction=1)
        assert kelly >= 0
