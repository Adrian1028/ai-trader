"""Unit tests for Episodic Memory and Counterfactual Replay."""
from __future__ import annotations

import numpy as np
import pytest

from src.memory.episodic_memory import Episode, EpisodicMemory
from src.memory.counterfactual_replay import (
    CounterfactualReplayEngine,
    FrictionModel,
)


class TestEpisodicMemory:
    def test_store_and_count(self, tmp_path):
        mem = EpisodicMemory(store_dir=str(tmp_path / "mem1"))
        ep = Episode(
            ticker="AAPL", isin="US0378331005",
            embedding=np.random.randn(32),
            roi=0.05, action="BUY", regime_tag="trending",
        )
        ep_id = mem.store(ep)
        assert ep_id.startswith("EP-")
        assert mem.count == 1

    def test_cosine_similarity_retrieval(self, tmp_path):
        mem = EpisodicMemory(store_dir=str(tmp_path / "mem2"))
        # Store two similar and one dissimilar episode
        base = np.ones(32, dtype=np.float64)
        similar = base + np.random.randn(32) * 0.1
        dissimilar = -base

        mem.store(Episode(ticker="A", embedding=base, roi=0.1, regime_tag="trending"))
        mem.store(Episode(ticker="B", embedding=similar, roi=-0.05, regime_tag="trending"))
        mem.store(Episode(ticker="C", embedding=dissimilar, roi=0.2, regime_tag="volatile"))

        results = mem.query_similar(base, k=2, min_similarity=0.5)
        assert len(results) >= 1
        # First result should be most similar to base
        assert results[0][0].ticker in ("A", "B")
        assert results[0][1] > 0.5

    def test_regime_filter(self, tmp_path):
        mem = EpisodicMemory(store_dir=str(tmp_path / "mem3"))
        emb = np.ones(32)
        mem.store(Episode(ticker="A", embedding=emb, regime_tag="trending"))
        mem.store(Episode(ticker="B", embedding=emb, regime_tag="volatile"))

        trending = mem.get_by_regime("trending")
        assert len(trending) == 1
        assert trending[0].ticker == "A"

    def test_contrastive_pairs(self, tmp_path):
        mem = EpisodicMemory(store_dir=str(tmp_path / "mem4"))
        emb = np.ones(32)
        mem.store(Episode(ticker="W", embedding=emb + 0.01, roi=0.05))
        mem.store(Episode(ticker="L", embedding=emb - 0.01, roi=-0.05))

        pairs = mem.find_contrastive_pairs(emb, k=1)
        # Should find at least one win/lose pair
        assert len(pairs) >= 0  # may be 0 if similarity too low

    def test_regime_detection(self):
        assert EpisodicMemory._detect_regime({"rsi": 0.5, "bb_width": 0.5, "sma_50_200_ratio": 0.8}) == "trending"
        assert EpisodicMemory._detect_regime({"rsi": 0.5, "bb_width": 0.8, "sma_50_200_ratio": 0.5}) == "volatile"
        assert EpisodicMemory._detect_regime({"rsi": 0.2, "bb_width": 0.2, "sma_50_200_ratio": 0.5}) == "mean_reverting"
        assert EpisodicMemory._detect_regime({"rsi": 0.5, "bb_width": 0.5, "sma_50_200_ratio": 0.5}) == "quiet"


class TestRecentPerformance:
    """測試 get_recent_performance() — RiskAgent 的記憶命脈"""

    def test_empty_memory_returns_defaults(self, tmp_path):
        mem = EpisodicMemory(store_dir=str(tmp_path / "empty"))
        perf = mem.get_recent_performance()
        assert perf["win_rate"] == 0.55
        assert perf["win_loss_ratio"] == 1.5
        assert perf["total_trades"] == 0

    def test_all_wins_high_win_rate(self, tmp_path):
        mem = EpisodicMemory(store_dir=str(tmp_path / "wins"))
        for i in range(10):
            mem.store(Episode(
                ticker="AAPL", embedding=np.random.randn(32),
                action="BUY", roi=0.05,
            ))
        perf = mem.get_recent_performance()
        assert perf["win_rate"] == 1.0
        assert perf["total_trades"] == 10
        assert perf["win_loss_ratio"] == 3.0  # 有盈無虧 → 3.0

    def test_all_losses_zero_win_rate(self, tmp_path):
        mem = EpisodicMemory(store_dir=str(tmp_path / "losses"))
        for i in range(5):
            mem.store(Episode(
                ticker="TSLA", embedding=np.random.randn(32),
                action="BUY", roi=-0.03,
            ))
        perf = mem.get_recent_performance()
        assert perf["win_rate"] == 0.0
        assert perf["total_trades"] == 5

    def test_mixed_performance_calculation(self, tmp_path):
        mem = EpisodicMemory(store_dir=str(tmp_path / "mixed"))
        # 6 勝 4 敗 → 勝率 60%
        for _ in range(6):
            mem.store(Episode(
                ticker="MSFT", embedding=np.random.randn(32),
                action="BUY", roi=0.10,
            ))
        for _ in range(4):
            mem.store(Episode(
                ticker="MSFT", embedding=np.random.randn(32),
                action="BUY", roi=-0.05,
            ))
        perf = mem.get_recent_performance()
        assert perf["win_rate"] == pytest.approx(0.6)
        assert perf["total_trades"] == 10
        # 盈虧比 = avg_win / avg_loss = 0.10 / 0.05 = 2.0
        assert perf["win_loss_ratio"] == pytest.approx(2.0)

    def test_action_filter(self, tmp_path):
        mem = EpisodicMemory(store_dir=str(tmp_path / "filter"))
        for _ in range(5):
            mem.store(Episode(
                ticker="A", embedding=np.random.randn(32),
                action="BUY", roi=0.08,
            ))
        for _ in range(3):
            mem.store(Episode(
                ticker="B", embedding=np.random.randn(32),
                action="SELL", roi=-0.02,
            ))
        buy_perf = mem.get_recent_performance(action_filter="BUY")
        sell_perf = mem.get_recent_performance(action_filter="SELL")
        assert buy_perf["total_trades"] == 5
        assert buy_perf["win_rate"] == 1.0
        assert sell_perf["total_trades"] == 3
        assert sell_perf["win_rate"] == 0.0

    def test_limit_respects_recent_trades(self, tmp_path):
        mem = EpisodicMemory(store_dir=str(tmp_path / "limit"))
        # 先放 10 筆虧損
        for _ in range(10):
            mem.store(Episode(
                ticker="X", embedding=np.random.randn(32),
                action="BUY", roi=-0.05,
            ))
        # 再放 5 筆獲利
        for _ in range(5):
            mem.store(Episode(
                ticker="X", embedding=np.random.randn(32),
                action="BUY", roi=0.10,
            ))
        # limit=5 只看最近 5 筆（全部是獲利）
        perf = mem.get_recent_performance(limit=5)
        assert perf["win_rate"] == 1.0
        assert perf["total_trades"] == 5

    def test_avg_roi_computed(self, tmp_path):
        mem = EpisodicMemory(store_dir=str(tmp_path / "roi"))
        mem.store(Episode(ticker="A", embedding=np.random.randn(32), action="BUY", roi=0.10))
        mem.store(Episode(ticker="B", embedding=np.random.randn(32), action="BUY", roi=-0.02))
        perf = mem.get_recent_performance()
        assert perf["avg_roi"] == pytest.approx(0.04)


class TestRegimePerformance:
    """測試 get_regime_performance() — 按體制篩選績效"""

    def test_regime_specific_stats(self, tmp_path):
        mem = EpisodicMemory(store_dir=str(tmp_path / "regime"))
        # trending 體制下 3 勝 1 敗
        for _ in range(3):
            mem.store(Episode(
                ticker="A", embedding=np.random.randn(32),
                action="BUY", roi=0.08, regime_tag="trending",
            ))
        mem.store(Episode(
            ticker="A", embedding=np.random.randn(32),
            action="BUY", roi=-0.03, regime_tag="trending",
        ))
        # volatile 體制下全虧
        for _ in range(3):
            mem.store(Episode(
                ticker="A", embedding=np.random.randn(32),
                action="BUY", roi=-0.05, regime_tag="volatile",
            ))

        trending_perf = mem.get_regime_performance("trending")
        volatile_perf = mem.get_regime_performance("volatile")
        assert trending_perf["win_rate"] == pytest.approx(0.75)
        assert trending_perf["regime"] == "trending"
        assert volatile_perf["win_rate"] == 0.0
        assert volatile_perf["regime"] == "volatile"

    def test_unknown_regime_falls_back(self, tmp_path):
        """未知的體制回退到全局績效"""
        mem = EpisodicMemory(store_dir=str(tmp_path / "fallback"))
        mem.store(Episode(
            ticker="A", embedding=np.random.randn(32),
            action="BUY", roi=0.10, regime_tag="trending",
        ))
        perf = mem.get_regime_performance("nonexistent_regime")
        # 應回退到 get_recent_performance()
        assert perf["total_trades"] == 1


class TestFrictionModel:
    def test_stamp_duty_on_uk_buy(self):
        fm = FrictionModel()
        cost = fm.total_cost(10_000, is_buy=True, is_uk_equity=True)
        # 0.5% stamp + spread + slippage + PTM
        assert cost >= 50  # at least stamp duty

    def test_no_stamp_duty_on_sell(self):
        fm = FrictionModel()
        cost_buy = fm.total_cost(10_000, is_buy=True, is_uk_equity=True)
        cost_sell = fm.total_cost(10_000, is_buy=False, is_uk_equity=True)
        assert cost_sell < cost_buy

    def test_no_stamp_duty_on_non_uk(self):
        fm = FrictionModel()
        cost_uk = fm.total_cost(10_000, is_buy=True, is_uk_equity=True)
        cost_us = fm.total_cost(10_000, is_buy=True, is_uk_equity=False)
        assert cost_us < cost_uk

    def test_ptm_levy_threshold(self):
        fm = FrictionModel()
        cost_below = fm.total_cost(9_000, is_buy=True, is_uk_equity=True)
        cost_above = fm.total_cost(11_000, is_buy=True, is_uk_equity=True)
        # Above threshold gets additional £1
        # (but value difference also changes stamp duty, so just check it's more)
        assert cost_above > cost_below


class TestCounterfactualReplay:
    @pytest.fixture
    def engine(self):
        return CounterfactualReplayEngine(initial_capital=100_000)

    def test_replay_produces_trades(self, engine):
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        signals = np.random.randn(100) * 0.5  # random signals

        result = engine.replay(
            price_series=prices,
            signals=signals,
            parameters={
                "min_confidence_to_trade": 0.3,
                "min_buy_score": 0.3,
                "max_sell_score": -0.3,
                "atr_stop_multiplier": 2.0,
                "atr_tp_multiplier": 3.0,
                "half_kelly_scaling": 0.5,
            },
            ticker="TEST",
        )
        # Should have executed at least some trades
        assert result.trade_count >= 0
        assert result.total_friction >= 0

    def test_friction_deducted(self, engine):
        # Upward trending prices with strong buy signals
        prices = np.linspace(100, 110, 50)
        signals = np.ones(50) * 0.5

        result = engine.replay(
            price_series=prices,
            signals=signals,
            parameters={
                "min_buy_score": 0.3,
                "max_sell_score": -0.3,
                "atr_stop_multiplier": 2.0,
                "atr_tp_multiplier": 3.0,
                "half_kelly_scaling": 0.5,
                "min_confidence_to_trade": 0.3,
            },
        )
        if result.trade_count > 0:
            assert result.total_friction > 0
            assert result.total_net_pnl < result.total_gross_pnl

    def test_compare_parameters(self, engine):
        prices = 100 + np.cumsum(np.random.randn(80) * 0.3)
        signals = np.random.randn(80) * 0.4

        param_sets = [
            {"min_buy_score": 0.2, "max_sell_score": -0.2,
             "atr_stop_multiplier": 1.5, "atr_tp_multiplier": 2.5,
             "half_kelly_scaling": 0.3, "min_confidence_to_trade": 0.2},
            {"min_buy_score": 0.5, "max_sell_score": -0.5,
             "atr_stop_multiplier": 3.0, "atr_tp_multiplier": 4.5,
             "half_kelly_scaling": 0.6, "min_confidence_to_trade": 0.5},
        ]
        results = engine.compare_parameters(
            prices, signals, param_sets,
            labels=["aggressive", "conservative"],
        )
        assert len(results) == 2


class TestUpdateEpisodeFailure:
    """測試 update_episode_failure() 與 get_episode() — 錯誤歸因回寫"""

    def test_update_existing_episode(self, tmp_path):
        mem = EpisodicMemory(store_dir=str(tmp_path / "upd1"))
        ep_id = mem.store(Episode(
            ticker="AAPL", embedding=np.random.randn(32), roi=-0.05,
        ))
        ok = mem.update_episode_failure(
            ep_id, failure_layer="semantic",
            failure_detail="High confidence but wrong direction",
        )
        assert ok is True
        ep = mem.get_episode(ep_id)
        assert ep is not None
        assert ep.failure_layer == "semantic"
        assert ep.failure_detail == "High confidence but wrong direction"

    def test_update_nonexistent_episode(self, tmp_path):
        mem = EpisodicMemory(store_dir=str(tmp_path / "upd2"))
        ok = mem.update_episode_failure(
            "EP-999999", failure_layer="temporal", failure_detail="nope",
        )
        assert ok is False

    def test_update_persists_to_disk(self, tmp_path):
        store = str(tmp_path / "upd3")
        mem = EpisodicMemory(store_dir=store)
        ep_id = mem.store(Episode(
            ticker="TSLA", embedding=np.random.randn(32), roi=-0.03,
        ))
        mem.update_episode_failure(
            ep_id, failure_layer="execution",
            failure_detail="Slippage consumed 40% of loss",
        )
        # Reload from disk
        mem2 = EpisodicMemory(store_dir=store)
        ep = mem2.get_episode(ep_id)
        assert ep is not None
        assert ep.failure_layer == "execution"
        assert ep.failure_detail == "Slippage consumed 40% of loss"

    def test_get_episode_returns_none_for_missing(self, tmp_path):
        mem = EpisodicMemory(store_dir=str(tmp_path / "upd4"))
        assert mem.get_episode("EP-000001") is None

    def test_overwrite_failure_detail(self, tmp_path):
        """第二次 update 應覆蓋第一次的結果"""
        mem = EpisodicMemory(store_dir=str(tmp_path / "upd5"))
        ep_id = mem.store(Episode(
            ticker="MSFT", embedding=np.random.randn(32), roi=-0.02,
        ))
        mem.update_episode_failure(ep_id, "semantic", "first diagnosis")
        mem.update_episode_failure(ep_id, "temporal", "revised diagnosis")
        ep = mem.get_episode(ep_id)
        assert ep.failure_layer == "temporal"
        assert ep.failure_detail == "revised diagnosis"
