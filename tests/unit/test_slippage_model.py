"""
Unit Tests for SlippagePredictor (滑點預測模型)
================================================
測試：
  1. 基本預測計算
  2. 市場衝擊因子
  3. 時段因子
  4. 歷史偏差校正（學習）
  5. 訂單縮減判斷
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.agents.execution.slippage_model import SlippagePredictor, SlippagePrediction


class TestPredict:
    def test_basic_prediction(self):
        model = SlippagePredictor()
        pred = model.predict(
            order_value=1000,
            avg_daily_volume=1_000_000,
            current_price=100.0,
        )
        assert isinstance(pred, SlippagePrediction)
        assert pred.expected_slippage_bps >= 0
        assert pred.confidence_interval[0] <= pred.expected_slippage_bps
        assert pred.confidence_interval[1] >= pred.expected_slippage_bps

    def test_large_order_higher_slippage(self):
        model = SlippagePredictor()
        small = model.predict(100, 1_000_000, 100.0)
        large = model.predict(100_000, 1_000_000, 100.0)
        assert large.expected_slippage_bps > small.expected_slippage_bps

    def test_high_volatility_higher_slippage(self):
        model = SlippagePredictor()
        low_vol = model.predict(1000, 1_000_000, 100.0, atr=1.0)
        high_vol = model.predict(1000, 1_000_000, 100.0, atr=5.0)
        assert high_vol.expected_slippage_bps > low_vol.expected_slippage_bps

    def test_time_penalty_at_open(self):
        model = SlippagePredictor()
        # 9:45 AM ET = 13:45 UTC — within 30 min of US open (13:30 UTC)
        open_time = datetime(2026, 1, 5, 13, 45, tzinfo=timezone.utc)
        midday = datetime(2026, 1, 5, 17, 0, tzinfo=timezone.utc)

        pred_open = model.predict(1000, 1_000_000, 100.0, time_of_day=open_time)
        pred_mid = model.predict(1000, 1_000_000, 100.0, time_of_day=midday)
        # Open time has time penalty factor > 0
        assert pred_open.factors["time_of_day"] > 0
        # Midday has no time penalty
        assert pred_mid.factors["time_of_day"] == 0

    def test_factors_populated(self):
        model = SlippagePredictor()
        pred = model.predict(1000, 1_000_000, 100.0, atr=2.0)
        assert "market_impact" in pred.factors
        assert "volatility" in pred.factors
        assert "time_of_day" in pred.factors


class TestRecordObservation:
    def test_bias_correction(self):
        model = SlippagePredictor()
        assert model._historical_bias == 0.0

        # Actual slippage consistently higher than predicted
        for _ in range(10):
            model.record_observation(predicted_bps=5.0, actual_slippage_bps=10.0)

        assert model._historical_bias > 0  # Bias should shift positive

    def test_observation_limit(self):
        model = SlippagePredictor()
        for i in range(600):
            model.record_observation(predicted_bps=5.0, actual_slippage_bps=5.0 + i * 0.01)
        assert len(model._observations) <= 500


class TestShouldReduceSize:
    def test_below_threshold(self):
        model = SlippagePredictor(max_acceptable_slippage_bps=50.0)
        pred = SlippagePrediction(
            expected_slippage_bps=30.0,
            confidence_interval=(20.0, 40.0),
        )
        assert not model.should_reduce_size(pred)

    def test_above_threshold(self):
        model = SlippagePredictor(max_acceptable_slippage_bps=50.0)
        pred = SlippagePrediction(
            expected_slippage_bps=80.0,
            confidence_interval=(60.0, 100.0),
        )
        assert model.should_reduce_size(pred)


class TestAdjustedQuantity:
    def test_no_reduction_when_below_threshold(self):
        model = SlippagePredictor(max_acceptable_slippage_bps=50.0)
        pred = SlippagePrediction(
            expected_slippage_bps=30.0,
            confidence_interval=(20.0, 40.0),
        )
        assert model.adjusted_quantity(100, pred) == 100

    def test_reduction_when_above_threshold(self):
        model = SlippagePredictor(max_acceptable_slippage_bps=50.0)
        pred = SlippagePrediction(
            expected_slippage_bps=100.0,
            confidence_interval=(80.0, 120.0),
        )
        adjusted = model.adjusted_quantity(100, pred)
        assert adjusted < 100
        assert adjusted >= 1  # minimum 1


class TestStats:
    def test_empty_stats(self):
        model = SlippagePredictor()
        stats = model.stats
        assert stats["observations"] == 0

    def test_stats_after_observations(self):
        model = SlippagePredictor()
        model.record_observation(5.0, 7.0)
        model.record_observation(5.0, 3.0)
        stats = model.stats
        assert stats["observations"] == 2
        assert "mean_error" in stats
