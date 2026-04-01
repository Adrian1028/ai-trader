"""
Unit Tests for OrderSplitter (TWAP/VWAP 拆單引擎)
===================================================
測試：
  1. should_split 判斷
  2. TWAP 拆單計劃
  3. VWAP 拆單計劃
  4. 邊界條件
  5. SplitPlan 記錄與統計
"""
from __future__ import annotations

import time

import pytest

from src.agents.execution.order_splitter import (
    OrderSplitter,
    SplitPlan,
    SplitStrategy,
)


class TestShouldSplit:
    def test_below_threshold_no_split(self):
        splitter = OrderSplitter(split_threshold=5000)
        assert not splitter.should_split(10, 100.0)  # $1000

    def test_above_threshold_split(self):
        splitter = OrderSplitter(split_threshold=5000)
        assert splitter.should_split(100, 100.0)  # $10000

    def test_at_threshold_no_split(self):
        splitter = OrderSplitter(split_threshold=5000)
        assert not splitter.should_split(50, 100.0)  # $5000 exactly


class TestTWAP:
    def test_basic_twap(self):
        splitter = OrderSplitter(twap_interval_seconds=60)
        plan = splitter.create_twap_plan(100, 100.0, num_slices=5)

        assert plan.strategy == SplitStrategy.TWAP
        assert plan.num_slices == 5
        assert plan.original_quantity == 100

        total_qty = sum(s.quantity for s in plan.slices)
        assert total_qty == pytest.approx(100, abs=0.01)

    def test_twap_timing(self):
        splitter = OrderSplitter(twap_interval_seconds=120)
        start = time.time()
        plan = splitter.create_twap_plan(100, 100.0, num_slices=3, start_time=start)

        assert plan.slices[0].scheduled_time == pytest.approx(start)
        assert plan.slices[1].scheduled_time == pytest.approx(start + 120)
        assert plan.slices[2].scheduled_time == pytest.approx(start + 240)

    def test_auto_slice_count(self):
        splitter = OrderSplitter(min_slice_value=100)
        plan = splitter.create_twap_plan(50, 100.0)  # $5000
        assert 2 <= plan.num_slices <= splitter.max_slices

    def test_min_two_slices(self):
        splitter = OrderSplitter()
        plan = splitter.create_twap_plan(10, 100.0, num_slices=1)
        assert plan.num_slices >= 2


class TestVWAP:
    def test_basic_vwap(self):
        splitter = OrderSplitter()
        profile = [0.3, 0.2, 0.1, 0.2, 0.2]
        plan = splitter.create_vwap_plan(100, 100.0, profile)

        assert plan.strategy == SplitStrategy.VWAP
        assert plan.num_slices == 5
        total_qty = sum(s.quantity for s in plan.slices)
        assert total_qty == pytest.approx(100, abs=0.01)

    def test_vwap_proportional(self):
        splitter = OrderSplitter()
        profile = [0.5, 0.5]
        plan = splitter.create_vwap_plan(100, 100.0, profile)

        # Both slices should be roughly equal
        assert abs(plan.slices[0].quantity - plan.slices[1].quantity) < 1.0

    def test_empty_profile_fallback_to_twap(self):
        splitter = OrderSplitter()
        plan = splitter.create_vwap_plan(100, 100.0, [])
        assert plan.strategy == SplitStrategy.TWAP


class TestSplitPlan:
    def test_record_fill(self):
        splitter = OrderSplitter()
        plan = splitter.create_twap_plan(100, 100.0, num_slices=3)

        plan.record_fill(0, 99.5)
        assert plan.slices[0].status == "filled"
        assert plan.total_filled > 0
        assert plan.avg_fill_price == pytest.approx(99.5)

    def test_is_complete(self):
        splitter = OrderSplitter()
        plan = splitter.create_twap_plan(100, 100.0, num_slices=2)

        assert not plan.is_complete
        plan.record_fill(0, 100.0)
        assert not plan.is_complete
        plan.record_fill(1, 100.0)
        assert plan.is_complete

    def test_next_slice(self):
        splitter = OrderSplitter(twap_interval_seconds=1)
        now = time.time()
        plan = splitter.create_twap_plan(100, 100.0, num_slices=2, start_time=now - 10)

        nxt = plan.next_slice
        assert nxt is not None
        assert nxt.slice_index == 0
