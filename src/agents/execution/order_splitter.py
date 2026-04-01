"""
Order Splitter — TWAP/VWAP 拆單引擎
=====================================
將大額訂單拆分為多個小訂單，減少市場衝擊。

策略：
  - TWAP (Time-Weighted Average Price): 等量分時拆單
  - VWAP (Volume-Weighted Average Price): 按歷史成交量分佈拆單

使用者：
  - ExecutionAgent：大單拆分執行
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class SplitStrategy(Enum):
    TWAP = auto()
    VWAP = auto()


@dataclass
class OrderSlice:
    """拆單中的單個子訂單。"""
    slice_index: int
    quantity: float
    scheduled_time: float       # Unix timestamp
    status: str = "pending"     # pending | submitted | filled | failed
    fill_price: float | None = None
    submitted_at: float | None = None
    filled_at: float | None = None


@dataclass
class SplitPlan:
    """完整的拆單計劃。"""
    strategy: SplitStrategy
    original_quantity: float
    slices: list[OrderSlice] = field(default_factory=list)
    total_filled: float = 0.0
    avg_fill_price: float = 0.0

    @property
    def num_slices(self) -> int:
        return len(self.slices)

    @property
    def is_complete(self) -> bool:
        return all(s.status in ("filled", "failed") for s in self.slices)

    @property
    def pending_slices(self) -> list[OrderSlice]:
        return [s for s in self.slices if s.status == "pending"]

    @property
    def next_slice(self) -> OrderSlice | None:
        """取得下一個待執行且已到排程時間的子訂單。"""
        now = time.time()
        for s in self.slices:
            if s.status == "pending" and s.scheduled_time <= now:
                return s
        return None

    def record_fill(self, slice_index: int, fill_price: float) -> None:
        """記錄子訂單成交。"""
        for s in self.slices:
            if s.slice_index == slice_index:
                s.status = "filled"
                s.fill_price = fill_price
                s.filled_at = time.time()
                break

        # Update aggregate stats
        filled = [s for s in self.slices if s.status == "filled" and s.fill_price]
        if filled:
            self.total_filled = sum(s.quantity for s in filled)
            total_value = sum(s.quantity * s.fill_price for s in filled)
            self.avg_fill_price = total_value / self.total_filled if self.total_filled > 0 else 0.0

    @property
    def summary(self) -> str:
        filled = sum(1 for s in self.slices if s.status == "filled")
        failed = sum(1 for s in self.slices if s.status == "failed")
        pending = sum(1 for s in self.slices if s.status == "pending")
        return (
            f"{self.strategy.name} plan: {self.num_slices} slices "
            f"(filled={filled}, failed={failed}, pending={pending}) "
            f"avg_price={self.avg_fill_price:.4f}"
        )


class OrderSplitter:
    """
    訂單拆分引擎。

    根據訂單金額決定是否需要拆單，並生成 TWAP 或 VWAP 執行計劃。

    Parameters
    ----------
    split_threshold : 超過此金額的訂單會被拆分（預設 $5000）
    min_slice_value : 單個子訂單最低金額（預設 $100）
    max_slices : 最大拆分數量
    twap_interval_seconds : TWAP 每個子訂單之間的間隔
    """

    def __init__(
        self,
        split_threshold: float = 5000.0,
        min_slice_value: float = 100.0,
        max_slices: int = 10,
        twap_interval_seconds: float = 300.0,
    ) -> None:
        self.split_threshold = split_threshold
        self.min_slice_value = min_slice_value
        self.max_slices = max_slices
        self.twap_interval = twap_interval_seconds

    def should_split(
        self, quantity: float, price: float,
    ) -> bool:
        """判斷訂單是否需要拆分。"""
        estimated_value = quantity * price
        return estimated_value > self.split_threshold

    def create_twap_plan(
        self,
        quantity: float,
        price: float,
        num_slices: int | None = None,
        start_time: float | None = None,
    ) -> SplitPlan:
        """
        建立 TWAP 拆單計劃 — 等量分時。

        Parameters
        ----------
        quantity : 總數量
        price : 當前價格（用於計算最小 slice）
        num_slices : 拆分數量（若為 None，自動計算）
        start_time : 開始時間（Unix timestamp）
        """
        if num_slices is None:
            estimated_value = quantity * price
            num_slices = min(
                self.max_slices,
                max(2, int(estimated_value / self.min_slice_value)),
            )

        num_slices = max(2, min(num_slices, self.max_slices))
        base_qty = math.floor(quantity / num_slices * 100) / 100  # 2 decimal places
        remainder = quantity - base_qty * num_slices

        start = start_time or time.time()
        slices: list[OrderSlice] = []

        for i in range(num_slices):
            qty = base_qty
            if i == num_slices - 1:
                qty = base_qty + remainder  # last slice absorbs remainder

            slices.append(OrderSlice(
                slice_index=i,
                quantity=round(qty, 6),
                scheduled_time=start + i * self.twap_interval,
            ))

        plan = SplitPlan(
            strategy=SplitStrategy.TWAP,
            original_quantity=quantity,
            slices=slices,
        )

        logger.info(
            "[OrderSplitter] TWAP plan: %.2f qty → %d slices @ %.0fs interval",
            quantity, num_slices, self.twap_interval,
        )
        return plan

    def create_vwap_plan(
        self,
        quantity: float,
        price: float,
        volume_profile: list[float],
        start_time: float | None = None,
    ) -> SplitPlan:
        """
        建立 VWAP 拆單計劃 — 按成交量分佈。

        Parameters
        ----------
        quantity : 總數量
        price : 當前價格
        volume_profile : 各時段的相對成交量比例
            e.g. [0.3, 0.2, 0.15, 0.15, 0.2] → 5 個時段
        start_time : 開始時間
        """
        if not volume_profile:
            return self.create_twap_plan(quantity, price, start_time=start_time)

        # Normalize volume profile
        total_vol = sum(volume_profile)
        if total_vol <= 0:
            return self.create_twap_plan(quantity, price, start_time=start_time)

        weights = [v / total_vol for v in volume_profile]
        num_slices = min(len(weights), self.max_slices)
        weights = weights[:num_slices]

        # Re-normalize after truncation
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

        start = start_time or time.time()
        slices: list[OrderSlice] = []
        allocated = 0.0

        for i, w in enumerate(weights):
            if i == num_slices - 1:
                qty = quantity - allocated
            else:
                qty = round(quantity * w, 6)
                allocated += qty

            qty = max(0.0, qty)

            slices.append(OrderSlice(
                slice_index=i,
                quantity=qty,
                scheduled_time=start + i * self.twap_interval,
            ))

        plan = SplitPlan(
            strategy=SplitStrategy.VWAP,
            original_quantity=quantity,
            slices=slices,
        )

        logger.info(
            "[OrderSplitter] VWAP plan: %.2f qty → %d slices (weighted)",
            quantity, num_slices,
        )
        return plan
