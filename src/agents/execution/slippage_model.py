"""
Slippage Prediction Model — 滑點預測引擎
=========================================
根據歷史觀測與市場微觀結構特徵，預測訂單的預期滑點。

特徵維度：
  1. 訂單規模 vs 平均成交量（市場衝擊）
  2. 當前波動率（ATR-based）
  3. 交易時段（開盤/收盤時段滑點較大）
  4. 歷史滑點觀測（自我學習）

使用者：
  - ExecutionAgent：下單前預測滑點，過大時縮減數量
  - AuditTrail：記錄預測 vs 實際滑點，回饋模型
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SlippagePrediction:
    """單筆訂單的滑點預測結果。"""
    expected_slippage_bps: float                    # 預期滑點（基點）
    confidence_interval: tuple[float, float]        # (低, 高) 信賴區間
    factors: dict[str, float] = field(default_factory=dict)  # 各因子貢獻

    @property
    def summary(self) -> str:
        return (
            f"Expected slippage: {self.expected_slippage_bps:.1f} bps "
            f"(CI: {self.confidence_interval[0]:.1f}–"
            f"{self.confidence_interval[1]:.1f} bps)"
        )


class SlippagePredictor:
    """
    滑點預測模型。

    基於多因子線性模型預測訂單滑點，並透過歷史觀測持續校正。

    因子：
      - market_impact: 訂單金額 / 日均成交額 → 大單衝擊
      - volatility: ATR% → 高波動 = 高滑點
      - time_of_day: 開盤/收盤時段 → 流動性較差
      - historical_bias: 歷史觀測偏差的移動平均

    使用方式：
      1. 下單前呼叫 predict() 取得預測
      2. 成交後呼叫 record_observation() 回饋實際滑點
      3. 模型自動校正 historical_bias
    """

    # 因子係數（基點）
    _MARKET_IMPACT_COEFF = 50.0    # 每 1% 的成交量佔比 → 50 bps
    _VOLATILITY_COEFF = 10.0       # 每 1% ATR → 10 bps
    _TIME_PENALTY_BPS = 5.0        # 開盤/收盤時段額外 5 bps
    _BASE_SLIPPAGE_BPS = 2.0       # 基礎滑點（最低值）

    # 學習參數
    _MAX_OBSERVATIONS = 500        # 最多保留的歷史觀測數
    _LEARNING_RATE = 0.1           # 偏差校正的指數移動平均係數

    def __init__(self, max_acceptable_slippage_bps: float = 50.0) -> None:
        self.max_acceptable_slippage_bps = max_acceptable_slippage_bps
        self._observations: list[dict[str, Any]] = []
        self._historical_bias: float = 0.0  # EMA of (actual - predicted)

    def predict(
        self,
        order_value: float,
        avg_daily_volume: float,
        current_price: float,
        atr: float | None = None,
        time_of_day: datetime | None = None,
    ) -> SlippagePrediction:
        """
        預測訂單的預期滑點。

        Parameters
        ----------
        order_value : 訂單金額 (USD/GBP)
        avg_daily_volume : 日均成交額
        current_price : 當前價格
        atr : 14 日 ATR（可選）
        time_of_day : 下單時間（可選，用於判斷開盤/收盤時段）
        """
        factors: dict[str, float] = {}

        # Factor 1: Market impact
        if avg_daily_volume > 0:
            volume_pct = (order_value / avg_daily_volume) * 100.0
            impact_bps = volume_pct * self._MARKET_IMPACT_COEFF
        else:
            volume_pct = 0.0
            impact_bps = self._BASE_SLIPPAGE_BPS
        factors["market_impact"] = impact_bps

        # Factor 2: Volatility
        vol_bps = 0.0
        if atr is not None and current_price > 0:
            atr_pct = (atr / current_price) * 100.0
            vol_bps = atr_pct * self._VOLATILITY_COEFF
        factors["volatility"] = vol_bps

        # Factor 3: Time of day penalty (all times in UTC)
        time_bps = 0.0
        if time_of_day is not None:
            hour = time_of_day.hour
            minute = time_of_day.minute
            total_minutes = hour * 60 + minute
            # US market (UTC): open 13:30, close 20:00
            us_open_utc = 13 * 60 + 30
            us_close_utc = 20 * 60
            if (us_open_utc <= total_minutes < us_open_utc + 30
                    or us_close_utc - 15 < total_minutes <= us_close_utc):
                time_bps = self._TIME_PENALTY_BPS
            # UK market (UTC): open 08:00, close 16:30
            uk_open_utc = 8 * 60
            uk_close_utc = 16 * 60 + 30
            if (uk_open_utc <= total_minutes < uk_open_utc + 30
                    or uk_close_utc - 15 < total_minutes <= uk_close_utc):
                time_bps = max(time_bps, self._TIME_PENALTY_BPS)
        factors["time_of_day"] = time_bps

        # Factor 4: Historical bias correction
        factors["historical_bias"] = self._historical_bias

        # Combine
        expected = (
            self._BASE_SLIPPAGE_BPS
            + impact_bps
            + vol_bps
            + time_bps
            + self._historical_bias
        )
        expected = max(0.0, expected)

        # Confidence interval (rough ±50% or minimum ±2 bps)
        margin = max(expected * 0.5, 2.0)
        ci = (max(0.0, expected - margin), expected + margin)

        prediction = SlippagePrediction(
            expected_slippage_bps=expected,
            confidence_interval=ci,
            factors=factors,
        )

        logger.debug(
            "[SlippagePredictor] %s | order=%.0f vol_pct=%.3f%%",
            prediction.summary, order_value, volume_pct,
        )
        return prediction

    def record_observation(
        self,
        predicted_bps: float,
        actual_slippage_bps: float,
        order_details: dict[str, Any] | None = None,
    ) -> None:
        """
        記錄實際滑點，校正模型偏差。

        Parameters
        ----------
        predicted_bps : 預測滑點（基點）
        actual_slippage_bps : 實際滑點（基點）
        order_details : 附加訂單資訊（用於後續分析）
        """
        error = actual_slippage_bps - predicted_bps

        # EMA update
        self._historical_bias = (
            self._LEARNING_RATE * error
            + (1 - self._LEARNING_RATE) * self._historical_bias
        )

        obs = {
            "predicted_bps": predicted_bps,
            "actual_bps": actual_slippage_bps,
            "error": error,
            "timestamp": time.time(),
        }
        if order_details:
            obs["details"] = order_details

        self._observations.append(obs)

        # Trim old observations
        if len(self._observations) > self._MAX_OBSERVATIONS:
            self._observations = self._observations[-self._MAX_OBSERVATIONS:]

        logger.info(
            "[SlippagePredictor] 觀測記錄: predicted=%.1f actual=%.1f "
            "error=%+.1f bias=%+.2f",
            predicted_bps, actual_slippage_bps, error, self._historical_bias,
        )

    def should_reduce_size(self, prediction: SlippagePrediction) -> bool:
        """判斷是否應因滑點過大而縮減訂單規模。"""
        return prediction.expected_slippage_bps > self.max_acceptable_slippage_bps

    def adjusted_quantity(
        self,
        original_quantity: float,
        prediction: SlippagePrediction,
    ) -> float:
        """
        根據滑點預測調整訂單數量。

        若預期滑點超過閾值，按比例縮減數量。
        """
        if not self.should_reduce_size(prediction):
            return original_quantity

        # Scale down proportionally
        ratio = self.max_acceptable_slippage_bps / prediction.expected_slippage_bps
        adjusted = original_quantity * ratio
        adjusted = max(1.0, math.floor(adjusted))

        logger.info(
            "[SlippagePredictor] 訂單縮減: %.0f → %.0f (slippage %.1f > %.1f bps)",
            original_quantity, adjusted,
            prediction.expected_slippage_bps, self.max_acceptable_slippage_bps,
        )
        return adjusted

    @property
    def stats(self) -> dict[str, Any]:
        """模型統計資訊。"""
        if not self._observations:
            return {
                "observations": 0,
                "historical_bias": self._historical_bias,
            }

        errors = [o["error"] for o in self._observations]
        return {
            "observations": len(self._observations),
            "historical_bias": self._historical_bias,
            "mean_error": sum(errors) / len(errors),
            "max_error": max(errors),
            "min_error": min(errors),
        }
