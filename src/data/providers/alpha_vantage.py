"""
Alpha Vantage 市場情報感測器 (Data Scout)
==========================================
長週期歷史 K 線 (OHLCV) + 基本面數據 + 技術指標。

職責：
  - 日/週/月 K 線 → 標準化 OHLCVBar 清單
  - 公司概覽 (P/E, ROE, Market Cap...)
  - 財務報表 (Income Statement, Balance Sheet, Cash Flow)
  - 盈餘數據 (Earnings, EPS Surprise)
  - 技術指標 (SMA, RSI, MACD, Bollinger Bands)

使用者：
  - FundamentalAgent：基本面分析
  - TechnicalAgent：技術指標 (日線以上)

API 限制：
  - 免費版：25 次/日
  - Premium：75-1200 次/分 (依方案)

所有方法都返回「乾淨的 Python 結構」，不返回 raw JSON，
讓上層 Agent 可以直接消費。
"""
from __future__ import annotations

import logging
from typing import Any

from .base_provider import BaseDataProvider, OHLCVBar, DataProviderError

_BASE = "https://www.alphavantage.co/query"

logger = logging.getLogger(__name__)


class AlphaVantageProvider(BaseDataProvider):
    """
    Alpha Vantage REST API 非同步客戶端。
    所有公開方法都返回標準化的 Python 資料結構。
    """

    def __init__(self, api_key: str, cache_ttl: float = 300.0) -> None:
        super().__init__("alpha_vantage", api_key, _BASE, cache_ttl_seconds=cache_ttl)

    async def health_check(self) -> bool:
        try:
            data = await self._call("TIME_SERIES_INTRADAY", symbol="IBM", interval="5min")
            return "Time Series (5min)" in data or "Meta Data" in data
        except Exception:
            return False

    # ══════════════════════════════════════════════════════════════
    # K 線資料 (OHLCV) — 標準化輸出
    # ══════════════════════════════════════════════════════════════

    async def daily(
        self,
        symbol: str,
        outputsize: str = "compact",
    ) -> list[OHLCVBar]:
        """
        取得每日 K 線，返回標準化 OHLCVBar 清單（由新到舊）。

        Parameters
        ----------
        symbol : 股票代號 (e.g. "AAPL")
        outputsize : "compact" (最近 100 天) 或 "full" (20+ 年)

        Returns
        -------
        list[OHLCVBar] : 標準化 K 線資料
        """
        raw = await self._call(
            "TIME_SERIES_DAILY_ADJUSTED", symbol=symbol, outputsize=outputsize,
        )
        return self._parse_time_series(raw, "Time Series (Daily)")

    async def weekly(self, symbol: str) -> list[OHLCVBar]:
        """取得每週 K 線"""
        raw = await self._call("TIME_SERIES_WEEKLY_ADJUSTED", symbol=symbol)
        return self._parse_time_series(raw, "Weekly Adjusted Time Series")

    async def monthly(self, symbol: str) -> list[OHLCVBar]:
        """取得每月 K 線"""
        raw = await self._call("TIME_SERIES_MONTHLY_ADJUSTED", symbol=symbol)
        return self._parse_time_series(raw, "Monthly Adjusted Time Series")

    async def daily_raw(
        self, symbol: str, outputsize: str = "compact",
    ) -> dict[str, Any]:
        """取得每日 K 線（原始 JSON，向後相容）"""
        return await self._call(
            "TIME_SERIES_DAILY_ADJUSTED", symbol=symbol, outputsize=outputsize,
        )

    # ══════════════════════════════════════════════════════════════
    # 基本面數據 (Fundamental Data)
    # ══════════════════════════════════════════════════════════════

    async def company_overview(self, symbol: str) -> dict[str, Any]:
        """
        公司概覽，包含：
          - P/E, Forward P/E, PEG Ratio
          - Profit Margin, ROE, ROA
          - Market Cap, Beta, 52-Week Range
          - EPS, Dividend Yield
        """
        raw = await self._call("OVERVIEW", symbol=symbol)
        if not raw or "Symbol" not in raw:
            self.logger.warning("company_overview 無資料: %s", symbol)
            return {}

        # 轉換關鍵欄位為正確的數值類型
        return self._parse_overview(raw)

    async def income_statement(self, symbol: str) -> dict[str, Any]:
        """損益表（年度 + 季度）"""
        return await self._call("INCOME_STATEMENT", symbol=symbol)

    async def balance_sheet(self, symbol: str) -> dict[str, Any]:
        """資產負債表"""
        return await self._call("BALANCE_SHEET", symbol=symbol)

    async def cash_flow(self, symbol: str) -> dict[str, Any]:
        """現金流量表"""
        return await self._call("CASH_FLOW", symbol=symbol)

    async def earnings(self, symbol: str) -> dict[str, Any]:
        """盈餘數據（年度 + 季度 EPS）"""
        return await self._call("EARNINGS", symbol=symbol)

    async def earnings_history(self, symbol: str) -> list[dict[str, Any]]:
        """
        提取季度盈餘歷史，返回簡化清單：
        [{"date": "2024-01-25", "reported_eps": 2.18, "estimated_eps": 2.10, "surprise_pct": 3.81}, ...]
        """
        raw = await self._call("EARNINGS", symbol=symbol)
        quarterly = raw.get("quarterlyEarnings", [])
        result = []
        for q in quarterly:
            result.append({
                "date": q.get("reportedDate", ""),
                "fiscal_quarter": q.get("fiscalDateEnding", ""),
                "reported_eps": self._safe_float(q.get("reportedEPS")),
                "estimated_eps": self._safe_float(q.get("estimatedEPS")),
                "surprise_pct": self._safe_float(q.get("surprisePercentage")),
            })
        return result

    # ══════════════════════════════════════════════════════════════
    # 技術指標 (Technical Indicators)
    # ══════════════════════════════════════════════════════════════

    async def sma(
        self,
        symbol: str,
        interval: str = "daily",
        time_period: int = 20,
        series_type: str = "close",
    ) -> list[dict[str, Any]]:
        """
        簡單移動平均線 (SMA)。
        返回：[{"date": "2024-01-15", "sma": 185.23}, ...]
        """
        raw = await self._call(
            "SMA", symbol=symbol, interval=interval,
            time_period=str(time_period), series_type=series_type,
        )
        return self._parse_indicator(raw, "Technical Analysis: SMA", "SMA")

    async def ema(
        self,
        symbol: str,
        interval: str = "daily",
        time_period: int = 20,
        series_type: str = "close",
    ) -> list[dict[str, Any]]:
        """指數移動平均線 (EMA)"""
        raw = await self._call(
            "EMA", symbol=symbol, interval=interval,
            time_period=str(time_period), series_type=series_type,
        )
        return self._parse_indicator(raw, "Technical Analysis: EMA", "EMA")

    async def rsi(
        self,
        symbol: str,
        interval: str = "daily",
        time_period: int = 14,
        series_type: str = "close",
    ) -> list[dict[str, Any]]:
        """
        相對強弱指標 (RSI)。
        返回：[{"date": "2024-01-15", "rsi": 62.45}, ...]
        """
        raw = await self._call(
            "RSI", symbol=symbol, interval=interval,
            time_period=str(time_period), series_type=series_type,
        )
        return self._parse_indicator(raw, "Technical Analysis: RSI", "RSI")

    async def macd(
        self,
        symbol: str,
        interval: str = "daily",
        series_type: str = "close",
    ) -> list[dict[str, Any]]:
        """
        MACD。
        返回：[{"date": "...", "macd": ..., "signal": ..., "histogram": ...}, ...]
        """
        raw = await self._call(
            "MACD", symbol=symbol, interval=interval, series_type=series_type,
        )
        key = "Technical Analysis: MACD"
        series = raw.get(key, {})
        result = []
        for date_str, values in sorted(series.items(), reverse=True):
            result.append({
                "date": date_str,
                "macd": self._safe_float(values.get("MACD")),
                "signal": self._safe_float(values.get("MACD_Signal")),
                "histogram": self._safe_float(values.get("MACD_Hist")),
            })
        return result

    async def bbands(
        self,
        symbol: str,
        interval: str = "daily",
        time_period: int = 20,
        series_type: str = "close",
    ) -> list[dict[str, Any]]:
        """
        布林通道 (Bollinger Bands)。
        返回：[{"date": "...", "upper": ..., "middle": ..., "lower": ...}, ...]
        """
        raw = await self._call(
            "BBANDS", symbol=symbol, interval=interval,
            time_period=str(time_period), series_type=series_type,
        )
        key = "Technical Analysis: BBANDS"
        series = raw.get(key, {})
        result = []
        for date_str, values in sorted(series.items(), reverse=True):
            result.append({
                "date": date_str,
                "upper": self._safe_float(values.get("Real Upper Band")),
                "middle": self._safe_float(values.get("Real Middle Band")),
                "lower": self._safe_float(values.get("Real Lower Band")),
            })
        return result

    async def atr(
        self,
        symbol: str,
        interval: str = "daily",
        time_period: int = 14,
    ) -> list[dict[str, Any]]:
        """
        平均真實範圍 (ATR)。
        返回：[{"date": "...", "atr": ...}, ...]
        """
        raw = await self._call(
            "ATR", symbol=symbol, interval=interval,
            time_period=str(time_period),
        )
        return self._parse_indicator(raw, "Technical Analysis: ATR", "ATR")

    # ══════════════════════════════════════════════════════════════
    # 內部輔助方法
    # ══════════════════════════════════════════════════════════════

    async def _call(self, function: str, **kwargs: str) -> dict[str, Any]:
        """呼叫 Alpha Vantage API"""
        params = {"function": function, "apikey": self._api_key, **kwargs}
        data = await self._get(_BASE, params=params)

        # Alpha Vantage 在超過速率限制時返回特殊 JSON
        if isinstance(data, dict):
            if "Note" in data:
                self.logger.warning(
                    "[AlphaVantage] 速率限制訊息: %s", data["Note"][:100],
                )
            if "Error Message" in data:
                raise DataProviderError(
                    self.name, f"API Error: {data['Error Message']}",
                )

        return data

    def _parse_time_series(
        self,
        raw: dict[str, Any],
        series_key: str,
    ) -> list[OHLCVBar]:
        """將 Alpha Vantage 時間序列 JSON 轉換為標準化 OHLCVBar 清單"""
        series = raw.get(series_key, {})
        if not series:
            self.logger.warning(
                "時間序列資料為空 (key=%s), raw keys=%s",
                series_key, list(raw.keys()),
            )
            return []

        bars: list[OHLCVBar] = []
        for date_str, values in sorted(series.items(), reverse=True):
            bars.append(OHLCVBar(
                timestamp=date_str,
                open=self._safe_float(values.get("1. open")),
                high=self._safe_float(values.get("2. high")),
                low=self._safe_float(values.get("3. low")),
                close=self._safe_float(values.get("4. close")),
                volume=self._safe_float(values.get("6. volume", values.get("5. volume"))),
                adjusted_close=self._safe_float(values.get("5. adjusted close")),
            ))

        self.logger.debug(
            "解析 %d 根 K 線 (%s → %s)",
            len(bars),
            bars[-1].timestamp if bars else "?",
            bars[0].timestamp if bars else "?",
        )
        return bars

    def _parse_indicator(
        self,
        raw: dict[str, Any],
        series_key: str,
        value_key: str,
    ) -> list[dict[str, Any]]:
        """將技術指標 JSON 轉換為簡潔清單"""
        series = raw.get(series_key, {})
        result = []
        for date_str, values in sorted(series.items(), reverse=True):
            result.append({
                "date": date_str,
                value_key.lower(): self._safe_float(values.get(value_key)),
            })
        return result

    def _parse_overview(self, raw: dict[str, Any]) -> dict[str, Any]:
        """將公司概覽的字串值轉換為正確的數值類型"""
        float_fields = [
            "PERatio", "ForwardPE", "PEGRatio", "PriceToBookRatio",
            "PriceToSalesRatioTTM", "EVToRevenue", "EVToEBITDA",
            "ProfitMargin", "OperatingMarginTTM", "ReturnOnAssetsTTM",
            "ReturnOnEquityTTM", "DividendYield", "Beta", "EPS",
            "BookValue", "DividendPerShare", "RevenuePerShareTTM",
            "52WeekHigh", "52WeekLow", "50DayMovingAverage",
            "200DayMovingAverage", "AnalystTargetPrice",
        ]
        int_fields = [
            "MarketCapitalization", "EBITDA", "RevenueTTM",
            "GrossProfitTTM", "SharesOutstanding",
        ]

        result: dict[str, Any] = {}
        for key, val in raw.items():
            if key in float_fields:
                result[key] = self._safe_float(val)
            elif key in int_fields:
                result[key] = self._safe_int(val)
            else:
                result[key] = val

        return result

    @staticmethod
    def _safe_float(val: Any) -> float:
        """安全轉換為 float，處理 'None', '-', '' 等異常值"""
        if val is None:
            return 0.0
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _safe_int(val: Any) -> int:
        """安全轉換為 int"""
        if val is None:
            return 0
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return 0
