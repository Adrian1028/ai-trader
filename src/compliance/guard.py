"""
合規守門員 (Compliance Guard) — 決定論式防火牆
===============================================
擁有**絕對否決權 (Absolute Veto Power)**，防止 AI 代理產生幻覺
導致帳戶崩潰或違反 FCA/MAR 規範。

每一項檢查都是純決定論邏輯 — 零 LLM 參與。

檢查項目：
  1. 肥手指防護 (Fat-Finger Protection) — 多階層
  2. 累計日成交量防護 (Overtrading Prevention)
  3. 速率與死循環監控 (Rate Limiting & Loop Detection)
  4. Kill Switch with API Key Revocation
  5. MAR 反幌騙/反洗盤 (Anti-Spoofing / Anti-Layering)
  6. 掛單數量上限 (Pending Order Cap)
  7. 價格異常偵測 (Price Anomaly)
  8. 交易成本閘門 (Transaction Cost Gate)
  9. 與虛擬子帳戶整合的資金驗證

設計原則：
  - 每個機械人持有獨立的 ComplianceGuard 實例（per-bot 隔離）
  - 閾值從全域 config 自動載入
  - 所有合規事件寫入 JSONL 審計日誌
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, List, TYPE_CHECKING

from config.settings import config

if TYPE_CHECKING:
    from src.core.virtual_account import VirtualSubAccount

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════

class VetoReason(Enum):
    FAT_FINGER = auto()
    KILL_SWITCH_RATE = auto()
    KILL_SWITCH_LOOP = auto()
    PENDING_LIMIT_EXCEEDED = auto()
    SPOOFING_SUSPICION = auto()
    LAYERING_SUSPICION = auto()
    EXCESSIVE_AMENDMENTS = auto()
    CUMULATIVE_EXPOSURE = auto()
    PRICE_ANOMALY = auto()
    COST_EXCEEDS_EDGE = auto()
    INSUFFICIENT_FUNDS = auto()
    INSUFFICIENT_POSITION = auto()


@dataclass
class VetoResult:
    """
    審查結果實體。
    同時支援新舊兩種存取方式：
      - 新版：veto.is_approved / veto.reason (str)
      - 舊版：veto.allowed / veto.reason (VetoReason)
    """
    is_approved: bool
    reason: str | VetoReason | None = None
    detail: str = ""
    severity: str = "info"   # "info", "warning", "critical"

    @property
    def allowed(self) -> bool:
        """向後相容別名"""
        return self.is_approved


@dataclass
class ComplianceEvent:
    """Immutable log entry for every compliance decision."""
    timestamp: float = field(default_factory=time.time)
    event_type: str = ""          # "veto", "allow", "kill_switch", "alert"
    reason: str = ""
    detail: str = ""
    order_value: float = 0.0
    ticker: str = ""
    bot_id: str = ""


# ═══════════════════════════════════════════════════════════════════
# ComplianceGuard — 主體
# ═══════════════════════════════════════════════════════════════════

class ComplianceGuard:
    """
    合規守門員 (決定論式防火牆)
    擁有絕對否決權，防止 AI 代理產生幻覺導致帳戶崩潰或違反 FCA 規範 (MAR)。

    每個機械人策略持有獨立實例，確保合規計數器互不干擾。

    支援兩種初始化方式：
      1. 新版 (per-bot)：ComplianceGuard(bot_id="GrowthBot")
      2. 舊版 (全域)：ComplianceGuard(max_order_pct_of_nav=0.05, ...)
    """

    def __init__(
        self,
        bot_id: str = "default",
        *,
        # Fat-finger tiers
        max_order_pct_of_nav: float | None = None,
        warn_order_pct_of_nav: float = 0.03,
        max_daily_turnover_pct: float | None = None,

        # Kill switch
        kill_switch_max_ops: int = 10,
        loop_detection_window: int = 30,
        loop_detection_repeat_threshold: int = 5,
        max_orders_per_minute: int = 20,

        # Pending orders
        max_pending_per_instrument: int = 50,

        # MAR anti-abuse
        cancel_rate_threshold: float = 0.85,
        amendment_rate_threshold: float = 0.70,
        layering_max_one_side: int = 5,

        # Price sanity
        max_price_deviation_pct: float = 0.10,

        # Transaction cost gate
        min_expected_roi_after_costs: float = 0.005,

        # UK-specific costs
        stamp_duty_rate: float = 0.005,
        ptm_levy_threshold: float = 10_000.0,
        ptm_levy_amount: float = 1.0,

        # Logging
        log_dir: str = "logs/compliance",
    ) -> None:
        self.bot_id = bot_id
        self.kill_switch_triggered = False
        self._api_key_revoked = False

        # 參數閾值（優先使用傳入值，否則從全域 config 載入）
        self._max_order_pct = max_order_pct_of_nav or config.MAX_ORDER_VALUE_PCT
        self._warn_order_pct = warn_order_pct_of_nav
        self._max_daily_turnover_pct = max_daily_turnover_pct or config.MAX_DAILY_VOLUME_PCT
        self._max_pending = max_pending_per_instrument or config.MAX_PENDING_ORDERS

        # Kill switch
        self._kill_switch_max = kill_switch_max_ops
        self._loop_window = loop_detection_window
        self._loop_repeat_threshold = loop_detection_repeat_threshold
        self._max_orders_per_minute = max_orders_per_minute

        # MAR
        self._cancel_rate_threshold = cancel_rate_threshold
        self._amendment_rate_threshold = amendment_rate_threshold
        self._layering_max = layering_max_one_side

        # Price
        self._max_price_dev = max_price_deviation_pct

        # Transaction cost
        self._min_roi_after_costs = min_expected_roi_after_costs
        self._stamp_duty_rate = stamp_duty_rate
        self._ptm_threshold = ptm_levy_threshold
        self._ptm_amount = ptm_levy_amount

        # ── 狀態追蹤計數器 ───────────────────────────────────────────
        self.daily_traded_volume = 0.0
        self.total_orders_placed = 0
        self.total_orders_cancelled = 0
        self._orders_amended = 0
        self.recent_order_timestamps: List[float] = []
        self._order_timestamps: deque[float] = deque()       # for per-second rate
        self._order_tickers: deque[tuple[float, str]] = deque()
        self._daily_reset_date: str = ""

        # Pending order tracker: ticker → count per side
        self._pending_buys: dict[str, int] = {}
        self._pending_sells: dict[str, int] = {}

        # Event log
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._events: list[ComplianceEvent] = []

    # ── Backward-compatible property aliases ──────────────────────
    @property
    def is_killed(self) -> bool:
        return self.kill_switch_triggered

    @property
    def is_api_key_revoked(self) -> bool:
        return self._api_key_revoked

    @property
    def _killed(self) -> bool:
        return self.kill_switch_triggered

    @_killed.setter
    def _killed(self, value: bool) -> None:
        self.kill_switch_triggered = value

    @property
    def _orders_placed(self) -> int:
        return self.total_orders_placed

    @_orders_placed.setter
    def _orders_placed(self, value: int) -> None:
        self.total_orders_placed = value

    @property
    def _orders_cancelled(self) -> int:
        return self.total_orders_cancelled

    @_orders_cancelled.setter
    def _orders_cancelled(self, value: int) -> None:
        self.total_orders_cancelled = value

    @property
    def _daily_turnover(self) -> float:
        return self.daily_traded_volume

    @_daily_turnover.setter
    def _daily_turnover(self, value: float) -> None:
        self.daily_traded_volume = value

    # ══════════════════════════════════════════════════════════════
    # 新版主入口：pre_trade_check (虛擬子帳戶整合)
    # ══════════════════════════════════════════════════════════════

    def pre_trade_check(
        self,
        ticker: str,
        quantity: float,
        estimated_price: float,
        account: "VirtualSubAccount",
        *,
        side: str = "buy",
        reference_price: float = 0.0,
        expected_roi: float = 0.0,
        is_uk_equity: bool = True,
        pending_count_for_instrument: int = 0,
    ) -> VetoResult:
        """
        下單前綜合審查 (Validate-before-actuate)。
        在 Execution Agent 真正呼叫 T212 API 前必須通過此閘門。

        Parameters
        ----------
        ticker : 股票代號
        quantity : 交易數量（正數買入、負數賣出）
        estimated_price : 預估成交價
        account : 該機械人的虛擬子帳戶
        side : "buy" or "sell"
        reference_price : 參考價格（用於價格異常偵測）
        expected_roi : AI 預估的報酬率
        is_uk_equity : 是否為英國股票（影響印花稅）
        pending_count_for_instrument : 該標的目前的掛單數
        """
        self._maybe_reset_daily()

        # ── 0. Kill Switch 已觸發？ ─────────────────────────────────
        if self.kill_switch_triggered:
            return self._veto(
                VetoReason.KILL_SWITCH_RATE,
                "拒絕：Kill Switch 已觸發，系統處於鎖定狀態。",
                severity="critical", ticker=ticker,
            )

        order_value = abs(quantity) * estimated_price

        # 採用虛擬子帳戶的初始分配資金作為淨值基準
        account_nav = account.allocated_capital

        # ── 1. 肥手指防護 (Fat-Finger Protection) ───────────────────
        if account_nav > 0 and order_value / account_nav > self._max_order_pct:
            pct = order_value / account_nav * 100
            detail = (
                f"攔截肥手指！{ticker} 委託金額 £{order_value:.2f} "
                f"超過淨值上限 ({self._max_order_pct * 100:.0f}% of £{account_nav:.2f})"
            )
            logger.critical("[%s] %s", self.bot_id, detail)
            return self._veto(
                VetoReason.FAT_FINGER, detail,
                severity="critical", ticker=ticker, order_value=order_value,
            )

        # 肥手指警告階層
        if account_nav > 0 and order_value / account_nav > self._warn_order_pct:
            pct = order_value / account_nav * 100
            detail = (
                f"委託金額 £{order_value:.2f} 佔淨值 {pct:.1f}%，"
                f"超過警告閾值 ({self._warn_order_pct * 100:.0f}%)"
            )
            return self._veto(
                VetoReason.FAT_FINGER, detail,
                severity="warning", ticker=ticker, order_value=order_value,
            )

        # ── 2. 虛擬帳戶資金/持倉驗證 ───────────────────────────────
        if side == "buy":
            if not account.can_afford(order_value):
                return self._veto(
                    VetoReason.INSUFFICIENT_FUNDS,
                    f"虛擬子帳戶 [{self.bot_id}] 可用資金不足 "
                    f"(需要 £{order_value:.2f}, 可用 £{account.available_cash:.2f})",
                    ticker=ticker, order_value=order_value,
                )
        elif side == "sell":
            owned_qty = account.get_position_qty(ticker)
            if owned_qty < abs(quantity):
                return self._veto(
                    VetoReason.INSUFFICIENT_POSITION,
                    f"虛擬子帳戶 [{self.bot_id}] 持倉不足 "
                    f"(需要 {abs(quantity):.4f}, 持有 {owned_qty:.4f})",
                    ticker=ticker, order_value=order_value,
                )

        # ── 3. 累計日成交量防護 (防止 AI 過度交易) ──────────────────
        if account_nav > 0:
            projected_daily = self.daily_traded_volume + order_value
            if projected_daily / account_nav > self._max_daily_turnover_pct:
                detail = (
                    f"達到單日交易量上限！累計 £{projected_daily:.0f} "
                    f"({projected_daily / account_nav:.0%} of NAV), "
                    f"超過 {self._max_daily_turnover_pct:.0%} 限制"
                )
                logger.warning("[%s] %s", self.bot_id, detail)
                return self._veto(
                    VetoReason.CUMULATIVE_EXPOSURE, detail,
                    severity="warning", ticker=ticker, order_value=order_value,
                )

        # ── 4. 掛單數量上限 ─────────────────────────────────────────
        if pending_count_for_instrument >= self._max_pending:
            return self._veto(
                VetoReason.PENDING_LIMIT_EXCEEDED,
                f"標的 {ticker} 已有 {pending_count_for_instrument} 筆掛單 "
                f"(上限 {self._max_pending})",
                ticker=ticker, order_value=order_value,
            )

        # ── 5. 分層偵測 (Layering Detection — MAR Article 12) ──────
        side_count = self._count_pending_side(ticker, side)
        if side_count >= self._layering_max:
            return self._veto(
                VetoReason.LAYERING_SUSPICION,
                f"已有 {side_count} 筆同方向 ({side}) 掛單 — "
                f"MAR Article 12: 疑似分層操縱",
                severity="warning", ticker=ticker, order_value=order_value,
            )

        # ── 6. 速率監控 — 每秒 (Kill Switch) ───────────────────────
        now = time.monotonic()
        self._order_timestamps.append(now)
        while self._order_timestamps and (now - self._order_timestamps[0]) > 1.0:
            self._order_timestamps.popleft()

        if len(self._order_timestamps) > self._kill_switch_max:
            reason = (
                f"偵測到異常高頻下單 ({len(self._order_timestamps)}/s > "
                f"{self._kill_switch_max}/s)"
            )
            self.trigger_kill_switch(reason)
            return self._veto(
                VetoReason.KILL_SWITCH_RATE, reason,
                severity="critical", ticker=ticker, order_value=order_value,
            )

        # ── 7. 速率監控 — 每分鐘 (死循環偵測) ──────────────────────
        now_time = time.time()
        self.recent_order_timestamps = [
            ts for ts in self.recent_order_timestamps if now_time - ts < 60.0
        ]
        if len(self.recent_order_timestamps) >= self._max_orders_per_minute:
            reason = (
                f"偵測到異常高頻下單 ({len(self.recent_order_timestamps)}/min > "
                f"{self._max_orders_per_minute}/min)，疑似邏輯死循環"
            )
            self.trigger_kill_switch(reason)
            return self._veto(
                VetoReason.KILL_SWITCH_RATE, reason,
                severity="critical", ticker=ticker, order_value=order_value,
            )

        # ── 8. 死循環偵測 — 同標的重複 ─────────────────────────────
        self._order_tickers.append((now, ticker))
        while self._order_tickers and (now - self._order_tickers[0][0]) > self._loop_window:
            self._order_tickers.popleft()

        if ticker:
            same_ticker_count = sum(1 for _, t in self._order_tickers if t == ticker)
            if same_ticker_count >= self._loop_repeat_threshold:
                reason = (
                    f"死循環偵測：{same_ticker_count} 筆 {ticker} 訂單 "
                    f"在 {self._loop_window}s 內"
                )
                self.trigger_kill_switch(reason)
                return self._veto(
                    VetoReason.KILL_SWITCH_LOOP, reason,
                    severity="critical", ticker=ticker, order_value=order_value,
                )

        # ── 9. 反幌騙：撤單率 (MAR Article 12(2)(c)) ───────────────
        self.total_orders_placed += 1
        if self.total_orders_placed > 20:
            cancel_rate = self.total_orders_cancelled / self.total_orders_placed
            if cancel_rate > self._cancel_rate_threshold:
                return self._veto(
                    VetoReason.SPOOFING_SUSPICION,
                    f"撤單率 {cancel_rate:.0%} 超過 {self._cancel_rate_threshold:.0%} — "
                    f"MAR Article 12(2)(c): 疑似幌騙 "
                    f"({self.total_orders_cancelled}/{self.total_orders_placed})",
                    severity="warning", ticker=ticker, order_value=order_value,
                )

        # ── 10. 反幌騙：改單率 ─────────────────────────────────────
        if self.total_orders_placed > 20:
            amend_rate = self._orders_amended / self.total_orders_placed
            if amend_rate > self._amendment_rate_threshold:
                return self._veto(
                    VetoReason.EXCESSIVE_AMENDMENTS,
                    f"改單率 {amend_rate:.0%} 超過 {self._amendment_rate_threshold:.0%} — "
                    f"過度修改訂單疑似操縱",
                    severity="warning", ticker=ticker, order_value=order_value,
                )

        # ── 11. 價格異常偵測 ───────────────────────────────────────
        order_price = estimated_price
        if reference_price > 0 and order_price > 0:
            deviation = abs(order_price - reference_price) / reference_price
            if deviation > self._max_price_dev:
                return self._veto(
                    VetoReason.PRICE_ANOMALY,
                    f"委託價 {order_price:.4f} 偏離參考價 {reference_price:.4f} "
                    f"達 {deviation:.1%} (上限 {self._max_price_dev:.0%})",
                    ticker=ticker, order_value=order_value,
                )

        # ── 12. 交易成本閘門 ───────────────────────────────────────
        if expected_roi > 0 and is_uk_equity:
            friction = self._estimate_friction(
                order_value, is_buy=(side == "buy"), is_uk=is_uk_equity,
            )
            friction_as_roi = friction / order_value if order_value > 0 else 0
            net_expected_roi = expected_roi - friction_as_roi
            if net_expected_roi < self._min_roi_after_costs:
                return self._veto(
                    VetoReason.COST_EXCEEDS_EDGE,
                    f"預期 ROI {expected_roi:.2%} - 摩擦 {friction_as_roi:.2%} "
                    f"= {net_expected_roi:.2%} < 門檻 {self._min_roi_after_costs:.2%}",
                    ticker=ticker, order_value=order_value,
                )

        # ── 全部通過 ────────────────────────────────────────────────
        self.daily_traded_volume += order_value
        self.recent_order_timestamps.append(now_time)
        self._record_pending(ticker, side)
        self._log_event(
            "allow", "", f"合規審查通過: {ticker} {side} £{order_value:.2f}",
            ticker=ticker, order_value=order_value,
        )

        return VetoResult(is_approved=True, reason=None, detail="合規審查通過")

    # ══════════════════════════════════════════════════════════════
    # 舊版入口 (Backward Compatible)
    # ══════════════════════════════════════════════════════════════

    def validate_order(
        self,
        order_value: float,
        nav: float,
        pending_count_for_instrument: int,
        *,
        ticker: str = "",
        side: str = "buy",
        reference_price: float = 0.0,
        order_price: float = 0.0,
        expected_roi: float = 0.0,
        is_uk_equity: bool = True,
    ) -> VetoResult:
        """
        舊版合規閘門入口 (Backward Compatible)。
        內部建構臨時 VirtualSubAccount 進行相容呼叫。
        """
        from src.core.virtual_account import VirtualSubAccount

        # Build a temporary virtual account with the given NAV
        temp_account = VirtualSubAccount(
            bot_id=self.bot_id,
            allocated_capital=nav,
            available_cash=nav,  # Assume all cash for backward compat
        )

        price = order_price if order_price > 0 else (
            reference_price if reference_price > 0 else 1.0
        )
        quantity = abs(order_value) / price if price > 0 else 0.0

        return self.pre_trade_check(
            ticker=ticker,
            quantity=quantity,
            estimated_price=price,
            account=temp_account,
            side=side,
            reference_price=reference_price,
            expected_roi=expected_roi,
            is_uk_equity=is_uk_equity,
            pending_count_for_instrument=pending_count_for_instrument,
        )

    # ══════════════════════════════════════════════════════════════
    # 狀態追蹤方法 (State Mutation)
    # ══════════════════════════════════════════════════════════════

    def record_order_placed(self, order_value: float) -> None:
        """記錄已發送的訂單（在 API 呼叫成功後觸發）"""
        # Note: total_orders_placed is already incremented in pre_trade_check
        self.recent_order_timestamps.append(time.time())
        self.daily_traded_volume += order_value

    def record_order_cancelled(self) -> None:
        """記錄被取消的訂單（用於反市場濫用 MAR 計算）"""
        self.total_orders_cancelled += 1
        self.check_spoofing()

    def record_cancellation(self, ticker: str = "") -> None:
        """向後相容別名 for record_order_cancelled"""
        self.total_orders_cancelled += 1
        self._release_pending(ticker, "buy")
        self._release_pending(ticker, "sell")
        self._log_event("cancel", "order_cancelled",
                        f"訂單已取消: {ticker}", ticker=ticker)
        self.check_spoofing()

    def record_amendment(self, ticker: str = "") -> None:
        """記錄改單事件"""
        self._orders_amended += 1
        self._log_event("amend", "order_amended",
                        f"訂單已修改: {ticker}", ticker=ticker)

    def record_fill(self, ticker: str = "", side: str = "buy") -> None:
        """記錄成交，釋放掛單槽位"""
        self._release_pending(ticker, side)

    def check_spoofing(self) -> None:
        """反市場濫用 (MAR): 幌騙 (Spoofing) 偵測"""
        if self.total_orders_placed < 10:
            return

        cancel_rate = self.total_orders_cancelled / self.total_orders_placed
        if cancel_rate > self._cancel_rate_threshold:
            self.trigger_kill_switch(
                f"偵測到極高撤單率 ({cancel_rate * 100:.1f}%)，"
                f"疑似幌騙 (Spoofing) 或洗盤行為！"
            )

    # ══════════════════════════════════════════════════════════════
    # KILL SWITCH
    # ══════════════════════════════════════════════════════════════

    def trigger_kill_switch(self, reason: str) -> None:
        """觸發緊急停止開關"""
        if not self.kill_switch_triggered:
            self.kill_switch_triggered = True
            self._api_key_revoked = True
            logger.critical("=" * 60)
            logger.critical(
                "[%s] KILL SWITCH TRIGGERED — %s", self.bot_id, reason,
            )
            logger.critical(
                "系統已鎖定。請人工介入檢查，並手動撤銷 T212 未成交掛單。"
            )
            logger.critical("=" * 60)
            self._log_event("kill_switch", reason, reason, severity="critical")

    def reset_kill_switch(self) -> None:
        """手動操作員重置 (需人工調查後才可使用)"""
        logger.warning("[%s] Kill switch 由操作員手動重置", self.bot_id)
        self.kill_switch_triggered = False
        self._api_key_revoked = False
        self._order_timestamps.clear()
        self._order_tickers.clear()
        self.recent_order_timestamps.clear()
        self._log_event("kill_switch_reset", "manual_reset",
                        "Kill switch 由操作員重置")

    async def emergency_shutdown(self, client: Any) -> None:
        """
        核彈選項：
          1. 觸發 Kill Switch
          2. 撤銷 API Key flag
          3. 透過 API 取消所有掛單
        """
        self.kill_switch_triggered = True
        self._api_key_revoked = True
        logger.critical("[%s] EMERGENCY SHUTDOWN — 取消所有掛單", self.bot_id)
        self._log_event("emergency_shutdown", "emergency",
                        "緊急關機啟動", severity="critical")
        try:
            await client.cancel_all_orders()
        except Exception:
            logger.exception("緊急關機期間取消掛單失敗")

    # ══════════════════════════════════════════════════════════════
    # OPRO 獎勵懲罰整合
    # ══════════════════════════════════════════════════════════════

    def compute_mar_penalty(self) -> float:
        """
        MAR 風險懲罰分數 (0 to -100)，注入 OPRO 獎勵函數。
        高撤單率 → 重罰  |  高改單率 → 中罰  |  Kill Switch → 極罰
        """
        penalty = 0.0

        if self.total_orders_placed > 10:
            cancel_rate = self.total_orders_cancelled / self.total_orders_placed
            if cancel_rate > 0.5:
                penalty -= cancel_rate * 60

            amend_rate = self._orders_amended / self.total_orders_placed
            if amend_rate > 0.3:
                penalty -= amend_rate * 30

        if self.kill_switch_triggered:
            penalty -= 100

        return max(penalty, -100.0)

    # ══════════════════════════════════════════════════════════════
    # 交易成本估算
    # ══════════════════════════════════════════════════════════════

    def _estimate_friction(
        self, value: float, is_buy: bool, is_uk: bool = True,
    ) -> float:
        """估算單邊交易摩擦成本"""
        cost = 0.0
        if is_buy and is_uk:
            cost += value * self._stamp_duty_rate
        if value > self._ptm_threshold:
            cost += self._ptm_amount
        cost += value * 0.0008   # spread 5bps + slippage 3bps = 8bps
        return cost

    def estimate_round_trip_cost(self, value: float, is_uk: bool = True) -> float:
        """估算完整來回交易摩擦成本"""
        buy_cost = self._estimate_friction(value, is_buy=True, is_uk=is_uk)
        sell_cost = self._estimate_friction(value, is_buy=False, is_uk=is_uk)
        return buy_cost + sell_cost

    # ══════════════════════════════════════════════════════════════
    # 日計數器重置
    # ══════════════════════════════════════════════════════════════

    def reset_daily_counters(self) -> None:
        """換日時重置單日計數器"""
        self.daily_traded_volume = 0.0
        self.total_orders_placed = 0
        self.total_orders_cancelled = 0
        self._orders_amended = 0
        self.recent_order_timestamps.clear()
        logger.info("[%s] 已重置單日合規計數器", self.bot_id)

    def _maybe_reset_daily(self) -> None:
        """自動換日重置"""
        today = time.strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            self.daily_traded_volume = 0.0
            self._daily_reset_date = today

    # ══════════════════════════════════════════════════════════════
    # 內部輔助方法
    # ══════════════════════════════════════════════════════════════

    def _veto(
        self,
        reason: VetoReason,
        detail: str,
        severity: str = "warning",
        ticker: str = "",
        order_value: float = 0.0,
    ) -> VetoResult:
        """Create a VetoResult and log it."""
        logger.log(
            logging.CRITICAL if severity == "critical" else logging.WARNING,
            "COMPLIANCE VETO [%s]: %s", reason.name, detail,
        )
        self._log_event("veto", reason.name, detail,
                        severity=severity, ticker=ticker, order_value=order_value)
        return VetoResult(
            is_approved=False, reason=reason, detail=detail, severity=severity,
        )

    def _record_pending(self, ticker: str, side: str) -> None:
        if not ticker:
            return
        if side == "buy":
            self._pending_buys[ticker] = self._pending_buys.get(ticker, 0) + 1
        else:
            self._pending_sells[ticker] = self._pending_sells.get(ticker, 0) + 1

    def _release_pending(self, ticker: str, side: str) -> None:
        if not ticker:
            return
        if side == "buy" and ticker in self._pending_buys:
            self._pending_buys[ticker] = max(0, self._pending_buys[ticker] - 1)
        elif side == "sell" and ticker in self._pending_sells:
            self._pending_sells[ticker] = max(0, self._pending_sells[ticker] - 1)

    def _count_pending_side(self, ticker: str, side: str) -> int:
        if side == "buy":
            return self._pending_buys.get(ticker, 0)
        return self._pending_sells.get(ticker, 0)

    # ── 事件日誌 ──────────────────────────────────────────────────

    def _log_event(
        self,
        event_type: str,
        reason: str,
        detail: str,
        severity: str = "info",
        ticker: str = "",
        order_value: float = 0.0,
    ) -> None:
        event = ComplianceEvent(
            event_type=event_type,
            reason=reason,
            detail=detail,
            ticker=ticker,
            order_value=order_value,
            bot_id=self.bot_id,
        )
        self._events.append(event)
        if severity == "critical":
            self._persist_events()

    def _persist_events(self) -> None:
        filepath = self._log_dir / "compliance_events.jsonl"
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                for event in self._events:
                    line = json.dumps({
                        "timestamp": event.timestamp,
                        "event_type": event.event_type,
                        "reason": event.reason,
                        "detail": event.detail,
                        "ticker": event.ticker,
                        "order_value": event.order_value,
                        "bot_id": event.bot_id,
                    }, ensure_ascii=False)
                    f.write(line + "\n")
            self._events.clear()
        except Exception:
            logger.exception("持久化合規事件失敗")

    def flush_events(self) -> None:
        """將緩衝事件寫入磁碟"""
        if self._events:
            self._persist_events()

    # ── 報表 ──────────────────────────────────────────────────────

    def compliance_report(self) -> dict[str, Any]:
        """回傳合規狀態摘要"""
        cancel_rate = (
            self.total_orders_cancelled / self.total_orders_placed
            if self.total_orders_placed > 0 else 0.0
        )
        amend_rate = (
            self._orders_amended / self.total_orders_placed
            if self.total_orders_placed > 0 else 0.0
        )
        return {
            "bot_id": self.bot_id,
            "killed": self.kill_switch_triggered,
            "api_key_revoked": self._api_key_revoked,
            "orders_placed": self.total_orders_placed,
            "orders_cancelled": self.total_orders_cancelled,
            "orders_amended": self._orders_amended,
            "cancel_rate": cancel_rate,
            "amendment_rate": amend_rate,
            "daily_turnover": self.daily_traded_volume,
            "mar_penalty": self.compute_mar_penalty(),
            "pending_buys": dict(self._pending_buys),
            "pending_sells": dict(self._pending_sells),
        }
