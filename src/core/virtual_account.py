"""
虛擬子帳戶管理器 (Virtual Sub-Account Manager)
===============================================
在軟體層面建立帳本 (Ledger)，將實體 Trading 212 帳戶的資金與部位
進行邏輯隔離。每個機械人策略擁有獨立的虛擬現金池與持倉記錄。

核心解決問題：
  1. 防踩踏：機械人 A 的部位不會被機械人 B 誤賣
  2. 曝險隔離：策略 A 虧光不影響策略 B 的資金安全
  3. 狀態持久化：JSON 檔案確保重啟後不遺失帳本

Trading 212 API 本身不知道多機械人的存在，所有隔離邏輯由此模組實現。
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════

@dataclass
class VirtualPosition:
    """單一標的的虛擬持倉紀錄"""
    ticker: str
    quantity: float
    average_price: float

    @property
    def market_value(self) -> float:
        """以平均成本計算的持倉市值（實際市值需外部價格更新）"""
        return self.quantity * self.average_price

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VirtualSubAccount:
    """
    虛擬子帳戶實體
    每個機械人策略持有一個，記錄獨立的資金池和持倉。
    """
    bot_id: str
    allocated_capital: float          # 最初分配的總資金
    available_cash: float             # 目前可用的虛擬現金
    positions: Dict[str, VirtualPosition] = field(default_factory=dict)
    trade_count: int = 0              # 累計交易筆數
    realised_pnl: float = 0.0        # 累計已實現損益
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    # ── 查詢方法 ──────────────────────────────────────────────────

    def can_afford(self, estimated_cost: float) -> bool:
        """檢查虛擬現金是否足夠支付這筆交易"""
        return self.available_cash >= estimated_cost

    def get_position_qty(self, ticker: str) -> float:
        """獲取該機械人持有的特定標的數量"""
        if ticker in self.positions:
            return self.positions[ticker].quantity
        return 0.0

    def get_position(self, ticker: str) -> Optional[VirtualPosition]:
        """取得完整持倉物件"""
        return self.positions.get(ticker)

    @property
    def total_invested_value(self) -> float:
        """所有持倉的成本基礎加總"""
        return sum(p.market_value for p in self.positions.values())

    @property
    def total_nav(self) -> float:
        """虛擬淨值 = 可用現金 + 持倉市值（以成本計算）"""
        return self.available_cash + self.total_invested_value

    @property
    def exposure_pct(self) -> float:
        """持倉曝險佔總淨值比例"""
        nav = self.total_nav
        if nav <= 0:
            return 0.0
        return self.total_invested_value / nav

    @property
    def position_count(self) -> int:
        return len(self.positions)

    # ── 交易記帳 ──────────────────────────────────────────────────

    def record_trade(
        self,
        ticker: str,
        quantity: float,
        execution_price: float,
    ) -> dict[str, Any]:
        """
        記錄交易並更新虛擬帳本。
        quantity > 0 為買入，quantity < 0 為賣出。

        Returns
        -------
        dict : 交易摘要 (用於審計日誌)
        """
        trade_value = abs(quantity) * execution_price
        trade_summary: dict[str, Any] = {
            "bot_id": self.bot_id,
            "ticker": ticker,
            "quantity": quantity,
            "execution_price": execution_price,
            "trade_value": trade_value,
            "timestamp": time.time(),
        }

        if quantity > 0:
            # ── 買入：扣除虛擬現金，增加部位 ────────────────────
            if trade_value > self.available_cash:
                logger.warning(
                    "[%s] 警告：虛擬現金不足 (需要 %.2f, 可用 %.2f)，發生超買！",
                    self.bot_id, trade_value, self.available_cash,
                )
            self.available_cash -= trade_value

            if ticker in self.positions:
                # 加碼：計算新的加權平均成本
                pos = self.positions[ticker]
                total_cost = (pos.quantity * pos.average_price) + trade_value
                pos.quantity += quantity
                pos.average_price = total_cost / pos.quantity if pos.quantity > 0 else 0.0
            else:
                self.positions[ticker] = VirtualPosition(
                    ticker=ticker,
                    quantity=quantity,
                    average_price=execution_price,
                )

            trade_summary["side"] = "BUY"
            trade_summary["new_position_qty"] = self.get_position_qty(ticker)

        elif quantity < 0:
            # ── 賣出：增加虛擬現金，減少部位 ────────────────────
            self.available_cash += trade_value

            if ticker in self.positions:
                pos = self.positions[ticker]
                # 計算這筆賣出的已實現損益
                cost_basis = abs(quantity) * pos.average_price
                pnl = trade_value - cost_basis
                self.realised_pnl += pnl
                trade_summary["realised_pnl"] = pnl

                pos.quantity += quantity  # quantity 是負數，所以是減少

                # 如果部位清空，將其從追蹤清單移除
                if pos.quantity <= 0.0001:  # 處理浮點數誤差
                    del self.positions[ticker]
                    trade_summary["position_closed"] = True
                else:
                    trade_summary["remaining_qty"] = pos.quantity
            else:
                logger.error(
                    "[%s] 嘗試賣出不存在的虛擬部位：%s", self.bot_id, ticker,
                )
                trade_summary["error"] = "position_not_found"

            trade_summary["side"] = "SELL"

        self.trade_count += 1
        self.last_updated = time.time()

        logger.info(
            "[%s] 交易記帳完成: %s %s %.4f @ %.2f | 可用現金: %.2f",
            self.bot_id, trade_summary.get("side", "?"),
            ticker, abs(quantity), execution_price, self.available_cash,
        )

        return trade_summary

    # ── 市值更新（使用最新市價）──────────────────────────────────

    def update_market_prices(self, price_map: dict[str, float]) -> None:
        """
        用最新市場價格更新持倉的估值。
        注意：這不會改變 average_price (成本基礎)，
        而是讓 total_nav 計算更準確。

        此方法預留給未來擴展 —— 若需要即時 NAV，
        可增加一個 current_price 欄位到 VirtualPosition。
        """
        # 目前 VirtualPosition 只有 average_price，
        # 未來可擴展為分開追蹤 cost_basis 和 current_price
        pass

    def summary(self) -> str:
        """人類可讀的帳戶摘要"""
        pos_strs = [
            f"  {p.ticker}: {p.quantity:.4f} @ avg {p.average_price:.2f}"
            for p in self.positions.values()
        ]
        positions_block = "\n".join(pos_strs) if pos_strs else "  (無持倉)"
        return (
            f"=== 虛擬子帳戶 [{self.bot_id}] ===\n"
            f"初始資金: {self.allocated_capital:.2f}\n"
            f"可用現金: {self.available_cash:.2f}\n"
            f"持倉市值: {self.total_invested_value:.2f}\n"
            f"虛擬淨值: {self.total_nav:.2f}\n"
            f"曝險比例: {self.exposure_pct:.1%}\n"
            f"累計損益: {self.realised_pnl:.2f}\n"
            f"交易筆數: {self.trade_count}\n"
            f"持倉:\n{positions_block}"
        )


# ═══════════════════════════════════════════════════════════════════
# Virtual Account Manager
# ═══════════════════════════════════════════════════════════════════

class VirtualAccountManager:
    """
    虛擬帳戶管理器
    負責加載、儲存與分配多個機械人的資金池。
    使用 JSON 檔案進行狀態持久化。
    """

    def __init__(self, storage_file: str = "data/virtual_accounts.json") -> None:
        self.storage_file = storage_file
        self.accounts: Dict[str, VirtualSubAccount] = {}
        self._ensure_storage_dir()
        self.load_state()

    # ── 帳戶管理 ──────────────────────────────────────────────────

    def allocate_account(
        self,
        bot_id: str,
        initial_capital: float,
    ) -> VirtualSubAccount:
        """
        為機械人分配新的虛擬帳戶，或回傳已存在的帳戶。
        若帳戶已存在，不會重置其狀態（保護斷電重啟場景）。
        """
        if bot_id in self.accounts:
            logger.info(
                "虛擬子帳戶已存在: %s (可用現金: %.2f)",
                bot_id, self.accounts[bot_id].available_cash,
            )
            return self.accounts[bot_id]

        logger.info(
            "建立新的虛擬子帳戶: %s, 初始資金: %.2f", bot_id, initial_capital,
        )
        account = VirtualSubAccount(
            bot_id=bot_id,
            allocated_capital=initial_capital,
            available_cash=initial_capital,
        )
        self.accounts[bot_id] = account
        self.save_state()
        return account

    def get_account(self, bot_id: str) -> Optional[VirtualSubAccount]:
        """取得特定的虛擬子帳戶"""
        return self.accounts.get(bot_id)

    def remove_account(self, bot_id: str) -> bool:
        """移除虛擬子帳戶（僅限無持倉時）"""
        acc = self.accounts.get(bot_id)
        if acc is None:
            return False
        if acc.positions:
            logger.error(
                "無法移除帳戶 %s：仍有 %d 個持倉", bot_id, len(acc.positions),
            )
            return False
        del self.accounts[bot_id]
        self.save_state()
        logger.info("已移除虛擬子帳戶: %s", bot_id)
        return True

    @property
    def total_allocated(self) -> float:
        """所有子帳戶的已分配資金加總"""
        return sum(a.allocated_capital for a in self.accounts.values())

    @property
    def total_available_cash(self) -> float:
        """所有子帳戶的可用現金加總"""
        return sum(a.available_cash for a in self.accounts.values())

    def validate_against_real_account(self, real_nav: float) -> dict[str, Any]:
        """
        與實體帳戶淨值做交叉驗證。
        若虛擬帳戶加總超過實體帳戶，發出警告。
        """
        total_virtual_nav = sum(a.total_nav for a in self.accounts.values())
        gap = total_virtual_nav - real_nav
        is_over_allocated = gap > 0

        result = {
            "real_nav": real_nav,
            "total_virtual_nav": total_virtual_nav,
            "gap": gap,
            "over_allocated": is_over_allocated,
            "accounts": {
                bid: {"nav": a.total_nav, "cash": a.available_cash}
                for bid, a in self.accounts.items()
            },
        }

        if is_over_allocated:
            logger.warning(
                "虛擬帳戶加總 (%.2f) 超過實體帳戶淨值 (%.2f)！差距: %.2f",
                total_virtual_nav, real_nav, gap,
            )

        return result

    # ── 持久化 ────────────────────────────────────────────────────

    def save_state(self) -> None:
        """將虛擬帳本狀態儲存到 JSON 檔案"""
        try:
            data: dict[str, Any] = {}
            for bot_id, acc in self.accounts.items():
                acc_dict = {
                    "bot_id": acc.bot_id,
                    "allocated_capital": acc.allocated_capital,
                    "available_cash": acc.available_cash,
                    "trade_count": acc.trade_count,
                    "realised_pnl": acc.realised_pnl,
                    "created_at": acc.created_at,
                    "last_updated": acc.last_updated,
                    "positions": {
                        ticker: asdict(pos)
                        for ticker, pos in acc.positions.items()
                    },
                }
                data[bot_id] = acc_dict

            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            logger.debug("虛擬帳本已儲存至 %s", self.storage_file)
        except Exception as e:
            logger.error("儲存虛擬帳本失敗: %s", e)

    def load_state(self) -> None:
        """從 JSON 檔案載入虛擬帳本狀態"""
        if not os.path.exists(self.storage_file):
            logger.info("未找到虛擬帳本檔案 (%s)，從空白狀態啟動", self.storage_file)
            return

        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for bot_id, acc_data in data.items():
                positions = {
                    ticker: VirtualPosition(**pos_data)
                    for ticker, pos_data in acc_data.get("positions", {}).items()
                }
                self.accounts[bot_id] = VirtualSubAccount(
                    bot_id=acc_data["bot_id"],
                    allocated_capital=acc_data["allocated_capital"],
                    available_cash=acc_data["available_cash"],
                    positions=positions,
                    trade_count=acc_data.get("trade_count", 0),
                    realised_pnl=acc_data.get("realised_pnl", 0.0),
                    created_at=acc_data.get("created_at", time.time()),
                    last_updated=acc_data.get("last_updated", time.time()),
                )

            logger.info(
                "成功載入 %d 個虛擬子帳戶狀態。", len(self.accounts),
            )
        except Exception as e:
            logger.error("載入虛擬帳本失敗: %s", e)

    def _ensure_storage_dir(self) -> None:
        """確保儲存目錄存在"""
        parent = Path(self.storage_file).parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
            logger.info("已建立虛擬帳本儲存目錄: %s", parent)

    # ── 報表 ──────────────────────────────────────────────────────

    def summary(self) -> str:
        """所有子帳戶的總覽報表"""
        lines = ["=" * 50, "  虛擬帳戶管理器 — 總覽報表", "=" * 50]
        for acc in self.accounts.values():
            lines.append(
                f"  [{acc.bot_id}] 淨值={acc.total_nav:.2f} "
                f"現金={acc.available_cash:.2f} "
                f"持倉={acc.position_count}檔 "
                f"損益={acc.realised_pnl:+.2f}"
            )
        lines.append(f"  ────────────────────")
        lines.append(f"  總已分配: {self.total_allocated:.2f}")
        lines.append(f"  總可用現金: {self.total_available_cash:.2f}")
        return "\n".join(lines)
