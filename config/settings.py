"""
全域系統配置 (Global System Configuration)
==========================================
所有環境變數、合規參數、連線設定都集中在此。
上層模組應透過 ``from config.settings import config`` 取用。
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SystemConfig:
    """系統核心配置與 FCA 合規參數設定"""

    # --- API 與環境設定 ---
    ENV: Literal["demo", "live"] = os.getenv("T212_ENV", "demo")
    API_KEY: str = os.getenv("T212_API_KEY", "your_api_key_here")
    API_SECRET: str = os.getenv("T212_API_SECRET", "")

    @property
    def BASE_URL(self) -> str:
        if self.ENV == "demo":
            return "https://demo.trading212.com/api/v0"
        return "https://live.trading212.com/api/v0"

    @property
    def AUTH_HEADER(self) -> str:
        """Build Basic Auth header: Base64(API_KEY:API_SECRET)."""
        creds = f"{self.API_KEY}:{self.API_SECRET}"
        encoded = base64.b64encode(creds.encode()).decode()
        return f"Basic {encoded}"

    # --- 系統連線與退避設定 ---
    MAX_RETRIES: int = 5
    INITIAL_BACKOFF: float = 1.0   # 初始退避秒數
    MAX_BACKOFF: float = 60.0      # 最大退避秒數

    # --- FCA 合規與風險守門員參數 ---
    MAX_ORDER_VALUE_PCT: float = 0.05   # 肥手指防護：單筆訂單不得超過帳戶淨值的 5%
    MAX_DAILY_VOLUME_PCT: float = 0.30  # 累計日成交量防護：不超過淨值的 30%
    MAX_PENDING_ORDERS: int = 50        # T212 API 限制：單一標的最多 50 筆未執行掛單

    # --- 外部資料 API 金鑰 (Data Scout 使用) ---
    ALPHA_VANTAGE_KEY: str = os.getenv("ALPHA_VANTAGE_KEY", "")
    POLYGON_KEY: str = os.getenv("POLYGON_KEY", "")
    FINNHUB_KEY: str = os.getenv("FINNHUB_KEY", "")


# ── 全域單例 ─────────────────────────────────────────────────────
config = SystemConfig()


# ── 向後相容別名 (Backward-compatible aliases) ───────────────────
# 讓已存在的上層模組 (compliance, execution, orchestrator…)
# 仍然可以用 ``from config import T212Config, DataSourceConfig, Environment``
from enum import Enum


class Environment(Enum):
    DEMO = "demo"
    LIVE = "live"


@dataclass(frozen=True)
class T212Config:
    """向後相容包裝：舊模組仍可透過此 dataclass 取得設定。"""
    environment: Environment = Environment.DEMO
    api_key: str = ""
    api_secret: str = ""

    rate_limit_buffer: int = 2
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 60.0
    backoff_multiplier: float = 2.0

    max_order_pct_of_nav: float = 0.05
    kill_switch_orders_per_second: int = 10
    max_pending_orders_per_instrument: int = 50
    default_page_limit: int = 50

    @property
    def base_url(self) -> str:
        if self.environment == Environment.LIVE:
            return "https://live.trading212.com/api/v0"
        return "https://demo.trading212.com/api/v0"

    @property
    def auth_header(self) -> str:
        """Build Basic Auth header: Base64(API_KEY:API_SECRET)."""
        creds = f"{self.api_key}:{self.api_secret}"
        encoded = base64.b64encode(creds.encode()).decode()
        return f"Basic {encoded}"

    @classmethod
    def from_env(cls) -> "T212Config":
        env_str = os.getenv("T212_ENV", "demo").lower()
        environment = Environment.LIVE if env_str == "live" else Environment.DEMO
        return cls(
            environment=environment,
            api_key=os.getenv("T212_API_KEY", ""),
            api_secret=os.getenv("T212_API_SECRET", ""),
        )


@dataclass(frozen=True)
class DataSourceConfig:
    """向後相容包裝：外部資料源 API 金鑰。"""
    alpha_vantage_key: str = field(default_factory=lambda: os.getenv("ALPHA_VANTAGE_KEY", ""))
    polygon_key: str = field(default_factory=lambda: os.getenv("POLYGON_KEY", ""))
    finnhub_key: str = field(default_factory=lambda: os.getenv("FINNHUB_KEY", ""))
    intrinio_key: str = field(default_factory=lambda: os.getenv("INTRINIO_KEY", ""))
