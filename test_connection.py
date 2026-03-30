"""
連線測試腳本 (Connection Test)
==============================
驗證：
  1. .env / 環境變數是否正確載入
  2. API 金鑰是否有效
  3. 速率限制器是否正常運作
  4. Demo / Live 環境路由是否正確

Usage:
    python test_connection.py
"""
import asyncio
import logging
import sys

# 把專案根目錄加到 path，確保 import 正常
sys.path.insert(0, ".")

from config.settings import config
from src.core.client import Trading212Client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_connection")


async def main():
    """執行連線測試"""
    print("=" * 60)
    print("  AI Trading System — 連線測試")
    print("=" * 60)

    # ── 1. 檢查配置 ─────────────────────────────────────────────
    print(f"\n[配置檢查]")
    print(f"  環境 (ENV):     {config.ENV}")
    print(f"  Base URL:       {config.BASE_URL}")
    print(f"  API Key:        {'***' + config.API_KEY[-4:] if len(config.API_KEY) > 4 else '(未設定)'}")
    print(f"  最大重試次數:   {config.MAX_RETRIES}")
    print(f"  初始退避秒數:   {config.INITIAL_BACKOFF}s")
    print(f"  最大退避秒數:   {config.MAX_BACKOFF}s")
    print(f"  肥手指防護:     {config.MAX_ORDER_VALUE_PCT:.0%}")
    print(f"  日成交量上限:   {config.MAX_DAILY_VOLUME_PCT:.0%}")
    print(f"  最大掛單數:     {config.MAX_PENDING_ORDERS}")

    if config.API_KEY in ("", "your_api_key_here"):
        print("\n[錯誤] API Key 尚未設定！")
        print("  請在環境變數或 .env 檔案中設定 T212_API_KEY。")
        print("  參考 .env.example 檔案。")
        return

    # ── 2. 建立客戶端並測試連線 ──────────────────────────────────
    print(f"\n[連線測試] 正在連接 {config.BASE_URL} ...")
    client = Trading212Client()

    try:
        result = await client.get_account_info()
        print(f"\n[成功] API 連線正常！")
        print(f"  帳戶資訊回應: {result}")

        # 顯示速率限制器狀態
        rl = client.rate_limiter
        print(f"\n[速率限制器狀態]")
        print(f"  剩餘請求額度:  {rl.remaining}")
        print(f"  重置時間:      {rl.reset_time}")
        print(f"  安全邊際:      {rl.safe_margin}")

    except Exception as e:
        print(f"\n[失敗] 連線錯誤: {e}")
        logger.exception("Connection test failed")

    finally:
        await client.close()
        print(f"\n[完成] 連線已關閉。")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
