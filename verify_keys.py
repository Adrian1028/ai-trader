"""
API Key Verification Script
============================
Run this AFTER creating your .env file to verify all keys work.

Usage:
    python verify_keys.py
"""
import asyncio
import os
import sys
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    print("[!] python-dotenv not installed. Run: pip install python-dotenv")
    sys.exit(1)

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))


async def check_trading212():
    """Verify Trading 212 API key using Basic Auth."""
    import aiohttp
    import base64
    env = os.getenv("T212_ENV", "demo")
    key = os.getenv("T212_API_KEY", "")
    secret = os.getenv("T212_API_SECRET", "")
    if not key or key == "your_trading212_api_key_here":
        return False, "T212_API_KEY not set"
    if not secret:
        return False, "T212_API_SECRET not set"

    # Trading 212 uses Basic Auth: Base64(API_KEY:API_SECRET)
    credentials = base64.b64encode(f"{key}:{secret}".encode()).decode()
    base = f"https://{env}.trading212.com/api/v0"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base}/equity/account/cash",
                headers={"Authorization": f"Basic {credentials}"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    cash = data.get("free", data.get("total", "?"))
                    return True, f"Connected ({env} mode) | Cash: {cash}"
                elif resp.status == 401:
                    return False, "Invalid API key or secret (401 Unauthorized)"
                else:
                    return False, f"HTTP {resp.status}"
    except Exception as e:
        return False, str(e)


async def check_alpha_vantage():
    """Verify Alpha Vantage API key."""
    import aiohttp
    key = os.getenv("ALPHA_VANTAGE_KEY", "")
    if not key or key == "your_alpha_vantage_key_here":
        return False, "ALPHA_VANTAGE_KEY not set"

    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={key}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()
                if "Global Quote" in data:
                    price = data["Global Quote"].get("05. price", "?")
                    return True, f"AAPL price: ${price}"
                elif "Note" in data:
                    return True, "Key valid (rate limit hit — normal for free tier)"
                elif "Error Message" in data:
                    return False, "Invalid API key"
                else:
                    return False, f"Unexpected response: {list(data.keys())}"
    except Exception as e:
        return False, str(e)


async def check_polygon():
    """Verify Polygon.io API key."""
    import aiohttp
    key = os.getenv("POLYGON_KEY", "")
    if not key or key == "your_polygon_key_here":
        return False, "POLYGON_KEY not set"

    url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey={key}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("results"):
                        close = data["results"][0].get("c", "?")
                        return True, f"AAPL prev close: ${close}"
                    return True, "Key valid"
                elif resp.status == 401 or resp.status == 403:
                    return False, "Invalid API key"
                else:
                    return False, f"HTTP {resp.status}"
    except Exception as e:
        return False, str(e)


async def check_finnhub():
    """Verify Finnhub API key."""
    import aiohttp
    key = os.getenv("FINNHUB_KEY", "")
    if not key or key == "your_finnhub_key_here":
        return False, "FINNHUB_KEY not set"

    url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={key}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price = data.get("c", "?")
                    return True, f"AAPL current: ${price}"
                elif resp.status == 401 or resp.status == 403:
                    return False, "Invalid API key"
                else:
                    return False, f"HTTP {resp.status}"
    except Exception as e:
        return False, str(e)


async def main():
    print("=" * 60)
    print("  AI Trading System — API Key Verification")
    print("=" * 60)
    print()

    checks = [
        ("Trading 212", check_trading212),
        ("Alpha Vantage", check_alpha_vantage),
        ("Polygon.io", check_polygon),
        ("Finnhub", check_finnhub),
    ]

    all_ok = True
    for name, check_fn in checks:
        ok, msg = await check_fn()
        status = "OK" if ok else "FAIL"
        icon = "+" if ok else "X"
        print(f"  [{icon}] {name:16s} {status:6s} | {msg}")
        if not ok:
            all_ok = False

    # Intrinio (optional)
    intrinio_key = os.getenv("INTRINIO_KEY", "")
    if intrinio_key and intrinio_key != "placeholder":
        print(f"  [~] {'Intrinio':16s} {'SKIP':6s} | Key provided but not verified (optional)")
    else:
        print(f"  [~] {'Intrinio':16s} {'SKIP':6s} | Not configured (optional — system works without it)")

    print()
    print("=" * 60)
    if all_ok:
        print("  ALL CHECKS PASSED — Ready to launch!")
        print()
        print("  Start the bot:        python src/main.py")
        print("  Start the dashboard:  streamlit run src/dashboard.py")
    else:
        print("  SOME CHECKS FAILED — Fix the .env file and try again")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
