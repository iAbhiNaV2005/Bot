"""Quick test for OI + L/S ratio endpoints."""
import asyncio
import aiohttp
import ccxt.async_support as ccxt


async def test():
    connector = aiohttp.TCPConnector(resolver=aiohttp.resolver.ThreadedResolver())
    session = aiohttp.ClientSession(connector=connector)

    exchange = ccxt.binance({
        "options": {"defaultType": "future", "fetchCurrencies": False},
        "session": session,
        "enableRateLimit": True,
    })
    try:
        # Test OI history
        url = "https://fapi.binance.com/futures/data/openInterestHist?symbol=BTCUSDT&period=5m&limit=3"
        result = await exchange.fetch(url)
        print(f"OI History: {len(result)} entries")
        if result:
            print(f"  Sample: {result[0]}")

        # Test L/S ratio
        url2 = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol=BTCUSDT&period=5m&limit=1"
        result2 = await exchange.fetch(url2)
        print(f"L/S Ratio: {result2}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
    finally:
        if not session.closed:
            await session.close()
        await exchange.close()


asyncio.run(test())
