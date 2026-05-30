"""
backtest/downloader.py — Phase 0: Historical data download and caching.

Downloads OHLCV and funding rate data from Binance Futures REST API.
All downloads are cached as CSV files. Existing files are NOT re-downloaded
unless --refresh-coins is set.

Pagination: Binance returns max 1500 candles per call. For 6 months:
  4H  → ~1080 candles → 1 call
  1H  → ~4320 candles → 3 calls
  15m → ~17280 candles → 12 calls
  5m  → ~51840 candles → 35 calls

Total API calls for 10 coins: ~510 calls at 0.2s delay = ~102 seconds.
"""

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

from backtest import config


# ---------------------------------------------------------------------------
# Stablecoin filter (same list as live bot)
# ---------------------------------------------------------------------------
_STABLECOINS = {
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "DAIUSDT",
    "FDUSDUSDT", "USDPUSDT",
}


def _get_6m_start_ms() -> int:
    """Return Unix milliseconds for 6 months ago UTC."""
    start = datetime.now(timezone.utc) - timedelta(days=183)
    return int(start.timestamp() * 1000)


def _symbol_to_filename(symbol: str, timeframe: str) -> str:
    """
    Convert a ccxt-style symbol to a flat filename.

    'BTC/USDT:USDT' → 'BTCUSDT_4h.csv'
    """
    base = symbol.replace("/", "").replace(":USDT", "").replace("USDT", "")
    return f"{base}USDT_{timeframe}.csv"


def _funding_filename(symbol: str) -> str:
    base = symbol.replace("/", "").replace(":USDT", "").replace("USDT", "")
    return f"{base}USDT_funding.csv"


def _binance_symbol(symbol: str) -> str:
    """Convert ccxt symbol to Binance REST format: 'BTC/USDT:USDT' → 'BTCUSDT'"""
    return symbol.replace("/", "").replace(":USDT", "")


def _download_klines(
    binance_sym: str,
    interval: str,
    start_ms: int,
) -> pd.DataFrame:
    """
    Download all klines for a symbol/interval from start_ms to now.

    Uses pagination: each call fetches up to 1500 candles starting from
    the last returned close time.

    Args:
        binance_sym: Binance symbol string (e.g. 'BTCUSDT').
        interval:    Binance interval string (e.g. '4h', '15m').
        start_ms:    Start timestamp in Unix milliseconds.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    url = f"{config.BINANCE_FUTURES_BASE}/fapi/v1/klines"
    all_rows: list[list] = []
    current_start = start_ms
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    while current_start < now_ms:
        params = {
            "symbol": binance_sym,
            "interval": interval,
            "startTime": current_start,
            "limit": config.KLINES_PER_CALL,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        all_rows.extend(data)

        # Next page starts from the close time of the last candle + 1ms
        last_close_time = int(data[-1][6])
        current_start = last_close_time + 1

        # Safety: if last batch was smaller than limit, we're done
        if len(data) < config.KLINES_PER_CALL:
            break

        time.sleep(config.DOWNLOAD_DELAY_SECONDS)

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df.rename(columns={"open_time": "timestamp"}, inplace=True)
    df["timestamp"] = df["timestamp"].astype(int)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Drop the last (incomplete) candle if it's still open
    df = df[df["timestamp"] < now_ms].reset_index(drop=True)

    return df


def _download_funding_rates(binance_sym: str) -> pd.DataFrame:
    """
    Download funding rate history for a symbol.

    Binance funding rate: every 8 hours. 6 months ≈ 546 entries.
    One call (limit=1000) covers the full period.

    Args:
        binance_sym: Binance symbol string.

    Returns:
        DataFrame with columns: timestamp (ms), funding_rate (float).
    """
    url = f"{config.BINANCE_FUTURES_BASE}/fapi/v1/fundingRate"
    start_ms = _get_6m_start_ms()
    all_rows: list[dict] = []

    # Binance funding rate endpoint returns max 1000 per call, paginate if needed
    current_start = start_ms
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    while current_start < now_ms:
        params = {
            "symbol": binance_sym,
            "startTime": current_start,
            "limit": 1000,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        all_rows.extend(data)

        last_ts = int(data[-1]["fundingTime"])
        current_start = last_ts + 1

        if len(data) < 1000:
            break

        time.sleep(config.DOWNLOAD_DELAY_SECONDS)

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    df = pd.DataFrame(all_rows)
    df = df.rename(columns={"fundingTime": "timestamp", "fundingRate": "funding_rate"})
    df = df[["timestamp", "funding_rate"]].copy()
    df["timestamp"] = df["timestamp"].astype(int)
    df["funding_rate"] = df["funding_rate"].astype(float)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _fetch_top_coins(n: int = config.TOP_N_COINS) -> list[str]:
    """
    Fetch top N coins by 24h quote volume from Binance Futures.

    Returns:
        List of Binance symbol strings (e.g. ['BTCUSDT', 'ETHUSDT', ...]).
    """
    # Get all perpetual futures
    markets_url = f"{config.BINANCE_FUTURES_BASE}/fapi/v1/exchangeInfo"
    resp = requests.get(markets_url, timeout=30)
    resp.raise_for_status()
    info = resp.json()

    usdt_perps = [
        s["symbol"] for s in info["symbols"]
        if s["quoteAsset"] == "USDT"
        and s["contractType"] == "PERPETUAL"
        and s["status"] == "TRADING"
        and s["symbol"] not in _STABLECOINS
        and not any(sc in s["symbol"] for sc in ["USDCUSDT", "BUSDUSDT", "DAIUSDT"])
    ]

    # Get 24h tickers for all
    ticker_url = f"{config.BINANCE_FUTURES_BASE}/fapi/v1/ticker/24hr"
    resp = requests.get(ticker_url, timeout=30)
    resp.raise_for_status()
    tickers = {t["symbol"]: float(t.get("quoteVolume", 0) or 0) for t in resp.json()}

    # Filter stablecoins and sort by volume
    valid = [s for s in usdt_perps if s not in _STABLECOINS]
    sorted_by_vol = sorted(valid, key=lambda s: tickers.get(s, 0.0), reverse=True)

    top = sorted_by_vol[:n]
    print(f"  Top {n} coins selected: {top}")
    return top


def select_coins(refresh: bool = False) -> list[str]:
    """
    Select or load the top N coins for backtesting.

    If coin_list.txt exists and refresh=False, loads from file.
    Otherwise fetches from Binance and saves.

    Args:
        refresh: Force re-fetch from Binance.

    Returns:
        List of Binance symbol strings.
    """
    coin_file = Path(config.COIN_LIST_FILE)
    coin_file.parent.mkdir(parents=True, exist_ok=True)

    if coin_file.exists() and not refresh:
        coins = coin_file.read_text().strip().splitlines()
        coins = [c.strip() for c in coins if c.strip()]
        if coins:
            print(f"  Loaded {len(coins)} coins from cache: {coins}")
            return coins

    print("  Fetching top coins from Binance...")
    coins = _fetch_top_coins(config.TOP_N_COINS)
    coin_file.write_text("\n".join(coins))
    return coins


def download_all(coins: list[str], skip: bool = False) -> None:
    """
    Download all OHLCV and funding rate data for every coin.

    Files that already exist are NOT re-downloaded.

    Args:
        coins: List of Binance symbol strings.
        skip:  If True, skip all downloads (use existing CSVs).
    """
    data_dir = Path(config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    if skip:
        print("  --skip-download set. Using existing CSV files.")
        return

    start_ms = _get_6m_start_ms()
    timeframes = [("4h", "4h"), ("1h", "1h"), ("15m", "15m"), ("5m", "5m")]

    total_coins = len(coins)
    for coin_idx, symbol in enumerate(coins, 1):
        print(f"  [{coin_idx}/{total_coins}] Downloading {symbol}...")

        for tf_key, interval in timeframes:
            filename = _symbol_to_filename(symbol, tf_key)
            filepath = data_dir / filename

            if filepath.exists():
                print(f"    ✓ {filename} already exists — skipping")
                continue

            print(f"    ↓ {filename}...", end=" ", flush=True)
            df = _download_klines(symbol, interval, start_ms)
            df.to_csv(filepath, index=False)
            print(f"  {len(df)} candles")

            time.sleep(config.DOWNLOAD_DELAY_SECONDS)

        # Funding rates
        funding_file = data_dir / _funding_filename(symbol)
        if not funding_file.exists():
            print(f"    ↓ {_funding_filename(symbol)}...", end=" ", flush=True)
            df_fund = _download_funding_rates(symbol)
            df_fund.to_csv(funding_file, index=False)
            print(f"  {len(df_fund)} records")
            time.sleep(config.DOWNLOAD_DELAY_SECONDS)
        else:
            print(f"    ✓ {_funding_filename(symbol)} already exists — skipping")


def load_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Load a cached OHLCV CSV into a DataFrame.

    Args:
        symbol:    Binance symbol string (e.g. 'BTCUSDT').
        timeframe: Timeframe key ('4h', '1h', '15m', '5m').

    Returns:
        DataFrame with timestamp (ms int), open/high/low/close/volume floats.

    Raises:
        FileNotFoundError if the CSV does not exist.
    """
    filename = _symbol_to_filename(symbol, timeframe)
    filepath = Path(config.DATA_DIR) / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}. "
            f"Run without --skip-download first."
        )
    df = pd.read_csv(filepath)
    df["timestamp"] = df["timestamp"].astype(int)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_funding_rates(symbol: str) -> pd.DataFrame:
    """
    Load cached funding rate CSV for a symbol.

    Args:
        symbol: Binance symbol string.

    Returns:
        DataFrame with timestamp (ms int) and funding_rate (float).
    """
    filename = _funding_filename(symbol)
    filepath = Path(config.DATA_DIR) / filename
    if not filepath.exists():
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    df = pd.read_csv(filepath)
    df["timestamp"] = df["timestamp"].astype(int)
    df["funding_rate"] = df["funding_rate"].astype(float)
    return df.sort_values("timestamp").reset_index(drop=True)


def run(coins: list[str], refresh_coins: bool = False, skip_download: bool = False) -> list[str]:
    """
    Phase 0 entry point.

    Selects coins (or loads from cache), downloads all data.

    Args:
        coins:          Pre-selected coin list (empty to auto-select).
        refresh_coins:  Force re-select top coins.
        skip_download:  Skip download, use existing CSVs.

    Returns:
        List of selected Binance symbol strings.
    """
    print("\n=== Phase 0: Data Download ===")
    selected = select_coins(refresh=refresh_coins) if not coins else coins
    download_all(selected, skip=skip_download)
    print(f"  Data ready for {len(selected)} coins.\n")
    return selected
