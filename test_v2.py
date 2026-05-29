"""Quick verification script for v2 changes."""
import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, '.')

# Test 1: Runner import
from scheduler.runner import runner
print("[OK] runner import OK")

# Test 2: Indicator calculations
import pandas as pd
import numpy as np

np.random.seed(42)
n = 100
close = pd.Series(np.cumsum(np.random.randn(n)) + 100)
high = close + abs(np.random.randn(n))
low = close - abs(np.random.randn(n))

from analysis.indicators import calc_adx, calc_atr, calc_rsi, calc_ema, calc_fibonacci_zones

rsi = calc_rsi(close)
print(f"[OK] RSI last: {rsi.iloc[-1]:.2f}")

atr = calc_atr(high, low, close)
print(f"[OK] ATR last: {atr.iloc[-1]:.4f}")

adx_res = calc_adx(high, low, close)
adx_val = adx_res["adx"].iloc[-1]
print(f"[OK] ADX last: {adx_val:.2f}")

# Test 3: Session detection
from utils.time_utils import get_trading_session
session = get_trading_session()
print(f"[OK] Session: {session['session_name']} (score: {session['session_score']:+d})")

# Test 4: Fibonacci zones
from analysis.smc import SwingPoint, Direction

sh = [SwingPoint(index=80, price=105.0, direction=Direction.BULLISH, timestamp=pd.Timestamp.now())]
sl = [SwingPoint(index=60, price=95.0, direction=Direction.BEARISH, timestamp=pd.Timestamp.now())]
fib = calc_fibonacci_zones(sh, sl, current_price=100.0)
print(f"[OK] Fibonacci: trend={fib['trend_direction']}, retrace={fib['current_retrace_pct']:.1%}")
print(f"     Fib levels: 38.2%={fib['fib_382']:.2f}, 50%={fib['fib_500']:.2f}, 61.8%={fib['fib_618']:.2f}")

# Test 5: Full indicator pipeline
df_mock = pd.DataFrame({
    'open': close - 0.5,
    'high': high,
    'low': low,
    'close': close,
    'volume': np.random.randint(1000, 10000, n).astype(float),
})

from analysis.indicators import calculate_all_indicators

ohlcv = {'4h': df_mock.copy(), '1h': df_mock.copy(), '15m': df_mock.copy(), '5m': df_mock.copy()}
indicators = calculate_all_indicators(ohlcv)

print(f"[OK] Nested dict keys: {list(indicators.keys())}")
print(f"     EMA 4h_50: {indicators['ema']['4h_50']:.2f}")
print(f"     RSI 5m: {indicators['rsi']['5m_current']:.2f}")
print(f"     ADX 1h: {indicators['adx']['1h_current']:.2f}")
print(f"     ATR 15m: {indicators['atr']['15m_current']:.4f}")
print(f"     Volume 5m ratio: {indicators['volume']['5m_ratio']:.2f}")
print(f"     MACD hist last3: {indicators['macd']['histogram_last3']}")

print("\n=== ALL VERIFICATION TESTS PASSED ===")
