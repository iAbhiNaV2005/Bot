"""
delivery/message_templates.py — Signal message format templates (v2).

Uses HTML parse mode (not Markdown) for reliable Telegram rendering.
Updated to include session, BTC context, ADX, Fibonacci zone,
liquidity sweep, RSI divergence, OB touch/rejection info.
"""

from datetime import datetime, timezone
from typing import Optional

from signals.long_signal import Signal
from utils.logger import get_logger

logger = get_logger(__name__)


def _format_price(price: float) -> str:
    """
    Format a price with appropriate decimal places.

    High-value coins (>100) get 2 decimals, others get up to 6.

    Args:
        price: Price value to format.

    Returns:
        Formatted price string with comma separators.
    """
    if price >= 100:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:,.4f}"
    else:
        return f"${price:,.6f}"


def format_signal_message(signal: Signal) -> str:
    """
    Format a complete signal message in HTML for Telegram (v2).

    Updated format:
    - Direction emoji + coin + timeframe
    - Score X/32 + confidence tag
    - Entry zone, TP1, TP2, TP3, SL
    - SMC basis section (with sweep, divergence, OB touch, rejection, fib)
    - Indicator basis section
    - Market context section (with session, BTC, ADX)
    - Disclaimer + timestamp

    Args:
        signal: Complete Signal object.

    Returns:
        HTML-formatted string for Telegram.
    """
    # Direction header
    if signal.direction == "LONG":
        emoji = "🟢"
        direction = "LONG"
    else:
        emoji = "🔴"
        direction = "SHORT"

    # Clean symbol for display (ETH/USDT:USDT -> ETHUSDT)
    display_symbol = signal.symbol.replace("/", "").replace(":USDT", "")

    # Confidence tag
    confidence_tag = " 🔥 HIGH CONFIDENCE" if signal.confidence == "HIGH" else ""

    # TP3 line
    tp3_line = ""
    if signal.tp3 is not None:
        tp3_line = f"\n🎯 TP3 (1:5)  : {_format_price(signal.tp3)}"

    # SMC basis
    smc_lines = "\n".join(f"  • {item}" for item in signal.smc_basis)

    # Indicator basis
    indicator_lines = "\n".join(f"  • {item}" for item in signal.indicator_basis)

    # Market context
    ctx = signal.market_context
    context_lines = []

    # v2: Session + BTC context first
    if "session" in ctx:
        context_lines.append(f"  • Session: {ctx['session']}")
    if "btc_context" in ctx:
        context_lines.append(f"  • BTC: {ctx['btc_context']}")
    if "adx" in ctx:
        context_lines.append(f"  • ADX (1H): {ctx['adx']}")

    # Standard market data
    if "funding_rate" in ctx:
        context_lines.append(f"  • Funding: {ctx['funding_rate']}")
    if "open_interest" in ctx:
        context_lines.append(f"  • OI: {ctx['open_interest']}")
    if "ls_ratio" in ctx:
        context_lines.append(f"  • L/S Ratio: {ctx['ls_ratio']}")

    context_block = "\n".join(context_lines) if context_lines else "  • N/A"

    # Timestamp
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    message = (
        f"{emoji} <b>{direction} — {display_symbol} PERP [5m/15m]</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Score: {signal.score}/{signal.max_score}{confidence_tag}\n"
        f"\n"
        f"📍 Entry Zone : {_format_price(signal.entry)}\n"
        f"🎯 TP1 (1:1.5): {_format_price(signal.tp1)}\n"
        f"🎯 TP2 (1:3)  : {_format_price(signal.tp2)}"
        f"{tp3_line}\n"
        f"🛑 Stop Loss  : {_format_price(signal.stop_loss)}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📐 <b>SMC Basis:</b>\n"
        f"{smc_lines}\n"
        f"\n"
        f"📈 <b>Indicator Basis:</b>\n"
        f"{indicator_lines}\n"
        f"\n"
        f"🌐 <b>Market Context:</b>\n"
        f"{context_block}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ Not financial advice. Manage risk.\n"
        f"🕐 {now}"
    )

    return message


def format_startup_message() -> str:
    """
    Format the bot startup notification message.

    Returns:
        HTML-formatted startup message.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"✅ <b>Bot Started (v2)</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 SMC Momentum Confluence Bot\n"
        f"🔄 Scanning top 50 USDT futures by volume\n"
        f"⏰ 5-minute candle cycle active\n"
        f"🆕 ADX gate | Fibonacci zones | Session filter\n"
        f"🆕 Liquidity sweeps | RSI divergence | BTC correlation\n"
        f"🕐 {now}"
    )


def format_restart_message() -> str:
    """Format bot restart notification."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"🔄 <b>BOT RESTARTED (v2)</b>\n🕐 {now}"


def format_error_alert(key: str, count: int) -> str:
    """
    Format an error threshold alert.

    Args:
        key: Context key (e.g., symbol name).
        count: Number of consecutive errors.

    Returns:
        HTML-formatted error alert.
    """
    return (
        f"⚠️ <b>BOT WARNING</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{count} consecutive API errors for <b>{key}</b>.\n"
        f"Check logs for details."
    )


def format_daily_report(
    total_signals: int,
    long_signals: int,
    short_signals: int,
    high_confidence: int,
    avg_score: float,
    coins_scanned: int,
    tp1_hit_rate: Optional[float] = None,
) -> str:
    """
    Format the daily status report (v2: includes TP1 hit rate).

    Args:
        total_signals: Total signals sent today.
        long_signals: Number of long signals.
        short_signals: Number of short signals.
        high_confidence: Number of high-confidence signals.
        avg_score: Average signal score.
        coins_scanned: Number of coins actively scanned.
        tp1_hit_rate: TP1 hit rate percentage (v2, optional).

    Returns:
        HTML-formatted daily report.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # v2: TP1 hit rate line
    hit_rate_line = ""
    if tp1_hit_rate is not None:
        hit_rate_line = f"\n  📊 TP1 Hit Rate: {tp1_hit_rate:.0f}%"

    return (
        f"📋 <b>Daily Report — {now}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Total Signals: {total_signals}\n"
        f"  🟢 Long: {long_signals}\n"
        f"  🔴 Short: {short_signals}\n"
        f"  🔥 High Confidence: {high_confidence}\n"
        f"  📈 Avg Score: {avg_score:.1f}/32"
        f"{hit_rate_line}\n"
        f"  🔍 Coins Scanned: {coins_scanned}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ Bot operating normally"
    )
