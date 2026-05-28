"""
delivery/message_templates.py — Signal message format templates.

Uses HTML parse mode (not Markdown) for reliable Telegram rendering.
Every template matches the exact format specified in Part 10 of the strategy.
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
    Format a complete signal message in HTML for Telegram.

    Matches the exact template from Part 10:
    - Direction emoji + coin + timeframe
    - Score + confidence tag
    - Entry zone, TP1, TP2, TP3, SL
    - SMC basis section
    - Indicator basis section
    - Market context section
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
    funding_line = f"  • Funding Rate: {ctx.get('funding_rate', 'N/A')}"
    oi_line = f"  • Open Interest: {ctx.get('open_interest', 'N/A')}"
    ls_line = f"  • L/S Ratio: {ctx.get('ls_ratio', 'N/A')}"

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
        f"{funding_line}\n"
        f"{oi_line}\n"
        f"{ls_line}\n"
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
        f"✅ <b>Bot Started</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 SMC Momentum Confluence Bot\n"
        f"🔄 Scanning top 50 USDT futures by volume\n"
        f"⏰ 5-minute candle cycle active\n"
        f"🕐 {now}"
    )


def format_restart_message() -> str:
    """Format bot restart notification."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"🔄 <b>BOT RESTARTED</b>\n🕐 {now}"


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
) -> str:
    """
    Format the daily status report.

    Args:
        total_signals: Total signals sent today.
        long_signals: Number of long signals.
        short_signals: Number of short signals.
        high_confidence: Number of high-confidence signals.
        avg_score: Average signal score.
        coins_scanned: Number of coins actively scanned.

    Returns:
        HTML-formatted daily report.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return (
        f"📋 <b>Daily Report — {now}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Total Signals: {total_signals}\n"
        f"  🟢 Long: {long_signals}\n"
        f"  🔴 Short: {short_signals}\n"
        f"  🔥 High Confidence: {high_confidence}\n"
        f"  📈 Avg Score: {avg_score:.1f}\n"
        f"  🔍 Coins Scanned: {coins_scanned}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ Bot operating normally"
    )
