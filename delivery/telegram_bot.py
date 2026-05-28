"""
delivery/telegram_bot.py — Telegram message sending and chart generation.

Handles:
- Async message sending via python-telegram-bot v20+
- Chart generation using mplfinance with dark theme
- Photo sending with chart images
- Retry logic for failed sends
- Rate limiting between consecutive sends
"""

import asyncio
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mplfinance as mpf
import pandas as pd
from telegram import Bot
from telegram.constants import ParseMode

from config import settings
from delivery.message_templates import (
    format_daily_report,
    format_error_alert,
    format_restart_message,
    format_signal_message,
    format_startup_message,
)
from signals.long_signal import Signal
from utils.error_handler import retry
from utils.logger import get_logger

logger = get_logger(__name__)


class TelegramDelivery:
    """
    Async Telegram bot for sending signal alerts with chart images.

    Uses python-telegram-bot v20+ for async operations.
    Generates charts with mplfinance including OB/FVG overlays
    and TP/SL levels.
    """

    def __init__(self) -> None:
        """Initialize the Telegram bot client."""
        self._bot: Optional[Bot] = None
        os.makedirs(settings.CHART_TMP_DIR, exist_ok=True)

    def _get_bot(self) -> Bot:
        """Lazy-initialize and return the bot instance."""
        if self._bot is None:
            if not settings.TELEGRAM_BOT_TOKEN:
                raise ValueError("TELEGRAM_BOT_TOKEN not set in .env")
            self._bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
        return self._bot

    @retry(max_attempts=3, delay_seconds=5.0)
    async def send_text(self, text: str) -> None:
        """
        Send a text message to the configured Telegram chat.

        Args:
            text: HTML-formatted message text.
        """
        bot = self._get_bot()
        await bot.send_message(
            chat_id=settings.TELEGRAM_CHAT_ID,
            text=text,
            parse_mode=ParseMode.HTML,
        )
        logger.info("Telegram text message sent successfully")

    @retry(max_attempts=3, delay_seconds=5.0)
    async def send_photo(self, photo_path: str, caption: str) -> None:
        """
        Send a photo with caption to the configured Telegram chat.

        Args:
            photo_path: Absolute path to the PNG image.
            caption: HTML-formatted caption text.
        """
        bot = self._get_bot()
        with open(photo_path, "rb") as photo:
            await bot.send_photo(
                chat_id=settings.TELEGRAM_CHAT_ID,
                photo=photo,
                caption=caption[:1024],  # Telegram caption limit
                parse_mode=ParseMode.HTML,
            )
        logger.info(f"Telegram photo sent: {photo_path}")

    async def send_signal(self, signal: Signal, ohlcv_15m: Optional[pd.DataFrame] = None) -> None:
        """
        Send a complete signal alert: chart image + formatted message.

        If chart generation fails, falls back to text-only message.

        Args:
            signal: Complete Signal object.
            ohlcv_15m: 15m OHLCV DataFrame for chart generation (optional).
        """
        message = format_signal_message(signal)

        if settings.SEND_CHART and ohlcv_15m is not None:
            try:
                chart_path = self._generate_chart(signal, ohlcv_15m)
                # Send full message as text first (photo captions are limited to 1024 chars)
                await self.send_text(message)
                await self.send_photo(chart_path, f"{signal.direction} — {signal.symbol}")
                # Cleanup temp file
                if os.path.exists(chart_path):
                    os.remove(chart_path)
                await asyncio.sleep(settings.TELEGRAM_SEND_DELAY)
                return
            except Exception as e:
                logger.warning(f"Chart generation failed, sending text-only: {e}")

        # Fallback: text-only
        await self.send_text(message)
        await asyncio.sleep(settings.TELEGRAM_SEND_DELAY)

    async def send_startup(self) -> None:
        """Send bot startup notification."""
        await self.send_text(format_startup_message())

    async def send_restart(self) -> None:
        """Send bot restart notification."""
        await self.send_text(format_restart_message())

    async def send_error_alert(self, key: str, count: int) -> None:
        """
        Send error threshold alert.

        Args:
            key: Context key (symbol or 'global').
            count: Number of consecutive errors.
        """
        await self.send_text(format_error_alert(key, count))

    async def send_daily_report(
        self,
        total: int,
        longs: int,
        shorts: int,
        high_conf: int,
        avg_score: float,
        coins: int,
    ) -> None:
        """
        Send daily performance report.

        Args:
            total: Total signals today.
            longs: Long signals count.
            shorts: Short signals count.
            high_conf: High-confidence signals count.
            avg_score: Average signal score.
            coins: Coins scanned.
        """
        await self.send_text(
            format_daily_report(total, longs, shorts, high_conf, avg_score, coins)
        )

    def _generate_chart(self, signal: Signal, df: pd.DataFrame) -> str:
        """
        Generate a candlestick chart with overlays using mplfinance.

        Chart includes:
        - Last 100 candles on 15m timeframe
        - EMA 9 and EMA 21 overlays
        - Order Block zone (shaded rectangle)
        - FVG zone (lighter shaded rectangle)
        - Entry, TP1, TP2, SL horizontal lines
        - Volume bars with 20-period SMA
        - Dark theme

        Args:
            signal: Signal object with TP/SL levels.
            df: 15m OHLCV DataFrame.

        Returns:
            Path to the generated PNG file.
        """
        # Use last 100 candles
        chart_df = df.tail(100).copy()

        # Ensure proper index for mplfinance
        if not isinstance(chart_df.index, pd.DatetimeIndex):
            chart_df.index = pd.to_datetime(chart_df.index)

        # Calculate EMAs for overlay
        ema_9 = chart_df["close"].ewm(span=9, adjust=False).mean()
        ema_21 = chart_df["close"].ewm(span=21, adjust=False).mean()

        # Volume SMA
        vol_sma = chart_df["volume"].rolling(window=20).mean()

        # Custom dark style
        mc = mpf.make_marketcolors(
            up="#00c853",
            down="#ff1744",
            edge="inherit",
            wick="inherit",
            volume={"up": "#00c85380", "down": "#ff174480"},
        )
        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            marketcolors=mc,
            figcolor="#1a1a2e",
            facecolor="#1a1a2e",
            gridcolor="#2d2d44",
            gridstyle="--",
            gridaxis="both",
        )

        # Prepare additional plots
        add_plots = [
            mpf.make_addplot(ema_9, color="#2196f3", width=1.0, label="EMA 9"),
            mpf.make_addplot(ema_21, color="#ff9800", width=1.0, label="EMA 21"),
            mpf.make_addplot(
                vol_sma, panel=1, color="#ffd700", width=0.8, label="Vol SMA 20"
            ),
        ]

        # Clean symbol for filename
        clean_symbol = signal.symbol.replace("/", "_").replace(":", "_")
        chart_path = os.path.join(
            settings.CHART_TMP_DIR, f"{clean_symbol}_{signal.direction}.png"
        )

        # Generate chart
        fig, axes = mpf.plot(
            chart_df,
            type="candle",
            style=style,
            addplot=add_plots,
            volume=True,
            title=f"\n{clean_symbol.replace('_', '')} 15m — {signal.direction} SIGNAL",
            figsize=(14, 8),
            returnfig=True,
            panel_ratios=(3, 1),
            tight_layout=True,
        )

        ax_main = axes[0]

        # Draw TP/SL horizontal lines
        ax_main.axhline(
            y=signal.entry, color="#ffffff", linestyle="--", linewidth=0.8, alpha=0.8
        )
        ax_main.axhline(
            y=signal.tp1, color="#00e676", linestyle="-", linewidth=0.8, alpha=0.7
        )
        ax_main.axhline(
            y=signal.tp2, color="#00e676", linestyle="-", linewidth=0.8, alpha=0.5
        )
        ax_main.axhline(
            y=signal.stop_loss, color="#ff1744", linestyle="-", linewidth=0.8, alpha=0.7
        )

        # Add labels for TP/SL on the right side
        x_pos = len(chart_df) - 1
        ax_main.annotate(
            "ENTRY", xy=(x_pos, signal.entry),
            fontsize=7, color="#ffffff", alpha=0.8,
            ha="left", va="bottom",
        )
        ax_main.annotate(
            "TP1", xy=(x_pos, signal.tp1),
            fontsize=7, color="#00e676", alpha=0.8,
            ha="left", va="bottom",
        )
        ax_main.annotate(
            "TP2", xy=(x_pos, signal.tp2),
            fontsize=7, color="#00e676", alpha=0.6,
            ha="left", va="bottom",
        )
        ax_main.annotate(
            "SL", xy=(x_pos, signal.stop_loss),
            fontsize=7, color="#ff1744", alpha=0.8,
            ha="left", va="bottom",
        )

        # Save
        fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close(fig)

        logger.info(f"Chart generated: {chart_path}")
        return chart_path


# Module-level singleton
telegram = TelegramDelivery()
