"""
main.py — Entry point for the Crypto Signal Bot.

Starts the APScheduler with the main loop and periodic jobs,
handles graceful shutdown on SIGTERM/SIGINT.

Usage:
    python main.py             # Normal run
    python main.py --dry-run   # Print signals to console instead of Telegram
"""

import argparse
import asyncio
import signal
import sys
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from config import settings
from delivery.telegram_bot import telegram
from scheduler.runner import runner
from utils.logger import get_logger

logger = get_logger(__name__)

# Global scheduler reference
_scheduler: AsyncIOScheduler | None = None


async def main_loop_job() -> None:
    """Job executed every MAIN_LOOP_INTERVAL_SECONDS by APScheduler."""
    await runner.main_loop_tick()


async def daily_refresh_job() -> None:
    """Job executed daily at midnight UTC."""
    await runner.daily_refresh()


def handle_shutdown(sig: int, frame: any) -> None:
    """
    Handle SIGTERM/SIGINT for graceful shutdown.

    Args:
        sig: Signal number.
        frame: Current stack frame.
    """
    logger.info(f"Received signal {sig}. Initiating shutdown...")
    if _scheduler:
        _scheduler.shutdown(wait=False)


async def run_bot(dry_run: bool = False) -> None:
    """
    Main bot execution flow.

    Args:
        dry_run: If True, print signals to console instead of Telegram.
    """
    global _scheduler

    logger.info("=" * 60)
    logger.info("  SMC Momentum Confluence Signal Bot")
    logger.info("=" * 60)
    logger.info(f"  Exchange: {settings.EXCHANGE} ({settings.MARKET_TYPE})")
    logger.info(f"  Top coins: {settings.TOP_N_COINS}")
    logger.info(f"  Timeframes: {settings.MACRO_TF}/{settings.STRUCTURE_TF}/"
                f"{settings.SETUP_TF}/{settings.ENTRY_TF}")
    logger.info(f"  Min score: {settings.MIN_SCORE_TO_SIGNAL}/{settings.MAX_POSSIBLE_SCORE}")
    logger.info(f"  Min R:R: {settings.MIN_RR_TO_SIGNAL}")
    logger.info(f"  Dry run: {dry_run}")
    logger.info("=" * 60)

    if dry_run:
        logger.info("DRY RUN MODE — signals will be printed, not sent to Telegram")

    # ── Validate config ──
    if not settings.BINANCE_API_KEY or not settings.BINANCE_API_SECRET:
        logger.error("Binance API credentials not set! Check your .env file.")
        sys.exit(1)

    if not dry_run and (not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID):
        logger.error("Telegram credentials not set! Check your .env file.")
        sys.exit(1)

    # ── Run startup sequence ──
    try:
        await runner.startup()
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        try:
            await telegram.send_text(f"⚠️ <b>BOT STARTUP FAILED</b>\n{str(e)[:200]}")
        except Exception:
            pass
        await runner.shutdown()  # Clean up exchange connection
        sys.exit(1)

    # ── Configure scheduler ──
    _scheduler = AsyncIOScheduler(timezone="UTC")

    # Main loop: every 60 seconds
    _scheduler.add_job(
        main_loop_job,
        IntervalTrigger(seconds=settings.MAIN_LOOP_INTERVAL_SECONDS),
        id="main_loop",
        name="Main Analysis Loop",
        max_instances=1,
        replace_existing=True,
    )

    # Daily refresh: midnight UTC
    _scheduler.add_job(
        daily_refresh_job,
        CronTrigger(hour=0, minute=0),
        id="daily_refresh",
        name="Daily Coin Refresh",
        max_instances=1,
        replace_existing=True,
    )

    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # ── Start scheduler ──
    _scheduler.start()
    logger.info("Scheduler started. Bot is now running.")

    # Keep the event loop alive
    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Received shutdown signal")
    finally:
        await runner.shutdown()
        if _scheduler and _scheduler.running:
            _scheduler.shutdown(wait=False)
        logger.info("Bot stopped.")


def main() -> None:
    """Parse arguments and run the bot."""
    parser = argparse.ArgumentParser(
        description="SMC Momentum Confluence Signal Bot"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print signals to console instead of sending to Telegram",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_bot(dry_run=args.dry_run))
    except KeyboardInterrupt:
        logger.info("Bot terminated by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        # Try to send restart notification on next start
        sys.exit(1)


if __name__ == "__main__":
    main()
