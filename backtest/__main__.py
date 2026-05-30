"""
backtest/__main__.py — Entry point for the backtest system.

Run as: python -m backtest

Optional flags:
  --refresh-coins    Force re-select top 10 coins (ignores cached coin_list.txt)
  --skip-download    Skip data download (use existing CSVs)
  --phase1-only      Run Phase 0 + 1 and stop (saves candidates, exits)
  --phase2-only      Skip Phase 0 + 1, load candidates from DB, run Phase 2 + 3

Execution order:
  Phase 0 → Phase 1 → Phase 2 → Phase 3
"""

import argparse
import sys
import os
import time
from datetime import datetime, timezone

# Add project root to path so live bot modules are importable
_BACKTEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKTEST_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backtest import config
from backtest import downloader, walk_forward, rescorer, reporter, db


def _print_header() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print("=" * 60)
    print("  SMC Signal Bot — Backtesting Engine v1.0")
    print("=" * 60)
    print(f"  Started:        {now}")
    print(f"  Period:         {config.BACKTEST_PERIOD_MONTHS} months")
    print(f"  Coins:          Top {config.TOP_N_COINS} by 24h volume")
    print(f"  Combinations:   64 (4 × 4 × 2 × 2 parameter grid)")
    print(f"  Output DB:      {config.DB_PATH}")
    print(f"  Output HTML:    {config.HTML_REPORT_PATH}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m backtest",
        description="SMC Signal Bot — Full Backtesting System",
    )
    parser.add_argument(
        "--refresh-coins",
        action="store_true",
        help="Force re-select top 10 coins from Binance (ignores cached coin_list.txt)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip data download — use existing CSV files",
    )
    parser.add_argument(
        "--phase1-only",
        action="store_true",
        help="Run Phase 0 + Phase 1, save candidates to DB, then exit",
    )
    parser.add_argument(
        "--phase2-only",
        action="store_true",
        help="Skip Phase 0 + Phase 1, load candidates from DB, run Phase 2 + Phase 3",
    )
    args = parser.parse_args()

    _print_header()
    t_start = time.time()

    # ── Phase 0: Data Download ──
    if not args.phase2_only:
        coins = downloader.run(
            coins=[],
            refresh_coins=args.refresh_coins,
            skip_download=args.skip_download,
        )
    else:
        # Load coins from cached file
        import pathlib
        coin_file = pathlib.Path(config.COIN_LIST_FILE)
        if coin_file.exists():
            coins = coin_file.read_text().strip().splitlines()
            coins = [c.strip() for c in coins if c.strip()]
            print(f"\nLoaded {len(coins)} coins from cache for Phase 2.")
        else:
            print("ERROR: --phase2-only specified but no cached coin list found.")
            print("Run without --phase2-only first to download data.")
            sys.exit(1)

    # ── Phase 1: Walk-Forward Simulation ──
    if not args.phase2_only:
        db.init_db()
        t1 = time.time()
        candidates = walk_forward.run(coins, phase1_only=args.phase1_only)
        t1_elapsed = time.time() - t1
        print(f"\n  Phase 1 runtime: {t1_elapsed:.1f}s")

        if candidates:
            db.save_candidates(candidates)
            print(f"  {len(candidates)} candidates saved to DB.")
        else:
            print("  WARNING: No candidate signals generated. Check data and settings.")

        if args.phase1_only:
            print("\n  --phase1-only flag set. Stopping after Phase 1.")
            print(f"  Total runtime: {time.time() - t_start:.1f}s")
            return
    else:
        # Load candidates from DB
        print("\n  Loading candidate signals from DB...")
        raw_rows = db.load_candidates()
        if not raw_rows:
            print("ERROR: No candidate signals in DB. Run Phase 1 first.")
            sys.exit(1)

        # Reconstruct CandidateSignal objects from DB rows
        from backtest.candidate import CandidateSignal
        from datetime import datetime, timezone
        candidates = []
        for row in raw_rows:
            ts_str = row["timestamp"]
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except ValueError:
                ts = datetime.now(timezone.utc)

            c = CandidateSignal(
                coin=row["coin"],
                direction=row["direction"],
                timestamp=ts,
                candle_idx_5m=row["candle_idx_5m"],
                session=row["session"],
                entry_price=row["entry_price"],
                sl_raw_distance=row["sl_raw_distance"],
                ob_bottom=row["ob_bottom"],
                ob_top=row["ob_top"],
                sl_percentage=row["sl_percentage"],
                tp1_price=row["tp1_price"],
                tp2_price=row["tp2_price"],
                sl_price=row["sl_price"],
                gate_ema_4h_structure=bool(row["gate_ema_4h_structure"]),
                gate_bos_1h=bool(row["gate_bos_1h"]),
                gate_ob_present=bool(row["gate_ob_present"]),
                gate_adx_min=bool(row["gate_adx_min"]),
                gate_fib_zone=bool(row["gate_fib_zone"]),
                gate_rsi_range=bool(row["gate_rsi_range"]),
                gate_funding=bool(row["gate_funding"]),
                fvg_present=bool(row["fvg_present"]),
                volume_ratio_15m=row["volume_ratio_15m"],
                volume_ratio_5m=row["volume_ratio_5m"],
                oi_increasing=bool(row["oi_increasing"]),
                choch_present=bool(row["choch_present"]),
                macd_histogram_trend=bool(row["macd_histogram_trend"]),
                ls_ratio=row["ls_ratio"],
                funding_rate=row["funding_rate"],
                ema200_gap_pct=row["ema200_gap_pct"],
                sweep_detected=bool(row["sweep_detected"]),
                rsi_divergence=bool(row["rsi_divergence"]),
                adx_value=row["adx_value"],
                ob_rejection_candle=bool(row["ob_rejection_candle"]),
                ob_touch_count=row["ob_touch_count"],
                fib_deep_discount=bool(row["fib_deep_discount"]),
                btc_bullish=bool(row["btc_bullish"]),
                btc_bearish=bool(row["btc_bearish"]),
                equal_level_present=bool(row["equal_level_present"]),
                equal_level_count=row["equal_level_count"],
                ob_volume_quality=bool(row["ob_volume_quality"]),
                outcome=row["outcome"],
                tp1_hit=bool(row["tp1_hit"]),
                tp2_hit=bool(row["tp2_hit"]),
                sl_hit=bool(row["sl_hit"]),
                candles_to_outcome=row.get("candles_to_outcome"),
                max_adverse_excursion=row.get("max_adverse_excursion", 0.0),
                max_favorable_excursion=row.get("max_favorable_excursion", 0.0),
            )
            candidates.append(c)

        print(f"  Loaded {len(candidates)} candidates from DB.")

    # ── Phase 2: Re-Scoring ──
    t2 = time.time()
    combo_results = rescorer.run(candidates)
    t2_elapsed = time.time() - t2
    print(f"\n  Phase 2 runtime: {t2_elapsed:.1f}s")

    db.save_results(combo_results)
    print(f"  {len(combo_results)} combo results saved to DB.")

    # ── Phase 3: Report Generation ──
    t3 = time.time()
    reporter.run(combo_results)
    t3_elapsed = time.time() - t3
    print(f"\n  Phase 3 runtime: {t3_elapsed:.1f}s")

    total = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Backtest complete in {total:.1f}s ({total / 60:.1f} minutes)")
    print(f"  DB:   {config.DB_PATH}")
    print(f"  HTML: {config.HTML_REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
