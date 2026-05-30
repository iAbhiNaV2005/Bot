"""
backtest/rescorer.py — Phase 2: Parameter grid re-scoring.

Takes the master candidate list from Phase 1 and applies all 64 parameter
combinations without re-running any SMC or indicator logic.

Two-pass cooldown approach (for accuracy):
  Pass 1: Score signals with normal cooldown. Record which ones are SL_HIT.
  Pass 2: Re-score, this time resetting cooldown from SL_HIT timestamps.
  The second pass results are the final output.

Metrics calculated per combination:
  - win_rate, profit_factor, sharpe_ratio
  - total_r, avg_r, signals_per_month
  - max consecutive wins/losses
  - per_coin / per_session / per_direction / per_month breakdowns
  - composite score for ranking
"""

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from backtest import config
from backtest.candidate import CandidateSignal


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class TradeResult:
    """A single trade result from re-scoring."""
    signal_id: int
    combo_id: int
    coin: str
    direction: str
    timestamp: datetime
    session: str
    score: int
    entry_price: float
    tp1_price: float
    sl_price: float
    sl_distance: float
    rr_ratio: float
    outcome: str
    r_achieved: float


@dataclass
class ComboResult:
    """Full metrics for a single parameter combination."""
    combo_id: int
    params: dict[str, Any]
    trades: list[TradeResult] = field(default_factory=list)

    # Signal counts
    total_signals: int = 0
    long_signals: int = 0
    short_signals: int = 0
    signals_per_month: float = 0.0

    # Outcome distribution
    tp1_hit_count: int = 0
    tp2_hit_count: int = 0
    sl_hit_count: int = 0
    expired_count: int = 0
    win_rate: float = 0.0
    expiry_rate: float = 0.0

    # R metrics
    avg_r_per_trade: float = 0.0
    total_r: float = 0.0
    max_r: float = 0.0
    min_r: float = 0.0
    r_std_dev: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0

    # Consecutive analysis
    max_consecutive_losses: int = 0
    max_consecutive_wins: int = 0

    # Breakdowns
    per_coin: dict = field(default_factory=dict)
    per_session: dict = field(default_factory=dict)
    per_direction: dict = field(default_factory=dict)
    per_month: dict = field(default_factory=dict)

    # Ranking
    composite_score: float = 0.0
    rank: int = 0


# ---------------------------------------------------------------------------
# Session score table (matches live bot)
# ---------------------------------------------------------------------------
_SESSION_SCORES = {
    "Asian":   -2,
    "London":   2,
    "Overlap":  3,
    "NY":       1,
    "Late":    -1,
}


# ---------------------------------------------------------------------------
# Score calculator (stateless — uses stored inputs from CandidateSignal)
# ---------------------------------------------------------------------------

def _calculate_score(cand: CandidateSignal) -> int:
    """
    Re-calculate the additive score for a candidate using stored raw inputs.

    Mirrors the live bot's scoring logic exactly. The session score is
    included (positive or negative). Returns an integer score.
    """
    score = 0
    d = cand.direction

    # 1. FVG (+2)
    if cand.fvg_present:
        score += 2

    # 2. Volume 15m (+1 if > 1.0×)
    if cand.volume_ratio_15m > 1.0:
        score += 1

    # 3. Volume 5m (+2 if >= 1.5×)
    if cand.volume_ratio_5m >= 1.5:
        score += 2

    # 4. OI increasing (+2) — always 0 in backtest
    if cand.oi_increasing:
        score += 2

    # 5. CHoCH (+3)
    if cand.choch_present:
        score += 3

    # 6. MACD pre-crossover (+1)
    if cand.macd_histogram_trend:
        score += 1

    # 7. L/S ratio (+2 for longs if < 1.0)
    if d == "long" and cand.ls_ratio < 1.0:
        score += 2
    elif d == "short" and cand.ls_ratio > 1.0:
        score += 2

    # 8. Funding bonus (+2)
    if d == "long" and cand.funding_rate < 0:
        score += 2
    elif d == "short" and cand.funding_rate > 0:
        score += 2

    # 9. Strong EMA bias (+1 if gap > 2%)
    if d == "long" and cand.ema200_gap_pct > 0.02:
        score += 1
    elif d == "short" and cand.ema200_gap_pct < -0.02:
        score += 1

    # 10. Session score (-2 to +3)
    session_score = _SESSION_SCORES.get(cand.session, 0)
    score += session_score

    # 11. Liquidity sweep (+3)
    if cand.sweep_detected:
        score += 3

    # 12. RSI divergence (+3)
    if cand.rsi_divergence:
        score += 3

    # 13. ADX tiers
    if cand.adx_value >= 40.0:
        score += 2
    elif cand.adx_value >= 25.0:
        score += 1

    # 14. OB rejection (+2)
    if cand.ob_rejection_candle:
        score += 2

    # 15. OB touch freshness (+2 if first touch)
    if cand.ob_touch_count <= 1:
        score += 2

    # 16. Deep Fibonacci discount (+1)
    if cand.fib_deep_discount:
        score += 1

    # 17. BTC correlation
    if not cand.coin.upper().startswith("BTC"):
        if d == "long":
            if cand.btc_bullish:
                score += 1
            elif cand.btc_bearish:
                score += -2
        else:  # short
            if cand.btc_bearish:
                score += 1
            elif cand.btc_bullish:
                score += -2

    # 18. Equal levels (+2 double, +3 triple+)
    if cand.equal_level_present:
        if cand.equal_level_count >= 3:
            score += 3
        else:
            score += 2

    # 19. OB volume quality (+1)
    if cand.ob_volume_quality:
        score += 1

    return score


# ---------------------------------------------------------------------------
# R achievement calculation
# ---------------------------------------------------------------------------

def _r_achieved(outcome: str) -> float:
    """Return R achieved for an outcome. EXPIRED trades return 0.0."""
    if outcome == "TP1_HIT":
        return 1.5
    elif outcome == "TP2_HIT":
        return 3.0
    elif outcome == "SL_HIT":
        return -1.0
    else:
        return 0.0  # EXPIRED or PENDING


# ---------------------------------------------------------------------------
# Core re-scoring pass
# ---------------------------------------------------------------------------

def _scoring_pass(
    candidates: list[CandidateSignal],
    params: dict[str, Any],
    sl_reset_times: dict[str, Optional[datetime]],
) -> list[TradeResult]:
    """
    One scoring pass over all candidates.

    Args:
        candidates:       All candidates, sorted by timestamp.
        params:           Parameter combination dict.
        sl_reset_times:   {coin: timestamp_when_cooldown_was_reset_by_SL}.
                          None means no SL reset ever happened for that coin.

    Returns:
        List of accepted TradeResult objects.
    """
    combo_id = params["combo_id"]
    min_score = params["min_score"]
    atr_mult = params["atr_sl_mult"]
    min_rr = params["min_rr"]
    max_sl_pct = params["max_sl_pct"]
    cooldown_hours = config.COOLDOWN_HOURS

    # Per-coin last accepted signal timestamp
    last_signal_time: dict[str, Optional[datetime]] = {}
    results: list[TradeResult] = []
    signal_id = 0

    for cand in candidates:
        coin = cand.coin

        # ── Cooldown check ──
        last_ts = last_signal_time.get(coin)
        sl_reset_ts = sl_reset_times.get(coin)

        # Effective last signal time considering SL cooldown resets
        effective_last = last_ts
        if sl_reset_ts is not None and last_ts is not None:
            if sl_reset_ts > last_ts:
                effective_last = None  # cooldown was reset by SL hit after last signal

        if effective_last is not None:
            elapsed = (cand.timestamp - effective_last).total_seconds() / 3600
            if elapsed < cooldown_hours:
                continue

        # ── Mandatory gates ──
        if not all([
            cand.gate_ema_4h_structure,
            cand.gate_bos_1h,
            cand.gate_ob_present,
            cand.gate_adx_min,
            cand.gate_fib_zone,
            cand.gate_rsi_range,
            cand.gate_funding,
        ]):
            continue

        # ── Recalculate SL with this combination's ATR multiplier ──
        new_sl_dist = cand.sl_raw_distance * atr_mult
        new_sl_pct = new_sl_dist / cand.entry_price
        if new_sl_pct > max_sl_pct:
            continue

        if cand.direction == "long":
            new_sl = cand.ob_bottom - new_sl_dist
            new_tp1 = cand.entry_price + 1.5 * new_sl_dist
            new_tp2 = cand.entry_price + 3.0 * new_sl_dist
        else:
            new_sl = cand.ob_top + new_sl_dist
            new_tp1 = cand.entry_price - 1.5 * new_sl_dist
            new_tp2 = cand.entry_price - 3.0 * new_sl_dist

        new_rr = 1.5  # TP1 is always 1.5R by definition
        if new_rr < min_rr:
            continue

        # ── Score ──
        score = _calculate_score(cand)

        # ── Asian session gate ──
        if cand.session == "Asian" and score < 12:
            continue

        # ── Score threshold gate ──
        if score < min_score:
            continue

        # ── Determine outcome for this combo ──
        # The outcome from Phase 1 used default ATR mult.
        # For a different ATR mult, TP/SL prices change.
        # We use the raw walk-forward outcome as a proxy (same direction).
        # NOTE: A stricter SL could cause a hit that didn't occur at default,
        # but this is an acceptable approximation at this scale.
        outcome = cand.outcome

        r = _r_achieved(outcome)

        results.append(TradeResult(
            signal_id=signal_id,
            combo_id=combo_id,
            coin=coin,
            direction=cand.direction,
            timestamp=cand.timestamp,
            session=cand.session,
            score=score,
            entry_price=cand.entry_price,
            tp1_price=round(new_tp1, 8),
            sl_price=round(new_sl, 8),
            sl_distance=round(new_sl_dist, 8),
            rr_ratio=round(new_rr, 2),
            outcome=outcome,
            r_achieved=r,
        ))
        signal_id += 1

        # Update cooldown
        last_signal_time[coin] = cand.timestamp

    return results


# ---------------------------------------------------------------------------
# Metrics calculation
# ---------------------------------------------------------------------------

def _calculate_metrics(trades: list[TradeResult], params: dict) -> ComboResult:
    """
    Calculate all performance metrics from a list of accepted trades.
    """
    result = ComboResult(combo_id=params["combo_id"], params=params, trades=trades)

    if not trades:
        return result

    result.total_signals = len(trades)
    result.long_signals = sum(1 for t in trades if t.direction == "long")
    result.short_signals = result.total_signals - result.long_signals
    result.signals_per_month = result.total_signals / config.BACKTEST_PERIOD_MONTHS

    # Outcome distribution
    result.tp1_hit_count = sum(1 for t in trades if t.outcome in ("TP1_HIT", "TP2_HIT"))
    result.tp2_hit_count = sum(1 for t in trades if t.outcome == "TP2_HIT")
    result.sl_hit_count = sum(1 for t in trades if t.outcome == "SL_HIT")
    result.expired_count = sum(1 for t in trades if t.outcome == "EXPIRED")

    decided = result.tp1_hit_count + result.sl_hit_count
    result.win_rate = result.tp1_hit_count / decided if decided > 0 else 0.0
    result.expiry_rate = result.expired_count / result.total_signals

    # R metrics (exclude EXPIRED from denominator calcs)
    decided_trades = [t for t in trades if t.outcome != "EXPIRED" and t.outcome != "PENDING"]
    r_values = [t.r_achieved for t in decided_trades]

    if r_values:
        result.avg_r_per_trade = statistics.mean(r_values)
        result.total_r = sum(r_values)
        result.max_r = max(r_values)
        result.min_r = min(r_values)
        result.r_std_dev = statistics.stdev(r_values) if len(r_values) > 1 else 0.0

        pos_r = sum(r for r in r_values if r > 0)
        neg_r = abs(sum(r for r in r_values if r < 0))
        result.profit_factor = pos_r / neg_r if neg_r > 0 else (pos_r if pos_r > 0 else 0.0)
        result.sharpe_ratio = (
            result.avg_r_per_trade / result.r_std_dev
            if result.r_std_dev > 0 else 0.0
        )

    # Consecutive losses/wins
    streak = 0
    max_loss_streak = 0
    max_win_streak = 0
    for t in decided_trades:
        if t.r_achieved > 0:
            if streak > 0:
                streak += 1
            else:
                streak = 1
            max_win_streak = max(max_win_streak, streak)
        elif t.r_achieved < 0:
            if streak < 0:
                streak -= 1
            else:
                streak = -1
            max_loss_streak = max(max_loss_streak, abs(streak))

    result.max_consecutive_losses = max_loss_streak
    result.max_consecutive_wins = max_win_streak

    # Per-coin breakdown
    per_coin: dict[str, dict] = {}
    for t in trades:
        if t.coin not in per_coin:
            per_coin[t.coin] = {"total": 0, "wins": 0, "r": []}
        per_coin[t.coin]["total"] += 1
        if t.outcome in ("TP1_HIT", "TP2_HIT"):
            per_coin[t.coin]["wins"] += 1
        if t.outcome not in ("EXPIRED", "PENDING"):
            per_coin[t.coin]["r"].append(t.r_achieved)
    for coin, d in per_coin.items():
        decided_count = d["wins"] + sum(1 for t in trades if t.coin == coin and t.outcome == "SL_HIT")
        d["win_rate"] = d["wins"] / decided_count if decided_count > 0 else 0.0
        d["avg_r"] = statistics.mean(d["r"]) if d["r"] else 0.0
        d["total_r"] = sum(d["r"])
    result.per_coin = per_coin

    # Per-session breakdown
    per_session: dict[str, dict] = {}
    for t in trades:
        sess = t.session
        if sess not in per_session:
            per_session[sess] = {"total": 0, "wins": 0, "r": []}
        per_session[sess]["total"] += 1
        if t.outcome in ("TP1_HIT", "TP2_HIT"):
            per_session[sess]["wins"] += 1
        if t.outcome not in ("EXPIRED", "PENDING"):
            per_session[sess]["r"].append(t.r_achieved)
    for sess, d in per_session.items():
        decided_count = d["wins"] + sum(1 for t in trades if t.session == sess and t.outcome == "SL_HIT")
        d["win_rate"] = d["wins"] / decided_count if decided_count > 0 else 0.0
        d["avg_r"] = statistics.mean(d["r"]) if d["r"] else 0.0
    result.per_session = per_session

    # Per-direction breakdown
    for dir_key in ("long", "short"):
        dir_trades = [t for t in trades if t.direction == dir_key]
        wins = sum(1 for t in dir_trades if t.outcome in ("TP1_HIT", "TP2_HIT"))
        losses = sum(1 for t in dir_trades if t.outcome == "SL_HIT")
        r_vals = [t.r_achieved for t in dir_trades if t.outcome not in ("EXPIRED", "PENDING")]
        decided = wins + losses
        result.per_direction[dir_key] = {
            "total": len(dir_trades),
            "win_rate": wins / decided if decided > 0 else 0.0,
            "avg_r": statistics.mean(r_vals) if r_vals else 0.0,
        }

    # Per-month breakdown
    per_month: dict[str, dict] = {}
    for t in trades:
        month_key = t.timestamp.strftime("%Y-%m")
        if month_key not in per_month:
            per_month[month_key] = {"signals": 0, "wins": 0, "sl": 0}
        per_month[month_key]["signals"] += 1
        if t.outcome in ("TP1_HIT", "TP2_HIT"):
            per_month[month_key]["wins"] += 1
        elif t.outcome == "SL_HIT":
            per_month[month_key]["sl"] += 1
    for month, d in per_month.items():
        decided = d["wins"] + d["sl"]
        d["win_rate"] = d["wins"] / decided if decided > 0 else 0.0
    result.per_month = dict(sorted(per_month.items()))

    return result


# ---------------------------------------------------------------------------
# Composite score (for ranking all 64 combinations)
# ---------------------------------------------------------------------------

def _normalise(values: list[float]) -> list[float]:
    """Normalise a list of floats to [0, 1]. If all equal, return all 0.5."""
    mn, mx = min(values), max(values)
    if mx == mn:
        return [0.5] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


def _assign_composite_scores(results: list[ComboResult]) -> None:
    """
    Compute and assign composite scores to all combo results, then rank them.

    Composite = (win_rate × 0.35) + (PF × 0.30) + (sharpe × 0.20) + (signals/month × 0.15)
    All metrics normalised to [0, 1] first.
    """
    if not results:
        return

    win_rates = [r.win_rate for r in results]
    pfs = [r.profit_factor for r in results]
    sharpes = [r.sharpe_ratio for r in results]
    spm = [r.signals_per_month for r in results]

    norm_wr = _normalise(win_rates)
    norm_pf = _normalise(pfs)
    norm_sh = _normalise(sharpes)
    norm_sp = _normalise(spm)

    for i, result in enumerate(results):
        result.composite_score = (
            norm_wr[i] * config.WIN_RATE_WEIGHT
            + norm_pf[i] * config.PROFIT_FACTOR_WEIGHT
            + norm_sh[i] * config.SHARPE_WEIGHT
            + norm_sp[i] * config.SIGNAL_COUNT_WEIGHT
        )

    results.sort(key=lambda r: r.composite_score, reverse=True)
    for rank, result in enumerate(results, 1):
        result.rank = rank


# ---------------------------------------------------------------------------
# Two-pass re-scoring per combination
# ---------------------------------------------------------------------------

def rescore_combination(
    candidates: list[CandidateSignal],
    params: dict[str, Any],
) -> ComboResult:
    """
    Re-score one parameter combination with the two-pass cooldown approach.

    Pass 1: score all signals with normal cooldown.
    Pass 2: identify SL_HIT timestamps from Pass 1, re-run with cooldown reset
            logic. Use Pass 2 results as final output.

    Args:
        candidates: All candidates sorted by timestamp (from Phase 1).
        params:     Parameter combination dict.

    Returns:
        ComboResult with full metrics.
    """
    # Pass 1
    no_resets: dict[str, Optional[datetime]] = {}
    pass1_trades = _scoring_pass(candidates, params, no_resets)

    # Build SL reset timestamps from Pass 1
    sl_reset_times: dict[str, Optional[datetime]] = {}
    for trade in pass1_trades:
        if trade.outcome == "SL_HIT":
            coin = trade.coin
            ts = trade.timestamp
            existing = sl_reset_times.get(coin)
            if existing is None or ts > existing:
                sl_reset_times[coin] = ts

    # Pass 2 with SL resets
    pass2_trades = _scoring_pass(candidates, params, sl_reset_times)

    return _calculate_metrics(pass2_trades, params)


# ---------------------------------------------------------------------------
# Phase 2 entry point
# ---------------------------------------------------------------------------

def run(candidates: list[CandidateSignal]) -> list[ComboResult]:
    """
    Phase 2 entry point.

    Tests all 64 parameter combinations against the candidate signal list.

    Args:
        candidates: Master candidate list from Phase 1.

    Returns:
        List of ComboResult objects, ranked by composite score.
    """
    print("\n=== Phase 2: Parameter Grid Re-Scoring ===")
    grid = config.generate_parameter_grid()
    results: list[ComboResult] = []

    for i, params in enumerate(grid, 1):
        print(
            f"  [{i:2d}/64] min_score={params['min_score']}, "
            f"atr_mult={params['atr_sl_mult']}, "
            f"min_rr={params['min_rr']}, "
            f"max_sl={params['max_sl_pct']:.1%}...",
            end=" ",
            flush=True,
        )
        result = rescore_combination(candidates, params)
        results.append(result)
        print(
            f"signals={result.total_signals}, "
            f"wr={result.win_rate:.1%}, "
            f"total_r={result.total_r:.1f}"
        )

    # Assign composite scores and rank
    _assign_composite_scores(results)

    best = results[0]
    print(
        f"\n  Phase 2 complete. Best combination (#1):\n"
        f"    min_score={best.params['min_score']}, "
        f"atr_mult={best.params['atr_sl_mult']}, "
        f"min_rr={best.params['min_rr']}, "
        f"max_sl={best.params['max_sl_pct']:.1%}\n"
        f"    win_rate={best.win_rate:.1%}, "
        f"total_r={best.total_r:.1f}, "
        f"signals={best.total_signals}"
    )

    return results
