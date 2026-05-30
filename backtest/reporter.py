"""
backtest/reporter.py — Phase 3: HTML report generation.

Produces a fully self-contained HTML file (no external CSS/JS/CDN deps).
All styles are inlined. The file renders correctly when opened from disk.

Report sections:
  1. Summary Banner — best combo, key stats, warnings
  2. Top 10 Combinations Table
  3. Best Combo Deep Dive (per-coin, per-session, per-month, per-direction)
  4. Parameter Sensitivity Analysis
  5. Warnings and Caveats
"""

import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backtest import config
from backtest.rescorer import ComboResult


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0d1117;
    color: #c9d1d9;
    line-height: 1.6;
    padding: 24px;
}
h1 { color: #58a6ff; font-size: 2rem; margin-bottom: 8px; }
h2 { color: #79c0ff; font-size: 1.3rem; margin: 28px 0 12px; border-bottom: 1px solid #21262d; padding-bottom: 6px; }
h3 { color: #d2a8ff; font-size: 1rem; margin: 16px 0 8px; }
p  { color: #8b949e; margin-bottom: 10px; }
a  { color: #58a6ff; }
.subtitle { color: #8b949e; font-size: 0.9rem; margin-bottom: 24px; }

/* Cards */
.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 16px;
}
.banner {
    background: linear-gradient(135deg, #1a2a4a 0%, #0d1f3c 100%);
    border: 1px solid #58a6ff44;
    border-radius: 10px;
    padding: 24px;
    margin-bottom: 24px;
}
.banner-title { color: #58a6ff; font-size: 1.5rem; font-weight: 700; margin-bottom: 12px; }
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 12px;
    margin-top: 16px;
}
.stat-item {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 12px;
    text-align: center;
}
.stat-value { font-size: 1.4rem; font-weight: 700; color: #58a6ff; }
.stat-label { font-size: 0.75rem; color: #8b949e; margin-top: 2px; }

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
}
th {
    background: #21262d;
    color: #8b949e;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
    white-space: nowrap;
}
td {
    padding: 7px 12px;
    border-bottom: 1px solid #21262d;
    color: #c9d1d9;
}
tr:hover td { background: #1c2128; }
.best-row td { background: #0c2a1c !important; color: #56d364; }
.best-row:hover td { background: #0f3320 !important; }
.rank-badge {
    display: inline-block;
    background: #1f6feb;
    color: white;
    border-radius: 12px;
    padding: 1px 8px;
    font-size: 0.75rem;
    font-weight: 700;
}
.best-badge {
    background: linear-gradient(90deg, #1a7f37, #2ea043);
}
.win { color: #56d364; }
.loss { color: #f85149; }
.neutral { color: #8b949e; }
.positive { color: #56d364; }
.negative { color: #f85149; }

/* Warnings */
.warn {
    background: #2d1f00;
    border-left: 4px solid #d29922;
    border-radius: 4px;
    padding: 12px 16px;
    margin-bottom: 10px;
    color: #e3b341;
    font-size: 0.875rem;
}
.warn-icon { margin-right: 6px; }
.info {
    background: #0c2d6b;
    border-left: 4px solid #388bfd;
    border-radius: 4px;
    padding: 12px 16px;
    margin-bottom: 10px;
    color: #79c0ff;
    font-size: 0.875rem;
}

/* Sensitivity table highlight */
.best-param td { background: #12261e !important; }

footer { margin-top: 40px; color: #484f58; font-size: 0.8rem; text-align: center; }
"""


def _pct(v: float, decimals: int = 1) -> str:
    return f"{v * 100:.{decimals}f}%"


def _r(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}R"


def _fmt(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}"


def _color_r(v: float) -> str:
    cls = "positive" if v >= 0 else "negative"
    return f'<span class="{cls}">{_r(v)}</span>'


def _color_wr(v: float) -> str:
    cls = "win" if v >= 0.5 else ("neutral" if v >= 0.4 else "loss")
    return f'<span class="{cls}">{_pct(v)}</span>'


def _build_combo_row(result: ComboResult, is_best: bool) -> str:
    p = result.params
    row_cls = 'class="best-row"' if is_best else ""
    badge = '<span class="rank-badge best-badge">★ BEST</span>' if is_best else f'<span class="rank-badge">#{result.rank}</span>'
    return f"""
    <tr {row_cls}>
        <td>{badge}</td>
        <td>{p["min_score"]}</td>
        <td>{p["atr_sl_mult"]}×</td>
        <td>{p["min_rr"]}</td>
        <td>{_pct(p["max_sl_pct"])}</td>
        <td>{result.signals_per_month:.1f}/mo</td>
        <td>{_color_wr(result.win_rate)}</td>
        <td>{_fmt(result.profit_factor)}</td>
        <td>{_fmt(result.sharpe_ratio)}</td>
        <td>{_color_r(result.total_r)}</td>
        <td><strong>{_fmt(result.composite_score, 3)}</strong></td>
    </tr>"""


def _table_header(cols: list[str]) -> str:
    ths = "".join(f"<th>{c}</th>" for c in cols)
    return f"<thead><tr>{ths}</tr></thead>"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _section_banner(best: ComboResult) -> str:
    p = best.params
    return f"""
    <div class="banner">
        <div class="banner-title">🏆 Best Parameter Combination</div>
        <p>
            Min Score ≥ <strong>{p["min_score"]}</strong> &nbsp;|&nbsp;
            ATR Mult <strong>{p["atr_sl_mult"]}×</strong> &nbsp;|&nbsp;
            Min R:R <strong>{p["min_rr"]}</strong> &nbsp;|&nbsp;
            Max SL <strong>{_pct(p["max_sl_pct"])}</strong>
        </p>
        <div class="stat-grid">
            <div class="stat-item">
                <div class="stat-value win">{_pct(best.win_rate)}</div>
                <div class="stat-label">Win Rate</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{_fmt(best.profit_factor)}</div>
                <div class="stat-label">Profit Factor</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{_fmt(best.sharpe_ratio)}</div>
                <div class="stat-label">Sharpe Ratio</div>
            </div>
            <div class="stat-item">
                <div class="stat-value {"positive" if best.total_r >= 0 else "negative"}">{_r(best.total_r)}</div>
                <div class="stat-label">Total R</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{best.total_signals}</div>
                <div class="stat-label">Total Signals</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{best.signals_per_month:.1f}</div>
                <div class="stat-label">Signals / Month</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{best.max_consecutive_losses}</div>
                <div class="stat-label">Max Consec. Losses</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{_pct(best.expiry_rate)}</div>
                <div class="stat-label">Expiry Rate</div>
            </div>
        </div>
    </div>"""


def _section_top10(results: list[ComboResult]) -> str:
    top10 = results[:10]
    rows = "".join(_build_combo_row(r, r.rank == 1) for r in top10)
    header = _table_header([
        "Rank", "Min Score", "ATR Mult", "Min R:R", "Max SL",
        "Signals/mo", "Win Rate", "PF", "Sharpe", "Total R", "Composite"
    ])
    return f"""
    <h2>📊 Top 10 Parameter Combinations</h2>
    <div class="card">
        <table>
            {header}
            <tbody>{rows}</tbody>
        </table>
    </div>"""


def _section_deep_dive(best: ComboResult) -> str:
    # Per-coin table
    coin_rows = ""
    for coin, d in sorted(best.per_coin.items(), key=lambda x: -x[1].get("total_r", 0)):
        coin_rows += f"""<tr>
            <td>{coin}</td>
            <td>{d["total"]}</td>
            <td>{_color_wr(d["win_rate"])}</td>
            <td>{_color_r(d["avg_r"])}</td>
            <td>{_color_r(d["total_r"])}</td>
        </tr>"""

    # Per-session table
    sess_rows = ""
    for sess in ["Overlap", "London", "NY", "Asian", "Late"]:
        d = best.per_session.get(sess)
        if d:
            sess_rows += f"""<tr>
                <td>{sess}</td>
                <td>{d["total"]}</td>
                <td>{_color_wr(d["win_rate"])}</td>
                <td>{_color_r(d["avg_r"])}</td>
            </tr>"""

    # Per-month table
    month_rows = ""
    for month, d in best.per_month.items():
        month_rows += f"""<tr>
            <td>{month}</td>
            <td>{d["signals"]}</td>
            <td class="win">{d["wins"]}</td>
            <td class="loss">{d["sl"]}</td>
            <td>{_color_wr(d["win_rate"])}</td>
        </tr>"""

    # Per-direction table
    dir_rows = ""
    for dir_key in ["long", "short"]:
        d = best.per_direction.get(dir_key, {})
        if d:
            dir_rows += f"""<tr>
                <td>{"🟢 Long" if dir_key == "long" else "🔴 Short"}</td>
                <td>{d["total"]}</td>
                <td>{_color_wr(d["win_rate"])}</td>
                <td>{_color_r(d["avg_r"])}</td>
            </tr>"""

    return f"""
    <h2>🔬 Best Combination Deep Dive</h2>
    <div class="card">
        <h3>Performance by Coin</h3>
        <table>
            {_table_header(["Coin", "Signals", "Win Rate", "Avg R", "Total R"])}
            <tbody>{coin_rows}</tbody>
        </table>
    </div>
    <div class="card">
        <h3>Performance by Session</h3>
        <table>
            {_table_header(["Session", "Signals", "Win Rate", "Avg R"])}
            <tbody>{sess_rows}</tbody>
        </table>
    </div>
    <div class="card">
        <h3>Monthly Performance</h3>
        <table>
            {_table_header(["Month", "Signals", "TP1 Hit", "SL Hit", "Win Rate"])}
            <tbody>{month_rows}</tbody>
        </table>
    </div>
    <div class="card">
        <h3>Long vs Short</h3>
        <table>
            {_table_header(["Direction", "Signals", "Win Rate", "Avg R"])}
            <tbody>{dir_rows}</tbody>
        </table>
    </div>"""


def _section_sensitivity(results: list[ComboResult]) -> str:
    """Show how win_rate, signals/month, and total R vary with each parameter."""
    best = results[0]
    best_params = best.params

    def _make_sensitivity_table(
        param_name: str,
        param_values: list,
        fixed_params: dict,
        results: list[ComboResult],
    ) -> str:
        rows = ""
        for val in param_values:
            # Find matching combos (other params at best value)
            matching = [
                r for r in results
                if r.params[param_name] == val
                and all(r.params[k] == v for k, v in fixed_params.items() if k != param_name)
            ]
            if not matching:
                continue
            # Average across matches (there may be multiple if >1 other param varies)
            avg_wr = sum(r.win_rate for r in matching) / len(matching)
            avg_spm = sum(r.signals_per_month for r in matching) / len(matching)
            avg_tr = sum(r.total_r for r in matching) / len(matching)
            is_best = val == best_params[param_name]
            cls = ' class="best-param"' if is_best else ""
            rows += f"""<tr{cls}>
                <td>{"★ " if is_best else ""}{val}</td>
                <td>{_color_wr(avg_wr)}</td>
                <td>{avg_spm:.1f}</td>
                <td>{_color_r(avg_tr)}</td>
            </tr>"""
        return rows

    fixed = best_params.copy()

    score_rows = _make_sensitivity_table(
        "min_score", config.MIN_SCORE_OPTIONS, fixed, results
    )
    atr_rows = _make_sensitivity_table(
        "atr_sl_mult", config.ATR_SL_MULT_OPTIONS, fixed, results
    )
    rr_rows = _make_sensitivity_table(
        "min_rr", config.MIN_RR_OPTIONS, fixed, results
    )
    sl_rows = _make_sensitivity_table(
        "max_sl_pct", config.MAX_SL_PCT_OPTIONS, fixed, results
    )

    hdr = _table_header(["Value", "Win Rate", "Signals/mo", "Total R"])

    return f"""
    <h2>🔧 Parameter Sensitivity Analysis</h2>
    <p>Other parameters held at best-combo values. ★ = best combination value.</p>
    <div class="card">
        <h3>Min Score Sensitivity</h3>
        <table>{hdr}<tbody>{score_rows}</tbody></table>
    </div>
    <div class="card">
        <h3>ATR Multiplier Sensitivity</h3>
        <table>{hdr}<tbody>{atr_rows}</tbody></table>
    </div>
    <div class="card">
        <h3>Min R:R Sensitivity</h3>
        <table>{hdr}<tbody>{rr_rows}</tbody></table>
    </div>
    <div class="card">
        <h3>Max SL% Sensitivity</h3>
        <table>{hdr}<tbody>{sl_rows}</tbody></table>
    </div>"""


def _section_warnings(results: list[ComboResult]) -> str:
    best = results[0]
    warnings: list[str] = []

    # Low signal count
    if best.total_signals < config.MIN_SIGNALS_FOR_VALIDITY:
        warnings.append(
            f"<strong>LOW SIGNAL COUNT:</strong> Results based on {best.total_signals} trades. "
            f"Statistical significance requires at least {config.MIN_SIGNALS_FOR_VALIDITY} trades. "
            f"Treat these results as directional only."
        )

    # OI filter always disabled
    warnings.append(
        "<strong>OI FILTER DISABLED:</strong> Open Interest filter was disabled in backtest "
        "due to data unavailability. Live results may differ."
    )

    # Overfitting risk
    if len(results) >= 3:
        best_score = results[0].composite_score
        third_score = results[2].composite_score
        if third_score > 0 and (best_score - third_score) / third_score > config.OVERFITTING_THRESHOLD:
            warnings.append(
                f"<strong>OVERFITTING RISK:</strong> Best parameters significantly outperform "
                f"alternatives (best: {best_score:.3f} vs 3rd: {third_score:.3f}). "
                f"This may indicate overfitting. Consider using the 3rd-ranked combination "
                f"for live trading."
            )

    # High expiry rate
    if best.expiry_rate > config.MAX_EXPIRY_RATE_WARN:
        warnings.append(
            f"<strong>HIGH EXPIRY RATE:</strong> {_pct(best.expiry_rate)} of signals expired "
            f"without hitting TP or SL. Consider widening TP targets or extending the "
            f"outcome window."
        )

    warn_html = "".join(
        f'<div class="warn"><span class="warn-icon">⚠️</span>{w}</div>'
        for w in warnings
    )
    return f"""
    <h2>⚠️ Warnings &amp; Caveats</h2>
    <div class="info">
        ℹ️ Backtest period: {config.BACKTEST_PERIOD_MONTHS} months.
        Top {config.TOP_N_COINS} coins by 24h volume.
        Warmup: {config.WARMUP_4H_CANDLES} × 4H candles.
        Outcome window: {config.OUTCOME_WINDOW_CANDLES} × 5m candles (24h).
    </div>
    {warn_html}"""


# ---------------------------------------------------------------------------
# Full report assembly
# ---------------------------------------------------------------------------

def generate(results: list[ComboResult]) -> str:
    """
    Generate the full self-contained HTML report.

    Args:
        results: All combo results, sorted by rank (rank 1 = best).

    Returns:
        HTML string.
    """
    if not results:
        return "<html><body><p>No results to display.</p></body></html>"

    best = results[0]
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    banner = _section_banner(best)
    top10 = _section_top10(results)
    deep_dive = _section_deep_dive(best)
    sensitivity = _section_sensitivity(results)
    warnings = _section_warnings(results)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SMC Bot Backtest Report — {generated_at}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>📈 SMC Signal Bot — Backtest Report</h1>
<p class="subtitle">Generated {generated_at} &nbsp;|&nbsp; {config.BACKTEST_PERIOD_MONTHS}-month period &nbsp;|&nbsp; Top {config.TOP_N_COINS} coins &nbsp;|&nbsp; {len(results)} combinations tested</p>

{banner}
{top10}
{deep_dive}
{sensitivity}
{warnings}

<footer>SMC Momentum Confluence Signal Bot — Backtest Engine v1.0<br>
Results are historical simulations only. Past performance does not guarantee future results.</footer>
</body>
</html>"""


def run(results: list[ComboResult]) -> None:
    """
    Phase 3 entry point.

    Generates the HTML report, saves it to disk, and opens it in the browser.

    Args:
        results: All combo results from Phase 2.
    """
    print("\n=== Phase 3: Report Generation ===")

    out_path = Path(config.HTML_REPORT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html = generate(results)
    out_path.write_text(html, encoding="utf-8")
    print(f"  Report saved to {out_path}")

    try:
        webbrowser.open(str(out_path))
        print("  Opening report in browser...")
    except Exception:
        pass
