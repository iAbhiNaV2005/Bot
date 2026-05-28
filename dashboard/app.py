"""
dashboard/app.py — Signal tracking dashboard.

A clean, standalone Flask web app that reads from the SQLite signal database.
Run separately from the main bot: python -m dashboard.app

Features:
- Live stats cards (total signals, active, TP hits, SL hits, win rate)
- Active signals table with current price, P&L, and status
- Signal history table with all past signals
- Auto-refresh every 30 seconds
- Dark theme matching the trading aesthetic
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, render_template_string
from database.signal_log import signal_logger

app = Flask(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# HTML TEMPLATE — Single-page dark dashboard
# ──────────────────────────────────────────────────────────────────────────────
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Signal Dashboard — SMC Bot</title>
<meta name="description" content="Crypto futures signal tracking dashboard">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-primary: #0a0e17;
    --bg-card: #111827;
    --bg-card-hover: #1a2332;
    --bg-table-row: #0d1320;
    --bg-table-row-alt: #111827;
    --border: #1e293b;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --green: #22c55e;
    --green-dim: rgba(34, 197, 94, 0.15);
    --red: #ef4444;
    --red-dim: rgba(239, 68, 68, 0.15);
    --blue: #3b82f6;
    --blue-dim: rgba(59, 130, 246, 0.15);
    --amber: #f59e0b;
    --amber-dim: rgba(245, 158, 11, 0.15);
    --purple: #a855f7;
    --purple-dim: rgba(168, 85, 247, 0.15);
    --cyan: #06b6d4;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
    line-height: 1.5;
  }

  /* ── Header ── */
  .header {
    padding: 20px 32px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(180deg, #111827 0%, var(--bg-primary) 100%);
  }
  .header h1 {
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: -0.02em;
  }
  .header h1 span { color: var(--cyan); }
  .header-meta {
    display: flex;
    align-items: center;
    gap: 16px;
    font-size: 0.8rem;
    color: var(--text-muted);
  }
  .pulse {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse-anim 2s infinite;
  }
  @keyframes pulse-anim {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(34,197,94,0.5); }
    50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(34,197,94,0); }
  }

  /* ── Stats Grid ── */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    padding: 24px 32px;
  }
  .stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    transition: border-color 0.2s, transform 0.2s;
  }
  .stat-card:hover {
    border-color: #334155;
    transform: translateY(-1px);
  }
  .stat-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
  }
  .stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1;
  }
  .stat-sub {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 6px;
  }

  /* ── Section ── */
  .section {
    padding: 0 32px 24px;
  }
  .section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
    padding-bottom: 8px;
  }
  .section-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
  }
  .section-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 500;
  }

  /* ── Table ── */
  .table-wrap {
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--bg-card);
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
  }
  thead th {
    text-align: left;
    padding: 12px 14px;
    font-weight: 600;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
    position: sticky;
    top: 0;
    background: var(--bg-card);
  }
  tbody tr {
    border-bottom: 1px solid var(--border);
    transition: background 0.15s;
  }
  tbody tr:last-child { border-bottom: none; }
  tbody tr:hover { background: var(--bg-card-hover); }
  td {
    padding: 10px 14px;
    white-space: nowrap;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
  }
  .empty-row td {
    text-align: center;
    color: var(--text-muted);
    padding: 40px;
    font-family: 'Inter', sans-serif;
    font-style: italic;
  }

  /* ── Badges ── */
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }
  .badge-long { background: var(--green-dim); color: var(--green); }
  .badge-short { background: var(--red-dim); color: var(--red); }
  .badge-active { background: var(--blue-dim); color: var(--blue); }
  .badge-tp1 { background: var(--green-dim); color: var(--green); }
  .badge-tp2 { background: var(--green-dim); color: #4ade80; }
  .badge-tp3 { background: var(--purple-dim); color: var(--purple); }
  .badge-sl { background: var(--red-dim); color: var(--red); }
  .badge-expired { background: var(--amber-dim); color: var(--amber); }
  .badge-high { background: var(--amber-dim); color: var(--amber); }
  .badge-standard { background: #1e293b; color: var(--text-secondary); }

  .pnl-positive { color: var(--green); }
  .pnl-negative { color: var(--red); }
  .pnl-zero { color: var(--text-muted); }

  /* ── Tab Nav ── */
  .tabs {
    display: flex;
    gap: 4px;
    padding: 0 32px;
    margin-bottom: 16px;
  }
  .tab-btn {
    padding: 8px 20px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: transparent;
    color: var(--text-secondary);
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
  }
  .tab-btn:hover { background: var(--bg-card); color: var(--text-primary); }
  .tab-btn.active {
    background: var(--blue-dim);
    color: var(--blue);
    border-color: rgba(59,130,246,0.3);
  }
  .tab-content { display: none; }
  .tab-content.active { display: block; }

  /* ── Refresh bar ── */
  .refresh-bar {
    padding: 8px 32px;
    font-size: 0.72rem;
    color: var(--text-muted);
    text-align: right;
  }

  /* ── Responsive ── */
  @media (max-width: 768px) {
    .header, .stats-grid, .section, .tabs, .refresh-bar { padding-left: 16px; padding-right: 16px; }
    .stats-grid { grid-template-columns: repeat(2, 1fr); gap: 10px; }
    .stat-value { font-size: 1.3rem; }
  }
</style>
</head>
<body>

<div class="header">
  <h1>📊 <span>SMC</span> Signal Dashboard</h1>
  <div class="header-meta">
    <div class="pulse"></div>
    <span id="bot-status">Auto-refresh: 30s</span>
  </div>
</div>

<!-- Stats Cards -->
<div class="stats-grid" id="stats-grid">
  <div class="stat-card">
    <div class="stat-label">Total Signals</div>
    <div class="stat-value" id="stat-total">—</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Active Now</div>
    <div class="stat-value" id="stat-active" style="color:var(--blue)">—</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">TP Hits</div>
    <div class="stat-value" id="stat-tp" style="color:var(--green)">—</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">SL Hits</div>
    <div class="stat-value" id="stat-sl" style="color:var(--red)">—</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Win Rate</div>
    <div class="stat-value" id="stat-winrate">—</div>
    <div class="stat-sub" id="stat-winrate-sub"></div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Avg P&L</div>
    <div class="stat-value" id="stat-avgpnl">—</div>
  </div>
</div>

<!-- Tabs -->
<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('active')" id="tab-active-btn">Active Signals</button>
  <button class="tab-btn" onclick="switchTab('history')" id="tab-history-btn">Signal History</button>
</div>

<!-- Active Signals -->
<div class="section tab-content active" id="tab-active">
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Symbol</th>
          <th>Direction</th>
          <th>Entry</th>
          <th>Current</th>
          <th>P&L</th>
          <th>SL</th>
          <th>TP1</th>
          <th>TP2</th>
          <th>TP3</th>
          <th>Score</th>
          <th>Conf.</th>
          <th>Time</th>
        </tr>
      </thead>
      <tbody id="active-body">
        <tr class="empty-row"><td colspan="13">No active signals yet</td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- History -->
<div class="section tab-content" id="tab-history">
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Symbol</th>
          <th>Dir</th>
          <th>Entry</th>
          <th>SL</th>
          <th>TP1</th>
          <th>TP2</th>
          <th>TP3</th>
          <th>Status</th>
          <th>P&L</th>
          <th>Score</th>
          <th>Conf.</th>
          <th>Opened</th>
          <th>Closed</th>
        </tr>
      </thead>
      <tbody id="history-body">
        <tr class="empty-row"><td colspan="14">No signals recorded yet</td></tr>
      </tbody>
    </table>
  </div>
</div>

<div class="refresh-bar">
  Last updated: <span id="last-updated">—</span>
</div>

<script>
function fmt(price) {
  if (price == null) return '—';
  const p = parseFloat(price);
  if (p >= 100) return '$' + p.toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2});
  if (p >= 1) return '$' + p.toFixed(4);
  return '$' + p.toFixed(6);
}

function pnlClass(pnl) {
  if (pnl > 0) return 'pnl-positive';
  if (pnl < 0) return 'pnl-negative';
  return 'pnl-zero';
}

function dirBadge(dir) {
  return dir === 'LONG'
    ? '<span class="badge badge-long">LONG</span>'
    : '<span class="badge badge-short">SHORT</span>';
}

function statusBadge(status) {
  const map = {
    'ACTIVE': 'badge-active',
    'TP1_HIT': 'badge-tp1',
    'TP2_HIT': 'badge-tp2',
    'TP3_HIT': 'badge-tp3',
    'SL_HIT': 'badge-sl',
    'EXPIRED': 'badge-expired',
  };
  const cls = map[status] || 'badge-active';
  const label = status.replace('_', ' ');
  return `<span class="badge ${cls}">${label}</span>`;
}

function confBadge(c) {
  return c === 'HIGH'
    ? '<span class="badge badge-high">🔥 HIGH</span>'
    : '<span class="badge badge-standard">STD</span>';
}

function timeAgo(ts) {
  if (!ts) return '—';
  const d = new Date(ts);
  const now = new Date();
  const diff = Math.floor((now - d) / 1000);
  if (diff < 60) return diff + 's ago';
  if (diff < 3600) return Math.floor(diff/60) + 'm ago';
  if (diff < 86400) return Math.floor(diff/3600) + 'h ago';
  return Math.floor(diff/86400) + 'd ago';
}

function shortTime(ts) {
  if (!ts) return '—';
  const d = new Date(ts);
  return d.toLocaleString('en-GB', {month:'short', day:'numeric', hour:'2-digit', minute:'2-digit', hour12:false});
}

function switchTab(tab) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + tab).classList.add('active');
  document.getElementById('tab-' + tab + '-btn').classList.add('active');
}

async function fetchData() {
  try {
    const [statsRes, activeRes, historyRes] = await Promise.all([
      fetch('/api/stats'),
      fetch('/api/signals/active'),
      fetch('/api/signals/all'),
    ]);

    const stats = await statsRes.json();
    const active = await activeRes.json();
    const history = await historyRes.json();

    // Update stats
    document.getElementById('stat-total').textContent = stats.total || 0;
    document.getElementById('stat-active').textContent = stats.active || 0;
    document.getElementById('stat-tp').textContent = stats.tp_hits || 0;
    document.getElementById('stat-sl').textContent = stats.sl_hits || 0;

    const closed = (stats.tp_hits || 0) + (stats.sl_hits || 0);
    const wr = closed > 0 ? ((stats.tp_hits / closed) * 100).toFixed(0) : '—';
    document.getElementById('stat-winrate').textContent = wr === '—' ? wr : wr + '%';
    document.getElementById('stat-winrate').style.color = wr === '—' ? '' : (parseFloat(wr) >= 50 ? 'var(--green)' : 'var(--red)');
    document.getElementById('stat-winrate-sub').textContent = closed > 0 ? `${stats.tp_hits}W / ${stats.sl_hits}L` : '';

    const avgPnl = stats.avg_pnl;
    if (avgPnl != null) {
      document.getElementById('stat-avgpnl').textContent = (avgPnl >= 0 ? '+' : '') + avgPnl.toFixed(2) + '%';
      document.getElementById('stat-avgpnl').style.color = avgPnl >= 0 ? 'var(--green)' : 'var(--red)';
    } else {
      document.getElementById('stat-avgpnl').textContent = '—';
    }

    // Update active table
    const activeBody = document.getElementById('active-body');
    if (active.length === 0) {
      activeBody.innerHTML = '<tr class="empty-row"><td colspan="13">No active signals — waiting for next cycle</td></tr>';
    } else {
      activeBody.innerHTML = active.map(s => `
        <tr>
          <td>${s.id}</td>
          <td style="font-weight:600">${s.symbol.replace('/USDT:USDT','')}</td>
          <td>${dirBadge(s.direction)}</td>
          <td>${fmt(s.entry_price)}</td>
          <td>${fmt(s.current_price)}</td>
          <td class="${pnlClass(s.pnl_pct)}">${s.pnl_pct != null ? (s.pnl_pct >= 0 ? '+' : '') + s.pnl_pct.toFixed(2) + '%' : '—'}</td>
          <td style="color:var(--red)">${fmt(s.stop_loss)}</td>
          <td style="color:var(--green)">${fmt(s.tp1)}</td>
          <td style="color:var(--green)">${fmt(s.tp2)}</td>
          <td style="color:${s.tp3 ? 'var(--purple)' : 'var(--text-muted)'}">${fmt(s.tp3)}</td>
          <td>${s.score}/${s.max_score}</td>
          <td>${confBadge(s.confidence)}</td>
          <td style="color:var(--text-muted)">${timeAgo(s.timestamp)}</td>
        </tr>
      `).join('');
    }

    // Update active count badge on tab
    document.getElementById('tab-active-btn').textContent = `Active Signals (${active.length})`;

    // Update history table
    const histBody = document.getElementById('history-body');
    if (history.length === 0) {
      histBody.innerHTML = '<tr class="empty-row"><td colspan="14">No signals recorded yet</td></tr>';
    } else {
      histBody.innerHTML = history.map(s => `
        <tr>
          <td>${s.id}</td>
          <td style="font-weight:600">${s.symbol.replace('/USDT:USDT','')}</td>
          <td>${dirBadge(s.direction)}</td>
          <td>${fmt(s.entry_price)}</td>
          <td style="color:var(--red)">${fmt(s.stop_loss)}</td>
          <td style="color:var(--green)">${fmt(s.tp1)}</td>
          <td style="color:var(--green)">${fmt(s.tp2)}</td>
          <td>${fmt(s.tp3)}</td>
          <td>${statusBadge(s.status || 'ACTIVE')}</td>
          <td class="${pnlClass(s.pnl_pct)}">${s.pnl_pct != null ? (s.pnl_pct >= 0 ? '+' : '') + s.pnl_pct.toFixed(2) + '%' : '—'}</td>
          <td>${s.score}/${s.max_score}</td>
          <td>${confBadge(s.confidence)}</td>
          <td style="color:var(--text-muted)">${shortTime(s.timestamp)}</td>
          <td style="color:var(--text-muted)">${s.closed_at ? shortTime(s.closed_at) : '—'}</td>
        </tr>
      `).join('');
    }

    document.getElementById('last-updated').textContent = new Date().toLocaleTimeString();
  } catch (err) {
    console.error('Fetch error:', err);
  }
}

// Initial load + auto-refresh every 30s
fetchData();
setInterval(fetchData, 30000);
</script>
</body>
</html>
"""


# ──────────────────────────────────────────────────────────────────────────────
# API Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the dashboard page."""
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/stats")
def api_stats():
    """Get overall signal statistics."""
    return jsonify(signal_logger.get_stats())


@app.route("/api/signals/active")
def api_active():
    """Get all active signals."""
    return jsonify(signal_logger.get_active_signals())


@app.route("/api/signals/all")
def api_all():
    """Get all signals (last 100)."""
    return jsonify(signal_logger.get_all_signals(limit=100))


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  SMC Signal Dashboard")
    print("  http://localhost:5050")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5050, debug=False)
