"""report — render audit results to standalone HTML.

Self-contained — no Jinja, no JS dependencies. Produces a single .html file
that can be emailed, committed, or opened in any browser.

Inputs supported:
    - GapReport (from compare.py)
    - SurvivabilityResult or SweepResult (from regime_runner.py)
    - ParsedLog (from log_parser.py — for live-only views)
"""

from __future__ import annotations

import html
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from backtest_audit.compare import GapReport, TradePair
from backtest_audit.log_parser import ParsedLog
from backtest_audit.regime_runner import (
    SurvivabilityResult, SweepResult, WindowResult,
)


# ─── HTML helpers ────────────────────────────────────────────────────────────

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1200px; margin: 1.5em auto; color: #222; }
h1 { border-bottom: 3px solid #333; padding-bottom: 8px; }
h2 { color: #2c4e7a; margin-top: 2em; }
h3 { color: #555; }
.section { background: #f7f7f9; padding: 1em 1.5em; border-radius: 6px;
           margin: 1em 0; border-left: 4px solid #2c4e7a; }
.kv { display: grid; grid-template-columns: 280px 1fr; gap: 4px 12px; }
.kv .k { color: #666; }
.kv .v { font-family: 'SF Mono', Menlo, monospace; }
.bad { color: #c0392b; font-weight: bold; }
.good { color: #27ae60; font-weight: bold; }
.warn { color: #c87f0a; font-weight: bold; }
table { border-collapse: collapse; width: 100%; margin: 0.5em 0;
        font-family: 'SF Mono', Menlo, monospace; font-size: 13px; }
th, td { border: 1px solid #ddd; padding: 5px 8px; text-align: left; }
th { background: #eef; }
tr:nth-child(even) { background: #fafafc; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 4px;
       font-size: 11px; font-weight: bold; }
.tag-blessed { background: #27ae60; color: white; }
.tag-rejected { background: #c0392b; color: white; }
.tag-warn { background: #c87f0a; color: white; }
.footer { color: #999; font-size: 11px; margin-top: 3em; text-align: right; }
"""


def _esc(x: Any) -> str:
    return html.escape(str(x))


def _fmt(v: Any, fmt: str = "") -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if fmt:
            return f"{v:{fmt}}"
        return f"{v:.4f}"
    return _esc(v)


def _bps_color(bps: float | None, threshold: float = 50) -> str:
    if bps is None:
        return "—"
    cls = "bad" if abs(bps) > threshold else ""
    return f'<span class="{cls}">{bps:+.1f}bp</span>'


def _pct_color(pct: float | None, neutral_band: float = 1.0) -> str:
    if pct is None:
        return "—"
    if pct < -neutral_band:
        cls = "bad"
    elif pct > neutral_band:
        cls = "good"
    else:
        cls = ""
    return f'<span class="{cls}">{pct:+.3f}%</span>'


# ─── Section renderers ───────────────────────────────────────────────────────

def render_parsed_log_section(log: ParsedLog, title: str = "Parsed Log") -> str:
    s = log.summary()
    rows = "".join(
        f'<div class="k">{_esc(k)}</div><div class="v">{_esc(v)}</div>'
        for k, v in s.items()
    )
    return f"""
    <div class="section">
      <h2>{_esc(title)}</h2>
      <div class="kv">{rows}</div>
    </div>
    """


def render_gap_report(rep: GapReport, title: str = "Backtest-vs-Live Gap") -> str:
    counts_row = (
        f'<div class="k">live trades</div><div class="v">{rep.n_live}</div>'
        f'<div class="k">backtest trades</div><div class="v">{rep.n_backtest}</div>'
        f'<div class="k">matched</div><div class="v">{rep.n_matched}</div>'
        f'<div class="k">live-only</div>'
        f'<div class="v"><span class="warn">{rep.n_live_only}</span></div>'
        f'<div class="k">backtest-only</div>'
        f'<div class="v"><span class="warn">{rep.n_backtest_only}</span></div>'
    )

    fill_table = f"""
    <h3>Fill price gap (positive = live paid worse)</h3>
    <table>
      <tr><th></th><th>mean</th><th>median</th><th>p90</th><th>p99</th></tr>
      <tr><td>entry</td>
        <td>{_bps_color(rep.mean_entry_gap_bps)}</td>
        <td>{_bps_color(rep.median_entry_gap_bps)}</td>
        <td>{_bps_color(rep.p90_entry_gap_bps)}</td>
        <td>{_bps_color(rep.p99_entry_gap_bps)}</td>
      </tr>
      <tr><td>exit</td>
        <td>{_bps_color(rep.mean_exit_gap_bps)}</td>
        <td>{_bps_color(rep.median_exit_gap_bps)}</td>
        <td>{_bps_color(rep.p90_exit_gap_bps)}</td>
        <td>{_bps_color(rep.p99_exit_gap_bps)}</td>
      </tr>
    </table>
    """

    slip_table = f"""
    <h3>Mean slippage</h3>
    <table>
      <tr><th></th><th>entry (bps)</th><th>exit (bps)</th></tr>
      <tr><td>live</td>
        <td>{rep.mean_live_entry_slip_bps:.1f}</td>
        <td>{rep.mean_live_exit_slip_bps:.1f}</td>
      </tr>
      <tr><td>backtest</td>
        <td>{rep.mean_bt_entry_slip_bps:.1f}</td>
        <td>{rep.mean_bt_exit_slip_bps:.1f}</td>
      </tr>
    </table>
    """

    if rep.attribution:
        attr_rows = "".join(
            f"<tr><td>{_esc(k)}</td><td>{_pct_color(v, neutral_band=0.05)}</td></tr>"
            for k, v in sorted(rep.attribution.items(), key=lambda kv: -abs(kv[1]))
        )
        attr_table = f"""
        <h3>Gap attribution (% of NAV per trade)</h3>
        <table>
          <tr><th>source</th><th>contribution</th></tr>
          {attr_rows}
        </table>
        """
    else:
        attr_table = ""

    return_row = (
        f'<div class="k">mean per-trade return gap</div>'
        f'<div class="v">{_pct_color(rep.mean_return_gap_pct, 0.5)}</div>'
        f'<div class="k">sum return gap</div>'
        f'<div class="v">{_pct_color(rep.sum_return_gap_pct, 5.0)}</div>'
    )

    # Per-pair table (matched only, top 50 by absolute return gap)
    matched = [p for p in rep.pairs if p.matched
               and p.return_gap_pct is not None]
    matched.sort(key=lambda p: -abs(p.return_gap_pct))
    rows = []
    for p in matched[:50]:
        rows.append(f"""
        <tr>
          <td>{_esc(p.symbol)}</td>
          <td>{_esc(p.live_entry_ts)}</td>
          <td>{_fmt(p.live_entry_price)}</td>
          <td>{_fmt(p.bt_entry_price)}</td>
          <td>{_bps_color(p.entry_price_gap_bps)}</td>
          <td>{_fmt(p.live_exit_price)}</td>
          <td>{_fmt(p.bt_exit_price)}</td>
          <td>{_bps_color(p.exit_price_gap_bps)}</td>
          <td>{_pct_color(p.return_gap_pct, 0.1)}</td>
        </tr>
        """)
    pair_table = ""
    if rows:
        pair_table = f"""
        <h3>Top matched pairs by |return gap| (max 50)</h3>
        <table>
          <tr><th>symbol</th><th>live entry ts</th>
              <th>live entry $</th><th>bt entry $</th><th>entry gap</th>
              <th>live exit $</th><th>bt exit $</th><th>exit gap</th>
              <th>return gap</th></tr>
          {''.join(rows)}
        </table>
        """

    return f"""
    <div class="section">
      <h2>{_esc(title)}</h2>
      <div class="kv">{counts_row}{return_row}</div>
      {fill_table}
      {slip_table}
      {attr_table}
      {pair_table}
    </div>
    """


def render_survivability(s: SurvivabilityResult,
                         title: str = "Walk-Forward Survivability") -> str:
    badge = ('<span class="tag tag-blessed">BLESSED</span>'
             if s.blessed else '<span class="tag tag-rejected">NOT BLESSED</span>')
    win_rows = []
    for w in s.windows:
        cat = '<span class="tag tag-rejected">CATASTROPHIC</span>' if w.catastrophic else ''
        win_rows.append(f"""
          <tr>
            <td>{_esc(w.window)}</td>
            <td>{_pct_color(w.net_return_pct, 1.0)}</td>
            <td>{w.drawdown_pct:.2f}%</td>
            <td>{w.sharpe:+.2f}</td>
            <td>{w.win_rate_pct:.1f}%</td>
            <td>{w.trades}</td>
            <td>{cat}</td>
          </tr>
        """)

    return f"""
    <div class="section">
      <h2>{_esc(title)}: {_esc(s.param_id)} {badge}</h2>
      <div class="kv">
        <div class="k">survivability score</div>
        <div class="v">{s.survivability_score:+.3f}</div>
        <div class="k">mean return / std</div>
        <div class="v">{s.mean_return_pct:+.2f}% / {s.std_return_pct:.2f}%</div>
        <div class="k">worst window return</div>
        <div class="v">{_pct_color(s.worst_window_return_pct, 1.0)}</div>
        <div class="k">worst window drawdown</div>
        <div class="v">{s.worst_window_dd_pct:.2f}%</div>
        <div class="k">positive windows</div>
        <div class="v">{s.n_positive_windows}/{len(s.windows)}</div>
        <div class="k">catastrophic windows</div>
        <div class="v">{s.n_catastrophic_windows}</div>
      </div>
      <h3>Per-window results</h3>
      <table>
        <tr><th>window</th><th>net return</th><th>MDD</th>
            <th>Sharpe</th><th>WR</th><th>trades</th><th>flags</th></tr>
        {''.join(win_rows)}
      </table>
    </div>
    """


def render_sweep(sw: SweepResult, title: str = "Walk-Forward Sweep") -> str:
    rows = []
    for r in sw.ranked[:20]:
        badge = ('<span class="tag tag-blessed">✓</span>'
                 if r.blessed else '')
        rows.append(f"""
          <tr>
            <td>{_esc(r.param_id)}</td>
            <td>{r.survivability_score:+.2f} {badge}</td>
            <td>{r.mean_return_pct:+.2f}%</td>
            <td>{r.std_return_pct:.2f}%</td>
            <td>{_pct_color(r.worst_window_return_pct, 1.0)}</td>
            <td>{r.worst_window_dd_pct:.2f}%</td>
            <td>{r.n_positive_windows}/{len(r.windows)}</td>
            <td>{r.n_catastrophic_windows}</td>
          </tr>
        """)
    return f"""
    <div class="section">
      <h2>{_esc(title)}</h2>
      <p>Total parameter sets: {len(sw.results)} —
         <strong>{len(sw.blessed)} blessed</strong></p>
      <table>
        <tr><th>param_id</th><th>score</th><th>mean ret</th><th>std</th>
            <th>worst ret</th><th>worst DD</th>
            <th>positive</th><th>catastrophic</th></tr>
        {''.join(rows)}
      </table>
    </div>
    """


# ─── Top-level page renderer ─────────────────────────────────────────────────

def render_audit_page(
    *,
    title: str,
    parsed_log: ParsedLog | None = None,
    gap_report: GapReport | None = None,
    sweep_result: SweepResult | None = None,
    survivability: SurvivabilityResult | None = None,
    notes: str | None = None,
) -> str:
    """Render a complete audit page from any combination of inputs."""
    sections = []
    if parsed_log is not None:
        sections.append(render_parsed_log_section(parsed_log, "Live log summary"))
    if gap_report is not None:
        sections.append(render_gap_report(gap_report))
    if survivability is not None:
        sections.append(render_survivability(survivability))
    if sweep_result is not None:
        sections.append(render_sweep(sweep_result))
    if notes:
        sections.append(
            f'<div class="section"><h2>Notes</h2><pre>{_esc(notes)}</pre></div>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_esc(title)}</title>
<style>{CSS}</style>
</head>
<body>
<h1>{_esc(title)}</h1>
{''.join(sections)}
<div class="footer">
  Generated {datetime.now(timezone.utc).isoformat()} by backtest_audit.report
</div>
</body>
</html>
"""


def write_audit_page(path: str, **kwargs) -> str:
    """Render and write the page; return absolute path."""
    html_text = render_audit_page(**kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_text)
    return os.path.abspath(path)
