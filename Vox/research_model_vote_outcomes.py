"""
research_model_vote_outcomes.py — Vox Model Vote Outcome Diagnostics
=====================================================================
Paste this script into the QuantConnect Research environment (or run cell by
cell in a Jupyter notebook) after a backtest has written
``vox/model_vote_outcomes.jsonl`` to the ObjectStore.

Usage
-----
1. Run a Vox backtest (this fills ``vox/model_vote_outcomes.jsonl``).
2. Open a QuantConnect Research notebook.
3. Paste the entire file (or cell by cell) and run.

What it produces
----------------
* Closed-trades table (one row per trade)
* Overall performance summary (win rate, avg win/loss, expectancy)
* Per-model yes-vote accuracy at threshold 0.50
* Threshold sweep (0.40 – 0.70) per model
* Ensemble signal diagnostics (vote_score / vote_yes_fraction / n_agree)
* Symbol performance
* Exit-reason performance
* Market-mode performance
* Active / shadow / diagnostic vote-group summaries
* ObjectStore CSV exports:
    vox/export_closed_trades_full.csv
    vox/export_model_summary.csv
    vox/export_threshold_sweep.csv
    vox/export_ensemble_summary.csv
    vox/export_symbol_perf.csv
    vox/export_exit_perf.csv
    vox/export_market_perf.csv

Requirements
------------
* pandas  (available in QC Research)
* No other third-party dependencies.
"""

import json

# ─── Load from ObjectStore ────────────────────────────────────────────────────

JSONL_KEY = "vox/model_vote_outcomes.jsonl"

try:
    raw = qb.object_store.read(JSONL_KEY)  # noqa: F821  (qb is injected by QC)
except Exception as _e:
    raise RuntimeError(
        f"Cannot read '{JSONL_KEY}' from ObjectStore: {_e}\n"
        "Run a Vox backtest first so the algorithm writes model vote outcomes."
    ) from _e

if not raw or not raw.strip():
    raise RuntimeError(
        f"'{JSONL_KEY}' is empty in the ObjectStore.\n"
        "Run a Vox backtest first — the algorithm writes this file on each "
        "closed trade when the diagnostics feature is enabled."
    )

records = []
for i, line in enumerate(raw.strip().split("\n"), start=1):
    line = line.strip()
    if not line:
        continue
    try:
        records.append(json.loads(line))
    except json.JSONDecodeError as exc:
        print(f"[warn] Skipping malformed line {i}: {exc}")

if not records:
    raise RuntimeError(
        f"'{JSONL_KEY}' contains no valid JSON records.\n"
        "Check the ObjectStore contents."
    )

print(f"Loaded {len(records)} closed-trade records from '{JSONL_KEY}'.")

# ─── Build DataFrame ──────────────────────────────────────────────────────────

import pandas as pd  # noqa: E402

df = pd.DataFrame(records)

# Coerce numeric columns
_num_cols = [
    "realized_return", "max_return_seen", "class_proba", "pred_return",
    "ev", "final_score", "tp", "sl", "vote_score", "vote_yes_fraction",
    "top3_mean", "n_agree", "std_proba", "hold_minutes", "entry_price", "exit_price",
]
for col in _num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "winner" in df.columns:
    df["winner"] = df["winner"].astype(bool)

if "entry_time" in df.columns:
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce", utc=True)
if "exit_time" in df.columns:
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce", utc=True)

print(f"\n=== Closed Trades ({len(df)} rows) ===")
_display_cols = [c for c in [
    "trade_id", "symbol", "entry_time", "exit_time",
    "exit_reason", "realized_return", "winner",
    "hold_minutes", "class_proba", "vote_score", "n_agree",
] if c in df.columns]
print(df[_display_cols].to_string(index=False))

# ─── Overall performance ──────────────────────────────────────────────────────

print("\n=== Overall Performance ===")
n_total  = len(df)
n_win    = int(df["winner"].sum()) if "winner" in df.columns else 0
n_loss   = n_total - n_win
win_rate = n_win / n_total if n_total else 0.0

avg_win  = df.loc[df["winner"],  "realized_return"].mean()  if "winner" in df.columns else float("nan")
avg_loss = df.loc[~df["winner"], "realized_return"].mean()  if "winner" in df.columns else float("nan")
avg_ret  = df["realized_return"].mean() if "realized_return" in df.columns else float("nan")

pl_ratio = abs(avg_win / avg_loss) if avg_loss and avg_loss != 0 else float("nan")
expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss) if pd.notna(avg_win) and pd.notna(avg_loss) else float("nan")

print(f"  Trades:      {n_total}")
print(f"  Winners:     {n_win}  ({win_rate:.1%})")
print(f"  Losers:      {n_loss}  ({1-win_rate:.1%})")
print(f"  Avg win:     {avg_win:+.4f}" if pd.notna(avg_win) else "  Avg win:     n/a")
print(f"  Avg loss:    {avg_loss:+.4f}" if pd.notna(avg_loss) else "  Avg loss:    n/a")
print(f"  P/L ratio:   {pl_ratio:.3f}" if pd.notna(pl_ratio) else "  P/L ratio:   n/a")
print(f"  Expectancy:  {expectancy:+.5f}" if pd.notna(expectancy) else "  Expectancy:  n/a")
print(f"  Avg return:  {avg_ret:+.4f}" if pd.notna(avg_ret) else "  Avg return:  n/a")

# ─── Per-model yes-vote accuracy at threshold 0.50 ───────────────────────────

def _expand_model_rows(df_in, vote_col, threshold=0.50):
    """Expand per-model vote dicts into one row per (trade, model)."""
    rows = []
    for _, row in df_in.iterrows():
        votes = row.get(vote_col)
        if not isinstance(votes, dict):
            continue
        for model_id, proba in votes.items():
            try:
                proba_f = float(proba)
            except (TypeError, ValueError):
                continue
            rows.append({
                "trade_id":    row.get("trade_id", ""),
                "symbol":      row.get("symbol", ""),
                "model_id":    model_id,
                "vote_source": vote_col.replace("_votes", ""),
                "proba":       proba_f,
                "voted_yes":   proba_f >= threshold,
                "realized_return": row.get("realized_return", float("nan")),
                "winner":      row.get("winner", False),
            })
    return pd.DataFrame(rows)


print("\n=== Per-Model Yes-Vote Accuracy (threshold=0.50) ===")
all_vote_dfs = []
for vcol in ("model_votes", "active_votes", "shadow_votes", "diagnostic_votes"):
    if vcol not in df.columns:
        continue
    vdf = _expand_model_rows(df, vcol, threshold=0.50)
    if not vdf.empty:
        all_vote_dfs.append(vdf)

if all_vote_dfs:
    all_models_df = pd.concat(all_vote_dfs, ignore_index=True)
    # Deduplicate (model_votes overlaps with active/shadow/diagnostic)
    all_models_df = all_models_df.drop_duplicates(subset=["trade_id", "model_id"])

    model_summary_rows = []
    for model_id, grp in all_models_df.groupby("model_id"):
        yes_grp  = grp[grp["voted_yes"]]
        no_grp   = grp[~grp["voted_yes"]]
        row = {
            "model_id":         model_id,
            "vote_source":      grp["vote_source"].iloc[0],
            "n_trades":         len(grp),
            "n_yes":            len(yes_grp),
            "n_no":             len(no_grp),
            "yes_win_rate":     yes_grp["winner"].mean() if len(yes_grp) else float("nan"),
            "no_win_rate":      no_grp["winner"].mean()  if len(no_grp)  else float("nan"),
            "yes_avg_return":   yes_grp["realized_return"].mean() if len(yes_grp) else float("nan"),
            "no_avg_return":    no_grp["realized_return"].mean()  if len(no_grp)  else float("nan"),
            "mean_proba":       grp["proba"].mean(),
        }
        model_summary_rows.append(row)

    model_summary_df = pd.DataFrame(model_summary_rows).sort_values("yes_win_rate", ascending=False)
    print(model_summary_df.to_string(index=False))
else:
    model_summary_df = pd.DataFrame()
    print("  No model vote columns found in records.")

# ─── Threshold sweep ─────────────────────────────────────────────────────────

print("\n=== Threshold Sweep (0.40 – 0.70, step 0.05) ===")
sweep_rows = []
thresholds = [round(t / 100, 2) for t in range(40, 75, 5)]

if all_vote_dfs:
    for thresh in thresholds:
        for model_id, grp in all_models_df.groupby("model_id"):
            yes_grp = grp[grp["proba"] >= thresh]
            if len(yes_grp) == 0:
                continue
            sweep_rows.append({
                "threshold": thresh,
                "model_id":  model_id,
                "n_yes":     len(yes_grp),
                "win_rate":  yes_grp["winner"].mean(),
                "avg_return": yes_grp["realized_return"].mean(),
            })
    threshold_sweep_df = pd.DataFrame(sweep_rows)
    print(threshold_sweep_df.to_string(index=False))
else:
    threshold_sweep_df = pd.DataFrame()
    print("  No model vote data available for threshold sweep.")

# ─── Ensemble signal diagnostics ─────────────────────────────────────────────

print("\n=== Ensemble Signal Diagnostics ===")
ens_cols = [c for c in ["vote_score", "vote_yes_fraction", "top3_mean", "n_agree", "std_proba"] if c in df.columns]
if ens_cols:
    ens_summary = []
    for col in ens_cols:
        ens_summary.append({
            "metric": col,
            "mean":   df[col].mean(),
            "median": df[col].median(),
            "min":    df[col].min(),
            "max":    df[col].max(),
            "win_corr": df[col].corr(df["winner"].astype(float)) if "winner" in df.columns else float("nan"),
        })
    ens_df = pd.DataFrame(ens_summary)
    print(ens_df.to_string(index=False))
    # Split by winner
    for col in ["vote_score", "vote_yes_fraction"]:
        if col in df.columns:
            w_mean = df.loc[df["winner"],  col].mean()
            l_mean = df.loc[~df["winner"], col].mean()
            print(f"  {col}: winners={w_mean:.4f}  losers={l_mean:.4f}")
else:
    ens_df = pd.DataFrame()
    print("  No ensemble metric columns found.")

# ─── Symbol performance ───────────────────────────────────────────────────────

print("\n=== Symbol Performance ===")
if "symbol" in df.columns and "realized_return" in df.columns:
    sym_perf = (
        df.groupby("symbol")
        .agg(
            n_trades=("realized_return", "count"),
            win_rate=("winner", "mean"),
            avg_return=("realized_return", "mean"),
            total_return=("realized_return", "sum"),
        )
        .sort_values("avg_return", ascending=False)
        .reset_index()
    )
    print(sym_perf.to_string(index=False))
else:
    sym_perf = pd.DataFrame()
    print("  Missing symbol/realized_return columns.")

# ─── Exit reason performance ──────────────────────────────────────────────────

print("\n=== Exit Reason Performance ===")
if "exit_reason" in df.columns and "realized_return" in df.columns:
    exit_perf = (
        df.groupby("exit_reason")
        .agg(
            n_trades=("realized_return", "count"),
            win_rate=("winner", "mean"),
            avg_return=("realized_return", "mean"),
            avg_hold_min=("hold_minutes", "mean"),
        )
        .sort_values("avg_return", ascending=False)
        .reset_index()
    )
    print(exit_perf.to_string(index=False))
else:
    exit_perf = pd.DataFrame()
    print("  Missing exit_reason/realized_return columns.")

# ─── Market mode performance ──────────────────────────────────────────────────

print("\n=== Market Mode Performance ===")
if "market_mode" in df.columns and "realized_return" in df.columns:
    market_perf = (
        df.groupby("market_mode", dropna=False)
        .agg(
            n_trades=("realized_return", "count"),
            win_rate=("winner", "mean"),
            avg_return=("realized_return", "mean"),
        )
        .sort_values("avg_return", ascending=False)
        .reset_index()
    )
    print(market_perf.to_string(index=False))
else:
    market_perf = pd.DataFrame()
    print("  Missing market_mode/realized_return columns.")

# ─── Vote-group summaries ─────────────────────────────────────────────────────

print("\n=== Vote-Group Summaries (active / shadow / diagnostic) ===")
for vcol in ("active_votes", "shadow_votes", "diagnostic_votes"):
    if vcol not in df.columns:
        continue
    vdf = _expand_model_rows(df, vcol, threshold=0.50)
    if vdf.empty:
        print(f"  {vcol}: no data")
        continue
    grp_rows = []
    for mid, gdf in vdf.groupby("model_id"):
        yes_mask = gdf["voted_yes"]
        grp_rows.append({
            "model_id":  mid,
            "n":         len(gdf),
            "yes_wr":    gdf.loc[yes_mask, "winner"].mean() if yes_mask.any() else float("nan"),
            "avg_proba": gdf["proba"].mean(),
        })
    grp = pd.DataFrame(grp_rows)
    print(f"\n  {vcol}:")
    print(grp.to_string(index=False))

# ─── Export CSVs to ObjectStore ───────────────────────────────────────────────

def _save_csv(df_out, key):
    """Save a DataFrame as CSV to the ObjectStore."""
    if df_out is None or df_out.empty:
        print(f"  [skip] {key} — empty DataFrame")
        return
    try:
        qb.object_store.save(key, df_out.to_csv(index=False))  # noqa: F821
        print(f"  Saved: {key}  ({len(df_out)} rows)")
    except Exception as exc:
        print(f"  [error] Could not save {key}: {exc}")


print("\n=== Exporting CSVs to ObjectStore ===")
_save_csv(df[_display_cols] if _display_cols else df, "vox/export_closed_trades_full.csv")
_save_csv(model_summary_df,   "vox/export_model_summary.csv")
_save_csv(threshold_sweep_df, "vox/export_threshold_sweep.csv")
_save_csv(ens_df,             "vox/export_ensemble_summary.csv")
_save_csv(sym_perf,           "vox/export_symbol_perf.csv")
_save_csv(exit_perf,          "vox/export_exit_perf.csv")
_save_csv(market_perf,        "vox/export_market_perf.csv")

print("\nDone. Use qb.object_store.read('vox/export_*.csv') to download results.")
