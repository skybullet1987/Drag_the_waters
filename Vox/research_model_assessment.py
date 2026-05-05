"""
research_model_assessment.py — QC Research Notebook for per-model vote analysis.

Upload this file to your QuantConnect project and run it in the Research Environment.
It reads vox/trade_log.jsonl from ObjectStore and produces a detailed per-model
accuracy breakdown showing which ensemble models have genuine predictive signal.

Usage in QC Research:
  1. Open your project on quantconnect.com
  2. Click "Research" tab
  3. Create a new Python notebook
  4. Paste this code into a cell and run

The output shows:
  - Per-model win rate, profit factor, and edge when the model votes YES
  - Model ranking by predictive power
  - Recommendations for which models to keep, demote, or use as veto
"""

# ── Cell 1: Load trade log from ObjectStore ──────────────────────────────────

from QuantConnect.Research import QuantBook
import json

qb = QuantBook()

# Read the trade log
raw = ""
for key in ["vox/trade_log.jsonl", "vox/trade_vote_audit.jsonl"]:
    if qb.object_store.contains_key(key):
        raw = qb.object_store.read(key)
        print(f"Loaded {key}: {len(raw)} bytes")
        break

if not raw:
    print("ERROR: No trade log found in ObjectStore.")
    print("Run a backtest first (risk_profile=gatling) to generate trade data.")
else:
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    print(f"Total records: {len(records)}")

    # Classify records
    events = {}
    for r in records:
        ev = r.get("event", "unknown")
        events[ev] = events.get(ev, 0) + 1
    print(f"Event types: {events}")


# ── Cell 2: Extract entry attempts with model votes ──────────────────────────

entries = [r for r in records if r.get("event") == "entry_attempt"]
exits = [r for r in records if r.get("event") == "exit"]

print(f"\nEntry attempts: {len(entries)}")
print(f"Exits: {len(exits)}")

if entries:
    sample = entries[0]
    print(f"\nSample entry keys: {list(sample.keys())}")
    # Show vote-related fields
    for k in ["active_votes", "shadow_votes", "diagnostic_votes",
              "model_votes", "vote_score", "vote_yes_fraction",
              "class_proba", "entry_path", "confirm", "market_mode"]:
        if k in sample:
            v = sample[k]
            if isinstance(v, dict):
                print(f"  {k}: {len(v)} models -> {list(v.keys())}")
            else:
                print(f"  {k}: {v}")


# ── Cell 3: Pair entries with exits, compute per-model accuracy ──────────────

# Build exit lookup by symbol + approximate time
def pair_trades(entries, exits):
    """Pair entry_attempt records with their corresponding exit records."""
    paired = []
    exit_idx = 0
    for entry in entries:
        sym = entry.get("symbol", "")
        # Find matching exit (same symbol, after entry time)
        best_exit = None
        for ex in exits:
            if ex.get("symbol", "") == sym:
                if best_exit is None:
                    best_exit = ex
                    break
        if best_exit:
            exits.remove(best_exit)
            paired.append({
                "entry": entry,
                "exit": best_exit,
                "symbol": sym,
                "realized_return": best_exit.get("realized_return",
                    best_exit.get("ret", 0.0)),
                "winner": best_exit.get("realized_return",
                    best_exit.get("ret", 0.0)) > 0,
                "exit_tag": best_exit.get("tag", best_exit.get("exit_reason", "")),
            })
    return paired

paired = pair_trades(list(entries), list(exits))
print(f"\nPaired trades: {len(paired)}")

# Alternatively, if exits have active_votes directly
direct_exits = [r for r in records
    if r.get("event") == "exit"
    and (r.get("active_votes") or r.get("model_votes"))]
print(f"Exits with vote data: {len(direct_exits)}")


# ── Cell 4: Per-model accuracy computation ───────────────────────────────────

def compute_model_accuracy(trade_records, vote_threshold=0.50):
    """Compute per-model accuracy from trade records.

    Accepts either paired trades (entry+exit) or flat records with
    active_votes + realized_return.
    """
    model_stats = {}

    for trade in trade_records:
        # Get votes from entry or directly from record
        if isinstance(trade, dict) and "entry" in trade:
            votes = (trade["entry"].get("active_votes") or
                     trade["entry"].get("model_votes") or {})
            ret = trade.get("realized_return", 0.0)
            winner = trade.get("winner", ret > 0)
        else:
            votes = (trade.get("active_votes") or
                     trade.get("model_votes") or {})
            ret = trade.get("realized_return", trade.get("ret", 0.0))
            winner = ret > 0

        # Also check shadow_votes and diagnostic_votes
        all_votes = dict(votes)
        for vk in ["shadow_votes", "diagnostic_votes"]:
            extra = None
            if isinstance(trade, dict) and "entry" in trade:
                extra = trade["entry"].get(vk, {})
            else:
                extra = trade.get(vk, {})
            if extra:
                for model_id, proba in extra.items():
                    all_votes[f"{model_id}[{vk[:4]}]"] = proba

        for model_id, proba in all_votes.items():
            if model_id not in model_stats:
                model_stats[model_id] = {
                    "yes_returns": [], "no_returns": [],
                    "yes_wins": 0, "yes_count": 0, "no_count": 0,
                }
            s = model_stats[model_id]
            try:
                proba = float(proba)
            except (TypeError, ValueError):
                continue
            if proba >= vote_threshold:
                s["yes_count"] += 1
                s["yes_returns"].append(ret)
                if winner:
                    s["yes_wins"] += 1
            else:
                s["no_count"] += 1
                s["no_returns"].append(ret)

    result = {}
    for model_id, s in model_stats.items():
        yc = s["yes_count"]
        nc = s["no_count"]
        yr = s["yes_returns"]
        nr = s["no_returns"]
        avg_yes = sum(yr) / len(yr) if yr else 0.0
        avg_no = sum(nr) / len(nr) if nr else 0.0
        wr = s["yes_wins"] / yc if yc > 0 else 0.0
        gw = sum(r for r in yr if r > 0)
        gl = abs(sum(r for r in yr if r < 0))
        pf = gw / gl if gl > 0 else (float("inf") if gw > 0 else 0.0)

        result[model_id] = {
            "yes_count": yc, "no_count": nc, "total": yc + nc,
            "win_rate_when_yes": wr,
            "avg_return_when_yes": avg_yes,
            "avg_return_when_no": avg_no,
            "profit_factor": pf,
            "edge": avg_yes - avg_no,
            "gross_wins": gw, "gross_losses": gl,
        }
    return result


# Use paired trades if available, otherwise use entries with votes
if paired:
    accuracy = compute_model_accuracy(paired)
elif direct_exits:
    accuracy = compute_model_accuracy(direct_exits)
elif entries:
    accuracy = compute_model_accuracy(entries)
else:
    accuracy = {}
    print("No usable trade data found!")


# ── Cell 5: Display results ──────────────────────────────────────────────────

print("\n" + "=" * 80)
print("PER-MODEL ACCURACY REPORT")
print("=" * 80)
print(f"{'Model':<20} {'Role':^8} {'Yes':>4} {'No':>4} {'WR%':>6} "
      f"{'AvgRet':>8} {'PF':>8} {'Edge':>8}")
print("-" * 80)

# Separate active, shadow, diagnostic
active_ids = set()
shadow_ids = set()
diag_ids = set()
for mid in accuracy:
    if "[shad]" in mid or "[diag]" in mid:
        base = mid.split("[")[0]
        if "[shad]" in mid:
            shadow_ids.add(mid)
        else:
            diag_ids.add(mid)
    else:
        active_ids.add(mid)

# Sort by profit factor
ranked = sorted(accuracy.items(), key=lambda kv: kv[1]["profit_factor"], reverse=True)

for model_id, s in ranked:
    if s["yes_count"] < 3:
        continue
    role = "active" if model_id in active_ids else (
        "shadow" if model_id in shadow_ids else "diag")
    wr = s["win_rate_when_yes"] * 100
    ar = s["avg_return_when_yes"] * 100
    pf = s["profit_factor"]
    edge = s["edge"] * 100
    pf_str = f"{pf:.2f}" if pf < 100 else "inf"
    print(f"{model_id:<20} {role:^8} {s['yes_count']:>4} {s['no_count']:>4} "
          f"{wr:>5.1f}% {ar:>+7.3f}% {pf_str:>8} {edge:>+7.3f}%")


# ── Cell 6: Recommendations ─────────────────────────────────────────────────

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

keep = []
demote = []
veto = []
insufficient = []

for mid, s in ranked:
    if mid in shadow_ids or mid in diag_ids:
        continue
    if s["yes_count"] < 5:
        insufficient.append(mid)
        continue
    wr = s["win_rate_when_yes"]
    pf = s["profit_factor"]
    edge = s["edge"]
    if pf >= 1.0 and wr >= 0.45:
        keep.append((mid, pf, wr))
    elif wr < 0.30:
        veto.append((mid, pf, wr))
    else:
        demote.append((mid, pf, wr))

print("\n  KEEP (active — genuine signal):")
if keep:
    for mid, pf, wr in keep:
        print(f"    {mid:<15} PF={pf:.2f}  WR={wr*100:.0f}%")
else:
    print("    (none with PF >= 1.0 and WR >= 45%)")

print("\n  DEMOTE to shadow (poor signal):")
if demote:
    for mid, pf, wr in demote:
        print(f"    {mid:<15} PF={pf:.2f}  WR={wr*100:.0f}%")
else:
    print("    (none)")

print("\n  USE AS VETO (anti-signal — their NO vote is informative):")
if veto:
    for mid, pf, wr in veto:
        print(f"    {mid:<15} PF={pf:.2f}  WR={wr*100:.0f}%  → when this model says YES, DON'T trade")
else:
    print("    (none)")

print("\n  INSUFFICIENT DATA (< 5 yes-votes):")
if insufficient:
    print(f"    {', '.join(insufficient)}")
else:
    print("    (none)")

print("\n" + "=" * 80)
print("To apply changes, update MODEL_ROLE_* and MODEL_WEIGHT_* in Vox/core.py")
print("or Vox/gatling_config.py, then re-run the backtest.")
print("=" * 80)
