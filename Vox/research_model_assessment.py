"""
research_model_assessment.py — QC Research Notebook for per-model vote analysis.

Upload to your QC project and run in the Research Environment.
Reads vox/trade_log.jsonl + vox/model_vote_outcomes.jsonl from ObjectStore.

Usage in QC Research:
  1. Open your project on quantconnect.com
  2. Click "Research" tab
  3. Create a new Python notebook
  4. Paste:  exec(open('research_model_assessment.py').read())
  5. Run
"""

from QuantConnect.Research import QuantBook
import json

qb = QuantBook()

# ── Load all available trade data sources ────────────────────────────────────

all_records = []
sources_loaded = []

for key in ["vox/trade_log.jsonl", "vox/trade_vote_audit.jsonl",
            "vox/model_vote_outcomes.jsonl"]:
    if qb.object_store.contains_key(key):
        raw = qb.object_store.read(key)
        recs = [json.loads(l) for l in raw.splitlines() if l.strip()]
        print(f"Loaded {key}: {len(raw):,} bytes, {len(recs)} records")
        all_records.extend(recs)
        sources_loaded.append(key)

if not all_records:
    print("ERROR: No trade data found. Run a backtest first.")
    raise SystemExit

# ── Inspect what fields are available ────────────────────────────────────────

print(f"\nTotal records across all sources: {len(all_records)}")

# Classify by event type
events = {}
for r in all_records:
    ev = r.get("event", r.get("entry_type", "unknown"))
    events[ev] = events.get(ev, 0) + 1
print(f"Event types: {events}")

# Find records that have per-model vote data
VOTE_KEYS = ["model_votes", "active_votes", "per_model", "votes"]
records_with_votes = []
for r in all_records:
    for vk in VOTE_KEYS:
        v = r.get(vk)
        if isinstance(v, dict) and len(v) > 0:
            records_with_votes.append(r)
            break

print(f"Records with per-model vote data: {len(records_with_votes)}")

# If entry_attempt records have votes, prefer those paired with exits
entries_with_votes = [r for r in records_with_votes
                      if r.get("event") == "entry_attempt"]
exits_with_votes = [r for r in records_with_votes
                    if r.get("event") == "exit"
                    or r.get("entry_type") == "exit"]

print(f"  Entry attempts with votes: {len(entries_with_votes)}")
print(f"  Exits with votes: {len(exits_with_votes)}")

# Show all available fields from a record with votes
if records_with_votes:
    sample = records_with_votes[0]
    print(f"\nSample record keys: {sorted(sample.keys())}")
    for vk in VOTE_KEYS:
        v = sample.get(vk)
        if isinstance(v, dict) and v:
            print(f"  {vk}: {len(v)} models -> {list(v.keys())[:15]}")

# ── Build trade list for assessment ──────────────────────────────────────────

def extract_votes(record):
    """Extract the best available vote dict from a record."""
    for vk in ["active_votes", "model_votes", "per_model", "votes"]:
        v = record.get(vk)
        if isinstance(v, dict) and len(v) > 0:
            return dict(v), vk
    return {}, None

def extract_shadow_diag(record):
    """Extract shadow and diagnostic votes."""
    shadow = record.get("shadow_votes", {})
    diag = record.get("diagnostic_votes", {})
    if isinstance(shadow, dict) and isinstance(diag, dict):
        return shadow, diag
    return {}, {}

# Strategy: pair entry_attempt (with votes) → matching exit (with return)
# If exits have their own votes, use those directly

trades_for_assessment = []

# Method 1: Use exits that have both votes AND realized_return
for r in exits_with_votes:
    votes, src = extract_votes(r)
    ret = r.get("realized_return", r.get("ret", None))
    if votes and ret is not None:
        shadow, diag = extract_shadow_diag(r)
        trades_for_assessment.append({
            "votes": votes,
            "shadow_votes": shadow,
            "diagnostic_votes": diag,
            "realized_return": float(ret),
            "winner": float(ret) > 0,
            "symbol": r.get("symbol", "?"),
            "exit_tag": r.get("tag", r.get("exit_reason", "")),
            "source": "exit_with_votes",
        })

# Method 2: Pair entries (with votes) to exits (with returns) by symbol
if len(trades_for_assessment) < 20 and entries_with_votes:
    exits_all = [r for r in all_records if r.get("event") == "exit"]
    # Build exit queue per symbol
    exit_queue = {}
    for ex in exits_all:
        sym = ex.get("symbol", "")
        if sym not in exit_queue:
            exit_queue[sym] = []
        exit_queue[sym].append(ex)

    used_exits = set()
    for entry in entries_with_votes:
        sym = entry.get("symbol", "")
        votes, src = extract_votes(entry)
        if not votes or sym not in exit_queue:
            continue
        for i, ex in enumerate(exit_queue[sym]):
            if id(ex) not in used_exits:
                ret = ex.get("realized_return", ex.get("ret", None))
                if ret is not None:
                    used_exits.add(id(ex))
                    shadow, diag = extract_shadow_diag(entry)
                    trades_for_assessment.append({
                        "votes": votes,
                        "shadow_votes": shadow,
                        "diagnostic_votes": diag,
                        "realized_return": float(ret),
                        "winner": float(ret) > 0,
                        "symbol": sym,
                        "exit_tag": ex.get("tag", ""),
                        "source": "paired_entry_exit",
                    })
                    exit_queue[sym].pop(i)
                    break

# Method 3: Use model_vote_outcomes records (from research_model_vote_outcomes.py)
for r in all_records:
    if r.get("event") not in (None, "entry_attempt", "exit"):
        votes, src = extract_votes(r)
        ret = r.get("realized_return", r.get("ret", r.get("return", None)))
        if votes and ret is not None:
            trades_for_assessment.append({
                "votes": votes,
                "shadow_votes": r.get("shadow_votes", {}),
                "diagnostic_votes": r.get("diagnostic_votes", {}),
                "realized_return": float(ret),
                "winner": float(ret) > 0,
                "symbol": r.get("symbol", "?"),
                "source": "other_source",
            })

print(f"\nTrades available for model assessment: {len(trades_for_assessment)}")
sources = {}
for t in trades_for_assessment:
    s = t["source"]
    sources[s] = sources.get(s, 0) + 1
print(f"  By source: {sources}")

winners = sum(1 for t in trades_for_assessment if t["winner"])
losers = len(trades_for_assessment) - winners
print(f"  Winners: {winners}, Losers: {losers}, WR: {winners/len(trades_for_assessment)*100:.1f}%"
      if trades_for_assessment else "")

# ── Compute per-model accuracy ───────────────────────────────────────────────

def compute_accuracy(trades, vote_threshold=0.50):
    model_stats = {}
    for t in trades:
        all_votes = dict(t["votes"])
        # Tag shadow and diagnostic votes
        for mid, p in t.get("shadow_votes", {}).items():
            all_votes[f"{mid} [shadow]"] = p
        for mid, p in t.get("diagnostic_votes", {}).items():
            all_votes[f"{mid} [diag]"] = p

        ret = t["realized_return"]
        win = t["winner"]
        for mid, proba in all_votes.items():
            try:
                proba = float(proba)
            except (TypeError, ValueError):
                continue
            if mid not in model_stats:
                model_stats[mid] = {"yr": [], "nr": [], "yw": 0, "yc": 0, "nc": 0}
            s = model_stats[mid]
            if proba >= vote_threshold:
                s["yc"] += 1; s["yr"].append(ret)
                if win: s["yw"] += 1
            else:
                s["nc"] += 1; s["nr"].append(ret)

    results = {}
    for mid, s in model_stats.items():
        yc, nc = s["yc"], s["nc"]
        yr, nr = s["yr"], s["nr"]
        avg_y = sum(yr)/len(yr) if yr else 0.0
        avg_n = sum(nr)/len(nr) if nr else 0.0
        wr = s["yw"]/yc if yc > 0 else 0.0
        gw = sum(r for r in yr if r > 0)
        gl = abs(sum(r for r in yr if r < 0))
        pf = gw/gl if gl > 0 else (float("inf") if gw > 0 else 0.0)
        results[mid] = {
            "yes": yc, "no": nc, "total": yc+nc,
            "wr": wr, "avg_y": avg_y, "avg_n": avg_n,
            "pf": pf, "edge": avg_y - avg_n,
            "gw": gw, "gl": gl,
        }
    return results

if not trades_for_assessment:
    print("\nNo trades with vote data available for assessment!")
    print("Make sure the backtest ran with LOG_MODEL_VOTES=True or that")
    print("entry_attempt records include model_votes/active_votes dicts.")
    raise SystemExit

accuracy = compute_accuracy(trades_for_assessment)

# ── Display results ──────────────────────────────────────────────────────────

print("\n" + "=" * 85)
print("PER-MODEL ACCURACY REPORT")
print(f"Based on {len(trades_for_assessment)} trades with vote data")
print("=" * 85)
print(f"{'Model':<22} {'Yes':>4} {'No':>4} {'WR%':>6} "
      f"{'AvgRet':>9} {'PF':>7} {'Edge':>8} {'GrossW':>8} {'GrossL':>8}")
print("-" * 85)

ranked = sorted(accuracy.items(), key=lambda kv: kv[1]["pf"], reverse=True)
for mid, s in ranked:
    if s["yes"] < 3:
        continue
    wr = s["wr"] * 100
    ar = s["avg_y"] * 100
    pf = s["pf"]
    edge = s["edge"] * 100
    pf_s = f"{pf:.2f}" if pf < 100 else "inf"
    print(f"{mid:<22} {s['yes']:>4} {s['no']:>4} {wr:>5.1f}% "
          f"{ar:>+8.3f}% {pf_s:>7} {edge:>+7.3f}% "
          f"{s['gw']*100:>+7.2f}% {s['gl']*100:>7.2f}%")

# ── Models with too few votes ────────────────────────────────────────────────
sparse = [(mid, s) for mid, s in ranked if s["yes"] < 3 and s["total"] > 0]
if sparse:
    print(f"\nModels with < 3 yes-votes (insufficient data):")
    for mid, s in sparse:
        print(f"  {mid}: {s['yes']} yes, {s['no']} no, total={s['total']}")

# ── Recommendations ──────────────────────────────────────────────────────────

print("\n" + "=" * 85)
print("RECOMMENDATIONS")
print("=" * 85)

keep, demote, veto, promote = [], [], [], []
for mid, s in ranked:
    if "[shadow]" in mid or "[diag]" in mid:
        base = mid.split(" [")[0]
        if s["yes"] >= 5 and s["pf"] >= 1.2 and s["wr"] >= 0.45:
            promote.append((mid, base, s))
        continue
    if s["yes"] < 5:
        continue
    if s["pf"] >= 1.0 and s["wr"] >= 0.40:
        keep.append((mid, s))
    elif s["wr"] < 0.30:
        veto.append((mid, s))
    else:
        demote.append((mid, s))

print("\n  ✓ KEEP as active (genuine signal):")
if keep:
    for mid, s in keep:
        print(f"    {mid:<18} PF={s['pf']:.2f}  WR={s['wr']*100:.0f}%  Edge={s['edge']*100:+.2f}%")
else:
    print("    (none meet PF >= 1.0 and WR >= 40%)")

print("\n  ↑ PROMOTE from shadow → active:")
if promote:
    for mid, base, s in promote:
        print(f"    {base:<18} PF={s['pf']:.2f}  WR={s['wr']*100:.0f}%  (currently {mid})")
else:
    print("    (none meet PF >= 1.2 and WR >= 45%)")

print("\n  ↓ DEMOTE to shadow (weak signal):")
if demote:
    for mid, s in demote:
        print(f"    {mid:<18} PF={s['pf']:.2f}  WR={s['wr']*100:.0f}%")
else:
    print("    (none)")

print("\n  ✗ USE AS VETO (anti-signal — their YES means DON'T trade):")
if veto:
    for mid, s in veto:
        print(f"    {mid:<18} PF={s['pf']:.2f}  WR={s['wr']*100:.0f}%")
else:
    print("    (none with WR < 30%)")

# ── Per-symbol breakdown ─────────────────────────────────────────────────────

print("\n" + "=" * 85)
print("PER-SYMBOL BREAKDOWN")
print("=" * 85)
sym_stats = {}
for t in trades_for_assessment:
    sym = t["symbol"]
    if sym not in sym_stats:
        sym_stats[sym] = {"n": 0, "wins": 0, "total_ret": 0.0}
    sym_stats[sym]["n"] += 1
    sym_stats[sym]["total_ret"] += t["realized_return"]
    if t["winner"]:
        sym_stats[sym]["wins"] += 1

print(f"{'Symbol':<12} {'Trades':>6} {'Wins':>5} {'WR%':>6} {'TotalRet':>10}")
print("-" * 42)
for sym, ss in sorted(sym_stats.items(), key=lambda x: x[1]["total_ret"], reverse=True):
    wr = ss["wins"]/ss["n"]*100 if ss["n"]>0 else 0
    print(f"{sym:<12} {ss['n']:>6} {ss['wins']:>5} {wr:>5.1f}% {ss['total_ret']*100:>+9.2f}%")

# ── Exit tag analysis ────────────────────────────────────────────────────────

print("\n" + "=" * 85)
print("EXIT REASON BREAKDOWN")
print("=" * 85)
tag_stats = {}
for t in trades_for_assessment:
    tag = t.get("exit_tag", "unknown") or "unknown"
    if tag not in tag_stats:
        tag_stats[tag] = {"n": 0, "wins": 0, "total_ret": 0.0}
    tag_stats[tag]["n"] += 1
    tag_stats[tag]["total_ret"] += t["realized_return"]
    if t["winner"]:
        tag_stats[tag]["wins"] += 1

print(f"{'Exit Reason':<20} {'Count':>6} {'Wins':>5} {'WR%':>6} {'AvgRet':>8}")
print("-" * 48)
for tag, ts in sorted(tag_stats.items(), key=lambda x: x[1]["n"], reverse=True):
    wr = ts["wins"]/ts["n"]*100 if ts["n"]>0 else 0
    ar = ts["total_ret"]/ts["n"]*100 if ts["n"]>0 else 0
    print(f"{tag:<20} {ts['n']:>6} {ts['wins']:>5} {wr:>5.1f}% {ar:>+7.2f}%")

print("\n" + "=" * 85)
print("DONE — Update MODEL_ROLE_* / MODEL_WEIGHT_* in core.py and re-backtest.")
print("=" * 85)
