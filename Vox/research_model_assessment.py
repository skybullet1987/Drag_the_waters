"""
research_model_assessment.py — Run in QC Research to analyze per-model votes.

STEP 1 (run this cell first to clear old data):
    exec(open('research_model_assessment.py').read())

If the results only show 5 models (lr, rf, lgbm, et, gnb), the trade log
is from an OLD backtest without full vote logging. Fix:
  1. Run Cell A below to clear old logs
  2. Re-run the gatling backtest (it now has _log_model_votes=True)
  3. Re-run this notebook
"""

from QuantConnect.Research import QuantBook
import json

qb = QuantBook()

# ── CELL A: Clear old logs (uncomment and run ONCE before re-backtesting) ────
# print("Clearing old trade logs...")
# for key in ["vox/trade_log.jsonl", "vox/model_vote_outcomes.jsonl",
#             "vox/trade_vote_audit.jsonl"]:
#     if qb.object_store.contains_key(key):
#         qb.object_store.delete(key)
#         print(f"  Deleted {key}")
# print("Done! Now re-run the backtest with risk_profile=gatling")
# raise SystemExit

# ── Load data ────────────────────────────────────────────────────────────────

all_records = []
for key in ["vox/trade_log.jsonl", "vox/model_vote_outcomes.jsonl"]:
    if qb.object_store.contains_key(key):
        raw = qb.object_store.read(key)
        recs = [json.loads(l) for l in raw.splitlines() if l.strip()]
        print(f"Loaded {key}: {len(recs)} records")
        all_records.extend(recs)

print(f"Total records: {len(all_records)}")

# Find all vote fields
VOTE_KEYS = ["active_votes", "model_votes", "per_model", "votes",
             "shadow_votes", "diagnostic_votes"]

# Collect ALL unique model names across all records
all_model_names = set()
records_with_votes = []
for r in all_records:
    found_votes = {}
    for vk in VOTE_KEYS:
        v = r.get(vk)
        if isinstance(v, dict) and v:
            found_votes.update(v)
            all_model_names.update(v.keys())
    if found_votes:
        records_with_votes.append(r)

print(f"Records with vote data: {len(records_with_votes)}")
print(f"Unique models found: {sorted(all_model_names)}")

# ── Build trades with all available votes ────────────────────────────────────

trades = []
# Use exits that have return data
for r in records_with_votes:
    ret = r.get("realized_return", r.get("ret", None))
    if ret is None:
        # Try to match with an exit
        continue
    all_votes = {}
    for vk in VOTE_KEYS:
        v = r.get(vk)
        if isinstance(v, dict) and v:
            for mid, prob in v.items():
                tag = ""
                if vk == "shadow_votes": tag = " [S]"
                elif vk == "diagnostic_votes": tag = " [D]"
                all_votes[mid + tag] = prob
    if all_votes:
        trades.append({
            "votes": all_votes,
            "ret": float(ret),
            "win": float(ret) > 0,
            "sym": r.get("symbol", "?"),
            "tag": r.get("tag", r.get("exit_reason", "")),
        })

# If no exits have votes, pair entries→exits
if len(trades) < 10:
    entries = [r for r in records_with_votes if r.get("event") == "entry_attempt"]
    exits = {r.get("symbol",""): r for r in all_records if r.get("event") == "exit"}
    for e in entries:
        sym = e.get("symbol", "")
        if sym in exits:
            ex = exits.pop(sym)
            ret = ex.get("realized_return", ex.get("ret", 0))
            if ret is None: continue
            all_votes = {}
            for vk in VOTE_KEYS:
                v = e.get(vk)
                if isinstance(v, dict) and v:
                    for mid, prob in v.items():
                        tag = ""
                        if vk == "shadow_votes": tag = " [S]"
                        elif vk == "diagnostic_votes": tag = " [D]"
                        all_votes[mid + tag] = prob
            if all_votes:
                trades.append({"votes": all_votes, "ret": float(ret),
                               "win": float(ret) > 0, "sym": sym,
                               "tag": ex.get("tag", "")})

wins = sum(1 for t in trades if t["win"])
print(f"\nTrades for assessment: {len(trades)} (W:{wins} L:{len(trades)-wins} WR:{wins/len(trades)*100:.1f}%)" if trades else "No trades!")

# ── Compute per-model accuracy ───────────────────────────────────────────────

results = {}
for t in trades:
    for mid, prob in t["votes"].items():
        try: prob = float(prob)
        except: continue
        if mid not in results:
            results[mid] = {"yr":[],"nr":[],"yw":0,"yc":0,"nc":0}
        s = results[mid]
        if prob >= 0.50:
            s["yc"] += 1; s["yr"].append(t["ret"])
            if t["win"]: s["yw"] += 1
        else:
            s["nc"] += 1; s["nr"].append(t["ret"])

# ── Print full report ────────────────────────────────────────────────────────

output = []
output.append("=" * 90)
output.append(f"PER-MODEL ACCURACY — {len(trades)} trades")
output.append("=" * 90)
output.append(f"{'Model':<20} {'Yes':>4} {'No':>4} {'WR%':>6} {'AvgRetY':>9} {'PF':>7} {'Edge':>8}")
output.append("-" * 90)

ranked = []
for mid, s in results.items():
    yc, nc = s["yc"], s["nc"]
    yr, nr = s["yr"], s["nr"]
    avg_y = sum(yr)/len(yr) if yr else 0
    avg_n = sum(nr)/len(nr) if nr else 0
    wr = s["yw"]/yc if yc > 0 else 0
    gw = sum(r for r in yr if r > 0)
    gl = abs(sum(r for r in yr if r < 0))
    pf = gw/gl if gl > 0 else (999 if gw > 0 else 0)
    ranked.append((mid, yc, nc, wr, avg_y, avg_n, pf, avg_y - avg_n))

ranked.sort(key=lambda x: x[6], reverse=True)

for mid, yc, nc, wr, avg_y, avg_n, pf, edge in ranked:
    if yc < 2: continue
    pf_s = f"{pf:.2f}" if pf < 100 else "inf"
    output.append(f"{mid:<20} {yc:>4} {nc:>4} {wr*100:>5.1f}% {avg_y*100:>+8.3f}% {pf_s:>7} {edge*100:>+7.3f}%")

output.append("")
output.append("=" * 90)
output.append("RECOMMENDATIONS")
output.append("=" * 90)
for mid, yc, nc, wr, avg_y, avg_n, pf, edge in ranked:
    if yc < 3: continue
    if pf >= 1.2 and wr >= 0.42:
        output.append(f"  KEEP    {mid:<18} PF={pf:.2f} WR={wr*100:.0f}% Edge={edge*100:+.2f}%")
    elif wr < 0.30:
        output.append(f"  VETO    {mid:<18} PF={pf:.2f} WR={wr*100:.0f}% (anti-signal)")
    elif pf < 0.8:
        output.append(f"  REMOVE  {mid:<18} PF={pf:.2f} WR={wr*100:.0f}% (destroys value)")
    else:
        output.append(f"  WATCH   {mid:<18} PF={pf:.2f} WR={wr*100:.0f}%")

# Per-symbol
output.append("")
output.append("PER-SYMBOL:")
sym_stats = {}
for t in trades:
    sym = t["sym"]
    if sym not in sym_stats: sym_stats[sym] = {"n":0,"w":0,"r":0}
    sym_stats[sym]["n"] += 1; sym_stats[sym]["r"] += t["ret"]
    if t["win"]: sym_stats[sym]["w"] += 1
for sym, ss in sorted(sym_stats.items(), key=lambda x: x[1]["r"], reverse=True):
    wr = ss["w"]/ss["n"]*100
    output.append(f"  {sym:<10} trades={ss['n']:>3} WR={wr:>5.1f}% totalRet={ss['r']*100:>+7.2f}%")

# Per exit tag
output.append("")
output.append("EXIT REASONS:")
tag_stats = {}
for t in trades:
    tag = t.get("tag","?") or "?"
    if tag not in tag_stats: tag_stats[tag] = {"n":0,"w":0,"r":0}
    tag_stats[tag]["n"] += 1; tag_stats[tag]["r"] += t["ret"]
    if t["win"]: tag_stats[tag]["w"] += 1
for tag, ts in sorted(tag_stats.items(), key=lambda x: x[1]["n"], reverse=True):
    wr = ts["w"]/ts["n"]*100
    output.append(f"  {tag:<15} n={ts['n']:>3} WR={wr:>5.1f}% avgRet={ts['r']/ts['n']*100:>+6.2f}%")

text = "\n".join(output)
print(text)

# Save to ObjectStore so it can be retrieved
try:
    qb.object_store.save("vox/model_assessment_report.txt", text)
    print("\nReport saved to ObjectStore: vox/model_assessment_report.txt")
except:
    pass
