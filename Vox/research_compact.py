"""Compact model assessment — paste output here for analysis."""
from QuantConnect.Research import QuantBook
import json
qb = QuantBook()
raw = qb.object_store.read("vox/trade_log.jsonl")
recs = [json.loads(l) for l in raw.splitlines() if l.strip()]
entries = [r for r in recs if r.get("event") == "entry_attempt"]
exits = [r for r in recs if r.get("event") == "exit"]
ex_q = {}
for ex in exits:
    s = ex.get("symbol","")
    ex_q.setdefault(s,[]).append(ex)
trades = []
for e in entries:
    s = e.get("symbol","")
    if s in ex_q and ex_q[s]:
        ex = ex_q[s].pop(0)
        ret = ex.get("realized_return", ex.get("ret", 0))
        if ret is None: continue
        votes = {}
        for vk in ["active_votes","model_votes","shadow_votes","diagnostic_votes"]:
            v = e.get(vk,{})
            if isinstance(v,dict):
                for m,p in v.items():
                    tag = ""
                    if vk == "shadow_votes": tag = "[S]"
                    elif vk == "diagnostic_votes": tag = "[D]"
                    votes[m+tag] = p
        trades.append({"v":votes,"r":float(ret),"w":float(ret)>0,
                        "s":s,"t":ex.get("tag","")})
W = sum(1 for t in trades if t["w"])
print(f"TRADES: {len(trades)} W:{W} L:{len(trades)-W} WR:{W/len(trades)*100:.1f}%")
print()
R = {}
for t in trades:
    for m,p in t["v"].items():
        try: p=float(p)
        except: continue
        if m not in R: R[m]={"yr":[],"nr":[],"yw":0,"yc":0,"nc":0}
        s=R[m]
        if p>=0.5:
            s["yc"]+=1;s["yr"].append(t["r"])
            if t["w"]:s["yw"]+=1
        else:
            s["nc"]+=1;s["nr"].append(t["r"])
print(f"{'MODEL':<18}{'Y':>3}{'N':>4}{'WR':>6}{'RET':>8}{'PF':>7}{'EDGE':>8}")
rk=[]
for m,s in R.items():
    yc=s["yc"];nc=s["nc"];yr=s["yr"];nr=s["nr"]
    ay=sum(yr)/len(yr) if yr else 0
    an=sum(nr)/len(nr) if nr else 0
    wr=s["yw"]/yc if yc>0 else 0
    gw=sum(r for r in yr if r>0);gl=abs(sum(r for r in yr if r<0))
    pf=gw/gl if gl>0 else(99 if gw>0 else 0)
    rk.append((m,yc,nc,wr,ay,pf,ay-an))
rk.sort(key=lambda x:x[5],reverse=True)
for m,yc,nc,wr,ay,pf,edge in rk:
    if yc<2:continue
    print(f"{m:<18}{yc:>3}{nc:>4}{wr*100:>5.0f}%{ay*100:>+7.2f}%{pf:>6.2f}{edge*100:>+7.2f}%")
print()
# symbols
print("SYMBOLS:")
ss={}
for t in trades:
    ss.setdefault(t["s"],{"n":0,"w":0,"r":0})
    ss[t["s"]]["n"]+=1;ss[t["s"]]["r"]+=t["r"]
    if t["w"]:ss[t["s"]]["w"]+=1
for sym,v in sorted(ss.items(),key=lambda x:x[1]["r"],reverse=True):
    print(f"  {sym:<8} n={v['n']:>2} WR={v['w']/v['n']*100:>4.0f}% ret={v['r']*100:>+6.1f}%")
print()
print("EXITS:")
ts={}
for t in trades:
    tg=t.get("t","?") or "?"
    ts.setdefault(tg,{"n":0,"w":0,"r":0})
    ts[tg]["n"]+=1;ts[tg]["r"]+=t["r"]
    if t["w"]:ts[tg]["w"]+=1
for tg,v in sorted(ts.items(),key=lambda x:x[1]["n"],reverse=True):
    print(f"  {tg:<12} n={v['n']:>3} WR={v['w']/v['n']*100:>4.0f}% avg={v['r']/v['n']*100:>+5.2f}%")
