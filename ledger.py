"""Prediction ledger: lock the Championship odds before the event, then score them after it, so the model can't quietly
rewrite its own history. The locked file is committed to git (the commit date is the proof of when it was made).

  python3 ledger.py lock    -> writes data/ledger/y6_champs.json once (refuses to overwrite)
  python3 ledger.py score   -> needs data/ledger/y6_champs_results.txt:  winner|Team   and   finalists|Team;Team;...
  python3 ledger.py status  -> writes out/ledger.json for the pages
update.py calls `auto()` every night: it locks on the first run on or after LOCK_ON and scores once results exist."""
import datetime, hashlib, json, math, sys
from pathlib import Path

ROOT = Path(__file__).parent
LOCKED = ROOT / "data" / "ledger" / "y6_champs.json"
RESULTS = ROOT / "data" / "ledger" / "y6_champs_results.txt"
OUT = ROOT / "out" / "ledger.json"
EVENT = "ALGS Year 6 Championship, Sapporo, 28-31 Jan 2027"
LOCK_ON = datetime.date(2027, 1, 26)       # last nightly run safely before day 1


def lock(force_date=None):
    if LOCKED.exists():
        return "already locked"
    R = json.loads((ROOT / "out" / "results.json").read_text())
    field = sorted([dict(team=t["team"], p_win=round(t["p_win"], 4), p_finals=round(t["p_finals"], 4)) for t in R["teams"] if t["in_field"]],
                   key=lambda r: -r["p_win"])
    body = dict(event=EVENT, locked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), teams=field)
    body["fingerprint"] = hashlib.sha256(json.dumps(body["teams"], sort_keys=True).encode()).hexdigest()[:16]
    LOCKED.write_text(json.dumps(body, indent=1))
    return "locked"


def read_results():
    if not RESULTS.exists():
        return None
    r = {}
    for l in RESULTS.read_text().splitlines():
        if "|" in l and not l.startswith("#"):
            k, v = l.split("|", 1); r[k.strip()] = [x.strip() for x in v.split(";") if x.strip()]
    return dict(winner=r["winner"][0], finalists=r.get("finalists", [])) if r.get("winner") else None


def score():
    L = json.loads(LOCKED.read_text()); res = read_results()
    if not res:
        return None
    P = {t["team"]: t for t in L["teams"]}; n = len(P)
    pw = P.get(res["winner"], {}).get("p_win", 0.0)
    rank = 1 + sum(t["p_win"] > pw for t in L["teams"])
    out = dict(winner=res["winner"], winner_p=pw, winner_rank=rank, even_p=1 / n,
               winner_vs_even=round(pw * n, 2))                         # how many times likelier than "everyone equal"
    if res["finalists"]:
        F = set(res["finalists"]); k = len(F)
        brier = sum((t["p_finals"] - (t["team"] in F)) ** 2 for t in L["teams"]) / n
        brier0 = sum((k / n - (t["team"] in F)) ** 2 for t in L["teams"]) / n
        top = [t["team"] for t in sorted(L["teams"], key=lambda r: -r["p_finals"])[:k]]
        out.update(finalists=sorted(F), finals_brier=round(brier, 4), finals_brier_even=round(brier0, 4),
                   finals_hits=len(F & set(top)), finals_k=k)
    return out


def status():
    s = dict(event=EVENT, lock_on=LOCK_ON.isoformat(), locked=LOCKED.exists())
    if LOCKED.exists():
        L = json.loads(LOCKED.read_text())
        s.update(locked_at=L["locked_at"], fingerprint=L["fingerprint"], top=L["teams"][:10])
        sc = score()
        if sc: s["score"] = sc
    OUT.write_text(json.dumps(s, indent=1))
    return s


def auto(today=None):
    today = today or datetime.date.today(); msg = []
    if not LOCKED.exists() and today >= LOCK_ON:
        msg.append("ledger: " + lock())
    s = status()
    if s.get("score"): msg.append(f"ledger: scored, winner {s['score']['winner']} had {s['score']['winner_p']:.1%}")
    elif s["locked"] and today > datetime.date(2027, 1, 31): msg.append("ledger: event over, add data/ledger/y6_champs_results.txt to score it")
    return msg


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    print(lock() if cmd == "lock" else score() if cmd == "score" else status())
