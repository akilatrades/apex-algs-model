"""Does landing near where the zone ends help? Links each ALGS game (apexlegendsstatus id) to its ring sequence
(placement.gg) and each team's landing spot (dropsData), then measures the distance from the spot to the final zone."""
from pathlib import Path
import numpy as np, pandas as pd
import rings

D = Path(__file__).parent / "data" / "rings"
MAPC = {"SP": "Storm Point", "WE": "World's Edge", "OL": "Olympus", "ED": "E-District"}


def _tagged():
    out = {}
    for l in (D / "algs_tagged.txt").read_text().splitlines():
        m, recs = l[1:].split("|")
        for r in recs.split(","):
            c = [rings.B[x] for x in r[10:17]]
            v = c[4] * 4096 + c[5] * 64 + c[6]
            out[r[:10]] = dict(map=m, r1=np.array(c[0:2]) / 63, r2=np.array(c[2:4]) / 63,
                               end=np.array([(v // 512) / 511, (v % 512) / 511]))
    return out


def _ids():
    g = {}
    for l in (D / "als_game_ids.txt").read_text().splitlines():
        if l.startswith("G|"):
            _, st, body = l.split("|")
            for x in body.split(","):
                n, h = x.split(":"); g[(st, int(n))] = h
    return g


def _drops():
    lines = [l for l in (D / "als_game_drops.txt").read_text().splitlines() if l and not l.startswith("#")]
    T = lines[0][2:].split(";"); N = {}; out = {}
    for l in lines[1:]:
        if l.startswith("N|"):
            _, m, names = l.split("|"); N[m] = names.split(";"); continue
        st, m, body = l.split("|")
        for p in body.split():
            t, s = p.split("."); out[(st, MAPC[m], T[int(t)])] = N[m][int(s)]
    return out


def table():
    """One row per (game, team) with ring data: dist from landing-spot centre to final zone (and to ring 1/2 centres)."""
    from load import load
    df, _ = load()
    tag, ids, drops = _tagged(), _ids(), _drops()
    P = rings.outlines()
    cent = {m: {n: p.mean(0) for n, p in P[m]} for m in P}
    rows = []
    for r in df.itertuples():
        h = ids.get((r.stage, int(r.game)))
        if h is None or h not in tag: continue
        z = tag[h]
        if z["map"] != r.map: continue
        spot = drops.get((r.stage, r.map, r.team))
        if spot is None or spot not in cent[r.map]: continue
        c = cent[r.map][spot]
        rows.append(dict(game_id=r.game_id, stage=r.stage, map=r.map, team=r.team, spot=spot, placement=r.placement,
                         kills=r.kills, d_end=float(np.hypot(*(c - z["end"]))), d_r1=float(np.hypot(*(c - z["r1"]))),
                         d_r2=float(np.hypot(*(c - z["r2"])))))
    return pd.DataFrame(rows)


def quintiles(t=None):
    t = table() if t is None else t
    q = t.groupby("game_id").d_end.rank(pct=True)
    return [round(float(x), 2) for x in t.groupby(pd.cut(q, [0, .2, .4, .6, .8, 1])).placement.mean()]


if __name__ == "__main__":
    t = table()
    print(len(t), t.game_id.nunique(), "games")
    print(t.groupby("map").game_id.nunique())
    t["q"] = t.groupby("game_id").d_end.rank(pct=True)
    t["top5"] = t.placement <= 5
    print(t.groupby(pd.cut(t.q, [0, .2, .4, .6, .8, 1])).agg(n=("placement", "size"), avg_place=("placement", "mean"),
          top5=("top5", "mean"), win=("placement", lambda x: (x == 1).mean()), kills=("kills", "mean")))
    print("corr placement vs d_end", t[["placement", "d_end", "d_r1", "d_r2"]].corr(method="spearman").placement)
