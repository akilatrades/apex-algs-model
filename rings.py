"""Where does the zone end? Ring sequences from placement.gg (data/rings/ring_sequences.txt) on the same map frame as the
landing-spot outlines (data/rings/poi_outlines.txt). Frame: u left->right, v top->bottom, both 0..1 of the map width.

predict(ring1, ring2): find past games whose first two rings sat close to these, weight them by closeness, and return
where their final zones ended. Tested on held-out games (leave-one-out) against simply using the whole map's history.
"""
from pathlib import Path
import numpy as np

D = Path(__file__).parent / "data" / "rings"
B = {c: i for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")}
MAPS = ["E-District", "Storm Point", "World's Edge", "Olympus"]


def load():
    out, cur = {}, None
    lines = [l for l in (D / "ring_sequences.txt").read_text().splitlines() if l and not l.startswith("# ")]
    i = 0
    while i < len(lines):
        name, radii, n = lines[i][1:].split("|"); n = int(n)
        s = lines[i + 1]; assert len(s) == 7 * n, (name, len(s), n)
        algs = [int(x) for x in lines[i + 2][2:].split(",") if x]
        rec = np.array([[B[c] for c in s[k * 7:(k + 1) * 7]] for k in range(n)])
        r1 = rec[:, 0:2] / 63; r2 = rec[:, 2:4] / 63
        v = rec[:, 4] * 4096 + rec[:, 5] * 64 + rec[:, 6]
        end = np.stack([(v // 512) / 511, (v % 512) / 511], 1)
        a = np.zeros(n, bool); a[algs] = True
        out[name] = dict(r1=r1, r2=r2, end=end, algs=a, radii=[float(x) for x in radii.split(",")])
        i += 3
    return out


def outlines():
    P, cur = {}, None
    for l in (D / "poi_outlines.txt").read_text().splitlines():
        if l.startswith("# ") or not l.strip():
            continue
        if l.startswith("#"):
            cur = l[1:]; P[cur] = []; continue
        n, p = l.split("|")
        P[cur].append((n, np.array([[float(a) for a in q.split(",")] for q in p.split()])))
    return P


def inside(pt, poly):
    x, y = pt; c = False
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, 0)):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            c = not c
    return c


def nearest_spot(pt, polys, maxd=0.05):
    """Landing spot the point is in, else the closest outline within maxd (5% of map width), else None."""
    for n, p in polys:
        if inside(pt, p):
            return n
    best, bd = None, maxd
    for n, p in polys:
        d = np.min(np.hypot(p[:, 0] - pt[0], p[:, 1] - pt[1]))
        if d < bd:
            best, bd = n, d
    return best


def weights(m, r1, r2, bw=0.05):
    """Kernel weights of past games given ring 1 and ring 2 centres (ring 2 matters most)."""
    d = np.hypot(*(m["r2"] - r2).T) ** 2 + 0.25 * np.hypot(*(m["r1"] - r1).T) ** 2
    return np.exp(-d / (2 * bw ** 2))


def loo_test(m, bws=(0.03, 0.05, 0.08), algs_only=True):
    """Leave-one-out: for each held-out game, predict its endzone from all other games. Compare mean distance between
    the true endzone and (a) the weighted average prediction, (b) the map-wide average endzone. Also: how often the true
    endzone's landing spot is among the 3 likeliest spots."""
    idx = np.where(m["algs"])[0] if algs_only else np.arange(len(m["end"]))
    res = {}
    for bw in bws:
        err, base, hit3, n3 = [], [], 0, 0
        for i in idx:
            w = weights(m, m["r1"][i], m["r2"][i], bw); w[i] = 0
            pred = (w[:, None] * m["end"]).sum(0) / w.sum()
            mask = np.ones(len(w), bool); mask[i] = False
            err.append(np.hypot(*(pred - m["end"][i]))); base.append(np.hypot(*(m["end"][mask].mean(0) - m["end"][i])))
        res[bw] = dict(err=float(np.mean(err)), base=float(np.mean(base)), n=len(idx))
    return res


def evaluate(m, polys, bw=0.03, subset=None, k_top=3):
    """Leave-one-out on `subset` (indexes; default: ALGS games). Reports mean distance (fraction of map width) for:
    kernel prediction, ring-2 centre, map-wide average; and how often the true final landing spot is in the top-k predicted
    spots, versus picking the k spots that are most common on the map overall."""
    idx = np.where(m["algs"])[0] if subset is None else subset
    spots = [nearest_spot(p, polys) for p in m["end"]]
    names = sorted({s for s in spots if s})
    S = np.array([names.index(s) if s else -1 for s in spots])
    overall = np.bincount(S[S >= 0], minlength=len(names))
    e_k, e_r2, e_avg, hit, hit_base, cover = [], [], [], 0, 0, []
    for i in idx:
        w = weights(m, m["r1"][i], m["r2"][i], bw); w[i] = 0
        pred = (w[:, None] * m["end"]).sum(0) / w.sum()
        e_k.append(np.hypot(*(pred - m["end"][i]))); e_r2.append(np.hypot(*(m["r2"][i] - m["end"][i])))
        msk = np.ones(len(w), bool); msk[i] = False
        e_avg.append(np.hypot(*(m["end"][msk].mean(0) - m["end"][i])))
        sw = np.bincount(S[S >= 0], weights=w[S >= 0], minlength=len(names))
        top = np.argsort(-sw)[:k_top]; ob = overall.copy(); ob[S[i]] -= 1 if S[i] >= 0 else 0
        if S[i] in top: hit += 1
        if S[i] in np.argsort(-ob)[:k_top]: hit_base += 1
        cover.append(sw[top].sum() / sw.sum())
    n = len(idx)
    return dict(n=n, err_pred=float(np.mean(e_k)), err_ring2=float(np.mean(e_r2)), err_avg=float(np.mean(e_avg)),
                top3_hit=hit / n, top3_hit_base=hit_base / n, top3_said=float(np.mean(cover)))


def page_data():
    """Everything the pages need: outlines, ring sizes, the raw sequences (decoded in the browser), per-spot end shares,
    team landing spots and how often the zone ended on them, and the held-out check."""
    import json, features as FE
    M, P = load(), outlines()
    raw = [l for l in (D / "ring_sequences.txt").read_text().splitlines() if l and not l.startswith("# ")]
    drops = FE.champs_drops()
    chk = json.loads((Path(__file__).parent / "out" / "rings_check.json").read_text())
    maps = {}
    for k, name in enumerate(MAPS):
        m, polys = M[name], P[name]
        spots = [nearest_spot(p, polys) for p in m["end"]]
        n = len(spots)
        share = {s: spots.count(s) / n for s, _ in polys}
        algs = [s for s, a in zip(spots, m["algs"]) if a]
        share_algs = {s: algs.count(s) / len(algs) for s, _ in polys} if len(algs) >= 30 else None
        teams = {}
        for t, d in drops.items():
            sp = d.get(name)
            if sp and sp in share:
                teams[t] = dict(spot=sp, share=round(share[sp], 4))
        hdr, seq, a = raw[3 * k], raw[3 * k + 1], raw[3 * k + 2]
        maps[name] = dict(radii=[float(x) for x in hdr.split("|")[1].split(",")], n=n, seq=seq, algs=a[2:],
                          outlines=[[s, [[round(x, 3), round(y, 3)] for x, y in p]] for s, p in polys],
                          share={s: round(v, 4) for s, v in share.items()},
                          share_algs={s: round(v, 4) for s, v in share_algs.items()} if share_algs else None,
                          teams=teams, check=chk[name])
    return dict(maps=maps, source="placement.gg", fetched="25 Nov 2026")
