"""Legend bans and team comps, on top of team strength. Data: data/deep/bans_comps.txt (scrape/als_bans_comps.js, packed).

1. Bans: for each team and game, 'reliance' = how often the team picked the banned legend in its other games (0..1).
   Effect = how much a team's result drops per unit of reliance. Measured on earlier games, scored on games it hasn't seen.
2. Comps: one small effect per legend (and per legend on each map), on top of team strength. Same held-out test.
Ratings come from the new model fitted exactly as in backtest3 (checks A and B). Writes out/bans_comps.json."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import backtest3 as BT, model3 as M3
from load import norm_team

SRC = Path(__file__).parent / "data" / "deep" / "bans_comps.txt"
KW = BT.CONFIGS["New model (players + spot + match point)"]


A62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def read():
    """-> {(stage, game): dict(ban=legend or None, comps={team: (leg, leg, leg)})}"""
    G = {}; L = C = T = None; st = teams = None
    for l in SRC.read_text().splitlines():
        if not l or l.startswith("#"): continue
        if l.startswith("L|"): L = l[2:].split(";"); continue
        if l.startswith("C|"): C = [tuple(sorted(L[A62.index(ch)] for ch in c)) for c in l[2:].split(",")]; continue
        if l.startswith("T|"): T = l[2:].split(";"); continue
        if l.startswith("S|"):
            _, st, idx = l.split("|"); teams = [T[int(i)] for i in idx.split(",")]; continue
        g, ban, body = l.split("|")
        comps = {}
        for k, t in enumerate(teams):
            code = body[2 * k:2 * k + 2]
            if code and code != "__": comps[norm_team(t, st)] = C[A62.index(code[0]) * 62 + A62.index(code[1])]
        G[(st, int(g))] = dict(ban=None if ban == "_" else L[A62.index(ban)], comps=comps)
    return G


def attach(df, G):
    """per row: comp (tuple) and ban of that game"""
    key = list(zip(df.stage, df.game.astype(int)))
    df = df.copy()
    df["comp"] = [G.get(k, {}).get("comps", {}).get(t) for k, t in zip(key, df.team)]
    df["ban"] = [G.get(k, {}).get("ban") for k in key]
    return df


def reliance(train):
    """team -> legend -> share of the team's games (with a known comp) where it picked that legend"""
    out = {}
    for t, g in train[train.comp.notna()].groupby("team"):
        c = Counter(l for comp in g.comp for l in comp); n = len(g)
        out[t] = {l: k / n for l, k in c.items()}
    return out


def ll(U, X, b):
    tot = 0.0; grad = np.zeros_like(b)
    for u, x in zip(U, X):
        v = u + x @ b; mx = v.max(); e = np.exp(v - mx); suf = np.cumsum(e[::-1])[::-1]
        tot += v[:-1].sum() - (np.log(suf[:-1]) + mx).sum()
        # gradient: sum_k x_k - sum_j over risk sets of softmax-weighted x
        cw = np.cumsum((e[:, None] * x)[::-1], 0)[::-1]
        grad += x[:-1].sum(0) - (cw[:-1] / suf[:-1, None]).sum(0)
    return tot, grad


def fit_coef(U, X, sd):
    k = X[0].shape[1]
    def f(b):
        v, g = ll(U, X, b)
        return -v + (b ** 2).sum() / (2 * sd ** 2), -g + b / sd ** 2
    return minimize(f, np.zeros(k), jac=True, method="L-BFGS-B").x


def packs(F, d, feat):
    d = d.copy(); d["u"] = M3.utilities(F, d)
    U, X = [], []
    for _, g in d.groupby("game_id"):
        g = g.sort_values("placement"); U.append(g.u.to_numpy()); X.append(np.array([feat(r) for r in g.itertuples()], float))
    return U, X


def gain(U, X, b):
    return (ll(U, X, b)[0] - ll(U, X, np.zeros_like(b))[0]) / max(len(U), 1)


if __name__ == "__main__":
    df, stages, reg = BT.prep_all()
    G = read(); df = attach(df, G)
    cov = df.comp.notna().mean(); nb = df[df.ban.notna()].game_id.nunique()
    print(f"comp known for {cov:.1%} of team-games; {nb} games with a ban", flush=True)
    use = df.source.isin(["lan", "pro_league", "pl_s1"])
    testA = df[(df.order >= 2.8) & (df.source == "lan")]; trainA = df[(df.order < 2.8) & use]
    lanA = set(trainA[trainA.source == "lan"].team) | set(testA.team)
    pl = df[df.source == "pro_league"]; day = pl.stage.str.extract(r"_D(\d)")[0].astype(float)
    early = pl[day <= 6]; testB = pl[(day >= 7) | pl.stage.str.endswith("FINALS")]
    trainB = df[use & ((df.order < 2.5) | ((df.source == "pro_league") & df.game_id.isin(early.game_id)))]
    lanB = set(trainB[trainB.source == "lan"].team)
    FA = BT.run_fit(trainA, lanA, reg, **KW); FB = BT.run_fit(trainB, lanB, reg, **KW)
    out = dict(coverage=round(float(cov), 3), ban_games=int(nb))

    # ---------- 1. bans ----------
    def ban_feat(rel):
        return lambda r: [rel.get(r.team, {}).get(r.ban, 0.0) if isinstance(r.ban, str) else 0.0]
    # effect learned on the B training games (incl. Split 1 and early Pro League), then frozen
    relB = reliance(trainB)
    Uf, Xf = packs(FB, trainB[trainB.ban.notna()], ban_feat(relB))
    b = fit_coef(Uf, Xf, 1.0)
    h = 1e-3; c = (ll(Uf, Xf, b + h)[0] - 2 * ll(Uf, Xf, b)[0] + ll(Uf, Xf, b - h)[0]) / h ** 2
    res = dict(beta=round(float(b[0]), 3), beta_sd=round(float((-c) ** -0.5), 3), fit_games=len(Uf))
    for name, F, tr, te in [("lan", FA, trainA, testA), ("pl", FB, trainB, testB)]:
        U, X = packs(F, te[te.ban.notna()], ban_feat(reliance(tr)))
        res[name + "_games"] = len(U); res[name + "_gain"] = round(gain(U, X, b), 4)
    out["ban"] = res; print("ban", res, flush=True)

    # ---------- 2. comps ----------
    legs = sorted({l for c in df.comp.dropna() for l in c}); li = {l: i for i, l in enumerate(legs)}
    maps = sorted(df["map"].unique()); mi = {m: i for i, m in enumerate(maps)}

    def leg_feat(r):
        v = np.zeros(len(legs))
        if isinstance(r.comp, tuple):
            for l in r.comp: v[li[l]] = 1
        return v

    def legmap_feat(r):
        v = np.zeros(len(legs) * (len(maps) + 1))
        if isinstance(r.comp, tuple):
            for l in r.comp: v[li[l]] = 1; v[len(legs) * (1 + mi[r.map]) + li[l]] = 1
        return v
    trials = []
    for label, feat in [("legend", leg_feat), ("legend on each map", legmap_feat)]:
        for sd in (0.05, 0.1, 0.2):
            r_ = dict(kind=label, sd=sd)
            for name, F, tr, te in [("lan", FA, trainA, testA), ("pl", FB, trainB, testB)]:
                Uf, Xf = packs(F, tr[tr.comp.notna()], feat)
                bb = fit_coef(Uf, Xf, sd)
                U, X = packs(F, te, feat)
                r_[name + "_gain"] = round(gain(U, X, bb), 4)
                if name == "pl": r_["top"] = sorted(((legs[i % len(legs)], round(float(bb[i]), 3)) for i in range(len(legs))), key=lambda x: -x[1])
            trials.append(r_); print(r_["kind"], sd, r_["lan_gain"], r_["pl_gain"], flush=True)
    out["comp"] = trials
    json.dump(out, open("out/bans_comps.json", "w"), indent=1)
