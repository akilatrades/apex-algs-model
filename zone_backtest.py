"""How much does the zone matter, on top of team strength? For ALGS games with ring data:
  x = how much closer a team's landing spot is to the zone centre than the lobby average (per 10% of map width).
Fit one effect size on the training games and score the held-out games, for three moments:
  'end'  = the final zone (only known after the game: an upper bound),
  'r2'   = ring 2 centre (known mid-game: what the live tracker can use),
  'r1'   = ring 1 centre (known shortly after the drop).
Ratings come from the new model fitted exactly as in backtest3 (checks A and B). Writes out/zone_effect.json."""
import json
import numpy as np
from scipy.optimize import minimize_scalar
import backtest3 as BT, model3 as M3, zone_effect as ZE

KW = BT.CONFIGS["New model (players + spot + match point)"]


def ll_rows(g_u, x, b):
    ll = 0.0
    for u, xx in zip(g_u, x):
        v = u + b * xx; mx = v.max(); suf = np.cumsum(np.exp(v - mx)[::-1])[::-1]
        ll += v[:-1].sum() - (np.log(suf[:-1]) + mx).sum()
    return ll


def packs(F, d, Z, col):
    d = d[d.game_id.isin(Z.game_id.unique())].copy()
    d["u"] = M3.utilities(F, d)
    d = d.merge(Z[["game_id", "team", col]], on=["game_id", "team"], how="left")
    U, X = [], []
    for _, g in d.groupby("game_id"):
        g = g.sort_values("placement")
        z = g[col].to_numpy(float); m = np.nanmean(z)
        x = np.where(np.isnan(z), 0.0, (m - z) / 0.1)
        U.append(g.u.to_numpy()); X.append(x)
    return U, X


if __name__ == "__main__":
    df, stages, reg = BT.prep_all()
    Z = ZE.table()
    use = df.source.isin(["lan", "pro_league", "pl_s1"])
    testA = df[(df.order >= 2.8) & (df.source == "lan")]; trainA = df[(df.order < 2.8) & use]
    lanA = set(trainA[trainA.source == "lan"].team) | set(testA.team)
    pl = df[df.source == "pro_league"].copy(); day = pl.stage.str.extract(r"_D(\d)")[0].astype(float)
    early = pl[day <= 6]; testB = pl[(day >= 7) | pl.stage.str.endswith("FINALS")]
    trainB = df[use & ((df.order < 2.5) | ((df.source == "pro_league") & df.game_id.isin(early.game_id)))]
    lanB = set(trainB[trainB.source == "lan"].team)
    FA = BT.run_fit(trainA, lanA, reg, **KW); FB = BT.run_fit(trainB, lanB, reg, **KW)
    out = {}
    for col in ["d_end", "d_r2", "d_r1"]:
        # effect size learned on PL days 1-6 (+ LCQ) using the B ratings, then frozen
        fitset = df[df.game_id.isin(early.game_id) | df.stage.str.startswith("LCQ_")]
        Uf, Xf = packs(FB, fitset, Z, col)
        b = minimize_scalar(lambda b: -ll_rows(Uf, Xf, b), bounds=(-1, 1), method="bounded").x
        # curvature -> rough sd
        h = 1e-3; c = (ll_rows(Uf, Xf, b + h) - 2 * ll_rows(Uf, Xf, b) + ll_rows(Uf, Xf, b - h)) / h ** 2
        r = dict(beta=round(float(b), 3), beta_sd=round(float((-c) ** -0.5), 3), fit_games=len(Uf))
        for name, F, test in [("lan", FA, testA), ("pl", FB, testB)]:
            U, X = packs(F, test, Z, col)
            r[name + "_games"] = len(U)
            r[name + "_gain"] = round((ll_rows(U, X, b) - ll_rows(U, X, 0.0)) / max(len(U), 1), 4)
        out[col] = r; print(col, r, flush=True)
    json.dump(out, open("out/zone_effect.json", "w"), indent=1)
