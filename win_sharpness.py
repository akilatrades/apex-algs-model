"""Does the model under-rate how often strong teams win games? Plackett-Luce picks the winner with softmax(u).
Here the winner pick uses softmax(c*u) (c>1 = favourites win more), the rest of the order stays the same.
c is fitted on the Pro League held-out games (model trained on earlier games) and then tested on the LAN held-out games."""
import json
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from backtest3 import prep_all, run_fit, score, CONFIGS
import model3 as M3

KW = CONFIGS["New model (players + spot + match point)"]


def game_ll(g, c):
    u = g.sort_values("placement").u.to_numpy()
    first = c * u[0] - np.log(np.exp(c * u).sum())
    rest = u[1:]; mx = rest.max(); suf = np.cumsum(np.exp(rest - mx)[::-1])[::-1]
    return first + (rest[:-1].sum() - (np.log(suf[:-1]) + mx).sum())


def total(d, c):
    return sum(game_ll(g, c) for _, g in d.groupby("game_id"))


def win_calib(d, c):
    rows = []
    for _, g in d.groupby("game_id"):
        p = np.exp(c * g.u.to_numpy()); p /= p.sum()
        rows.append(pd.DataFrame(dict(p=p, w=(g.placement.to_numpy() == 1))))
    r = pd.concat(rows); r["b"] = pd.qcut(r.p.rank(method="first"), 5, labels=False)
    return [(round(float(x.p.mean()), 4), round(float(x.w.mean()), 4)) for _, x in r.groupby("b")], float(((r.p - r.w) ** 2).mean())


if __name__ == "__main__":
    df, stages, reg = prep_all()
    use = df.source.isin(["lan", "pro_league", "pl_s1"])
    testA = df[(df.order >= 2.8) & (df.source == "lan")]; trainA = df[(df.order < 2.8) & use]
    lanA = set(trainA[trainA.source == "lan"].team) | set(testA.team); debut = set(testA.team) - set(trainA[trainA.source == "lan"].team)
    pl = df[df.source == "pro_league"].copy(); day = pl.stage.str.extract(r"_D(\d)")[0].astype(float)
    testB = pl[(day >= 7) | pl.stage.str.endswith("FINALS")]
    trainB = df[use & ((df.order < 2.5) | ((df.source == "pro_league") & df.game_id.isin(pl[day <= 6].game_id)))]
    FB = run_fit(trainB, set(trainB[trainB.source == "lan"].team), reg, **KW)
    B = testB.copy(); B["u"] = M3.utilities(FB, B)
    c = float(minimize_scalar(lambda c: -total(B, c), bounds=(0.5, 3), method="bounded").x)
    FA = run_fit(trainA, lanA, reg, **KW); _, beta = score(FA, testA, debut)
    A = testA.copy(); A["u"] = M3.utilities(FA, A) - A.team.isin(debut) * beta
    nA = A.game_id.nunique()
    out = dict(c=c, lan_ll_before=total(A, 1.0) / nA, lan_ll_after=total(A, c) / nA,
               lan_calib_before=win_calib(A, 1.0), lan_calib_after=win_calib(A, c))
    print(json.dumps(out, indent=1))
    json.dump(out, open("out/win_sharpness.json", "w"), indent=1)
