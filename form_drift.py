"""How far does a team's level move from our pre-event rating once an event starts?
For each held-out event, give every team a free shift d on top of its pre-event rating, fit all shifts by maximum
likelihood, and take sd(true shift)^2 = var(fitted shifts) - average(sampling noise^2). Converted to points a game."""
import json
import numpy as np
from backtest3 import prep_all, run_fit, score, CONFIGS
import model3 as M3
KW = CONFIGS["New model (players + spot + match point)"]


def shifts(d):
    teams = sorted(d.team.unique()); ti = {t: i for i, t in enumerate(teams)}
    G = [(np.array([ti[t] for t in g.sort_values("placement").team]), g.sort_values("placement").u.to_numpy()) for _, g in d.groupby("game_id")]
    x = np.zeros(len(teams))
    for it in range(60):
        gr = np.zeros(len(teams)); h = np.zeros(len(teams))
        for T, u0 in G:
            u = u0 + x[T]; e = np.exp(u - u.max()); suf = np.cumsum(e[::-1])[::-1]
            a1 = np.cumsum(1 / suf[:-1]); a1 = np.append(a1, a1[-1]); a2 = np.cumsum(1 / suf[:-1] ** 2); a2 = np.append(a2, a2[-1])
            g = np.ones(len(u)); g[-1] = 0; g -= e * a1
            np.add.at(gr, T, g); np.add.at(h, T, -(e * a1 - e * e * a2))
        h -= 1e-3; step = -gr / h; x += step; x -= x.mean()
        if np.abs(step).max() < 1e-6: break
    n = d.groupby("team").game_id.nunique().reindex(teams).to_numpy()
    return x, 1 / np.sqrt(-h), n


if __name__ == "__main__":
    df, stages, reg = prep_all()
    use = df.source.isin(["lan", "pro_league", "pl_s1"])
    testA = df[(df.order >= 2.8) & (df.source == "lan")]; trainA = df[(df.order < 2.8) & use]
    lanA = set(trainA[trainA.source == "lan"].team) | set(testA.team); debut = set(testA.team) - set(trainA[trainA.source == "lan"].team)
    FA = run_fit(trainA, lanA, reg, **KW); _, beta = score(FA, testA, debut)
    A = testA.copy(); A["u"] = M3.utilities(FA, A) - A.team.isin(debut) * beta
    pl = df[df.source == "pro_league"].copy(); day = pl.stage.str.extract(r"_D(\d)")[0].astype(float)
    testB = pl[(day >= 7) | pl.stage.str.endswith("FINALS")]
    trainB = df[use & ((df.order < 2.5) | ((df.source == "pro_league") & df.game_id.isin(pl[day <= 6].game_id)))]
    FB = run_fit(trainB, set(trainB[trainB.source == "lan"].team), reg, **KW)
    Bd = testB.copy(); Bd["u"] = M3.utilities(FB, Bd)
    out = {}
    for name, d in [("Split 2 Playoffs (LAN)", A), ("Pro League second half", Bd)]:
        x, se, n = shifts(d); keep = n >= 6
        tau2 = max(0.0, float(np.var(x[keep]) - np.mean(se[keep] ** 2)))
        out[name] = dict(teams=int(keep.sum()), sd_shift=float(np.sqrt(tau2)), sd_raw=float(np.std(x[keep])), noise=float(np.sqrt(np.mean(se[keep] ** 2))))
    print(json.dumps(out, indent=1)); json.dump(out, open("out/form_drift.json", "w"), indent=1)
