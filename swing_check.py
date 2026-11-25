"""Held-out check for boom-or-bust (see swing.py): does giving swingy teams a wider spread in the simulations predict
finishing positions better? Writes out/swing_check.json. Result so far: no (it predicts worse), so it is a label only."""
# Validate boom-or-bust on held-out data: train-fitted model + swing (from train games only) -> score test placements.
import numpy as np, pandas as pd, time
from backtest3 import *
import swing as SW
t0=time.time()
df, stages, reg = prep_all()
use = df.source.isin(["lan","pro_league","pl_s1"])
KW=dict(sig_team=0.15, sig_player=0.20, fit_beta=(True, True))
def evaluate(train, test, lanset, debut):
    F = run_fit(train, lanset, reg, **KW)
    tr = train.copy(); tr["u"] = M3.utilities(F, tr)
    t = SW.team_z2(tr); t, tau2 = SW.shrink(t)
    te = test.copy(); te["u"] = M3.utilities(F, te)
    if debut:
        _, b = score(F, test, debut); te["u"] -= te.team.isin(debut) * b
    # curve on a typical test lobby, middle team
    g0 = te[te.game_id == te.game_id.iloc[0]]; u0 = np.sort(g0.u.to_numpy())[::-1]
    grid = np.linspace(0.6, 1.8, 13); gr, ez2, sh = SW.scale_curve(u0, len(u0)//2, grid, n=3000)
    sc = dict(zip(t.index, SW.to_scale(t.excess, gr, ez2))); shift = dict(zip(t.index, np.interp(list(sc.values()), gr, sh)))
    rng = np.random.default_rng(9); ll0 = ll1 = 0; N = 4000
    for gid, g in te.groupby("game_id"):
        u = g.u.to_numpy(); s = g.team.map(sc).fillna(1.0).to_numpy(); u1 = u + g.team.map(shift).fillna(0.0).to_numpy()
        for uu, ss, which in [(u, None, 0), (u1, s, 1)]:
            _, _, place = SW.lobby_moments(uu, N, rng, ss)
            p = ((place == g.placement.to_numpy()[None, :]).sum(0) + 0.5) / (N + 10)
            if which == 0: ll0 += np.log(p).sum()
            else: ll1 += np.log(p).sum()
    n = te.game_id.nunique()
    return dict(tau=float(np.sqrt(tau2)), base=ll0/n, swing=ll1/n, gain=(ll1-ll0)/n, scales=pd.Series(sc).describe().round(2).to_dict())
testA = df[(df.order >= 2.8) & (df.source == "lan")]; trainA = df[(df.order < 2.8) & use]
lanA = set(trainA[trainA.source=="lan"].team) | set(testA.team); debut = set(testA.team) - set(trainA[trainA.source=="lan"].team)
rA = evaluate(trainA, testA, lanA, debut); print("LAN", rA, flush=True)
pl = df[df.source=="pro_league"].copy(); day = pl.stage.str.extract(r"_D(\d)")[0].astype(float)
testB = pl[(day>=7)|pl.stage.str.endswith("FINALS")]; trainB = df[use & ((df.order<2.5)|((df.source=="pro_league")&df.game_id.isin(pl[day<=6].game_id)))]
rB = evaluate(trainB, testB, set(trainB[trainB.source=="lan"].team), None); print("PL", rB)
import json; json.dump(dict(lan_gain=float(rA["gain"]), pl_gain=float(rB["gain"])), open("out/swing_check.json", "w"))

# Direct persistence check: does a team's swing in the training games predict its swing in the test games?
from scipy.stats import spearmanr, pearsonr
for name, train, test, lanset in [("LAN", trainA, testA, lanA), ("PL", trainB, testB, set(trainB[trainB.source=="lan"].team))]:
    F = run_fit(train, lanset, reg, **KW)
    tr = train.copy(); tr["u"] = M3.utilities(F, tr); te = test.copy(); te["u"] = M3.utilities(F, te)
    a = SW.team_z2(tr); b = SW.team_z2(te); j = a.join(b, lsuffix="_tr", rsuffix="_te").dropna()
    j = j[(j.n_tr >= 20) & (j.n_te >= 6)]
    print(name, "teams", len(j), "pearson", pearsonr(j.m_tr, j.m_te), "spearman", spearmanr(j.m_tr, j.m_te))
    # win-rate check: top-heavy teams?  share of points from wins vs expectation
