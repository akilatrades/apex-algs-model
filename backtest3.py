"""Which v3 pieces actually predict better? Two held-out checks, same as v2's:
  A. train on everything before the Y6 Split 2 Playoffs -> predict that LAN (63 games)
  B. Split 2 Pro League: train on everything up to day 6 -> predict days 7-9 and the finals
Score = average log-likelihood gain per game over "every finishing order equally likely" (higher is better).
Writes out/backtest3.json."""
import json, sys, time
from math import lgamma
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from load import load
import features as FE, model3 as M3

REG = {"NA": "Americas", "EMEA": "EMEA", "APN": "APAC North", "APS": "APAC South"}


def region_of(stage):
    for k in REG:
        if stage.startswith((f"PL_{k}_", f"LCQ_{k}_", f"PL1_{k}_")):
            return REG[k]
    return None


def prep_all():
    df, stages = load()
    df = FE.attach(df, stages)
    df["region"] = df.stage.map(region_of)
    reg = df.dropna(subset=["region"]).groupby("team").region.agg(lambda s: s.mode().iloc[0])
    return df, stages, reg


def groups(teams, lanset, reg):
    return {t: f"{reg.get(t, 'none')}|{'LAN' if t in lanset else 'regional'}" for t in teams}


def uni(d):
    return sum(-lgamma(len(g) + 1) for _, g in d.groupby("game_id"))


def players_of(d):
    return sorted({p for r in d.roster for p in r})


def run_fit(train, lanset, reg, **kw):
    teams = sorted(train.team.unique())
    P = M3.prepare(train, teams, players_of(train), groups(teams, lanset, reg), sorted(m for m in train["map"].unique() if m != "unknown"))
    return M3.fit(P, **kw)


def score(F, test, debut=None, beta=None):
    """Gain per game; if debut given, fits the LAN debut discount on the test set the same way v2 does."""
    n = test.game_id.nunique()
    if debut:
        if beta is None:
            beta = minimize_scalar(lambda b: -M3.loglik(F, test, {t: -b for t in debut}), bounds=(-0.5, 1), method="bounded").x
        ll = M3.loglik(F, test, {t: -beta for t in debut})
    else:
        ll = M3.loglik(F, test)
    return (ll - uni(test)) / n, beta


CONFIGS = {   # label shown on the page -> settings
    "Before (Split 2 model)":            dict(sig_team=0.15, fit_beta=(False, False)),
    "+ team strength by map":            dict(sig_team=0.15, sig_map=0.10, fit_beta=(False, False)),
    "+ ratings follow players":          dict(sig_team=0.15, sig_player=0.20, fit_beta=(False, False)),
    "+ shared landing spot":             dict(sig_team=0.15, fit_beta=(True, False)),
    "+ match point":                     dict(sig_team=0.15, fit_beta=(False, True)),
    "New model (players + spot + match point)": dict(sig_team=0.15, sig_player=0.20, fit_beta=(True, True)),
}
SPREAD_TRIALS = {f"spread {s}": dict(sig_team=s, sig_player=p, fit_beta=(True, True))
                 for s, p in [(0.2, 0.2), (0.25, 0.2), (0.3, 0.2), (0.25, 0.3), (0.35, 0.25)]}

if __name__ == "__main__":
    t0 = time.time()
    df, stages, reg = prep_all()
    use = df.source.isin(["lan", "pro_league", "pl_s1"])
    # A: Split 2 Playoffs
    testA = df[(df.order >= 2.8) & (df.source == "lan")]
    trainA = df[(df.order < 2.8) & use]
    lanA = set(trainA[trainA.source == "lan"].team) | set(testA.team)
    debut = set(testA.team) - set(trainA[trainA.source == "lan"].team)
    # B: Pro League second half
    pl = df[df.source == "pro_league"].copy(); day = pl.stage.str.extract(r"_D(\d)")[0].astype(float)
    testB = pl[(day >= 7) | pl.stage.str.endswith("FINALS")]
    trainB = df[use & ((df.order < 2.5) | ((df.source == "pro_league") & df.game_id.isin(pl[day <= 6].game_id)))]
    lanB = set(trainB[trainB.source == "lan"].team)
    ALL = {**CONFIGS, **SPREAD_TRIALS}
    only = sys.argv[1:] or list(CONFIGS)
    res = []
    for name in only:
        kw = ALL[name]
        FA = run_fit(trainA, lanA, reg, **kw); gA, bA = score(FA, testA, debut)
        FB = run_fit(trainB, lanB, reg, **kw); gB, _ = score(FB, testB)
        r = dict(config=name, lan_gain=round(gA, 4), pl_gain=round(gB, 4), beta_debut=round(float(bA), 3),
                 clash=round(float(FA["beta"]["clash"]), 3), mp=round(float(FA["beta"]["mp"]), 3), ok=FA["ok"] and FB["ok"])
        res.append(r); print(r, f"{time.time() - t0:.0f}s", flush=True)
    if not sys.argv[1:]:
        json.dump(res, open("out/backtest3.json", "w"), indent=1)
