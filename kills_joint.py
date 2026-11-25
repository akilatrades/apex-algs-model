"""Kills and placement together. Kills are already drawn given the finishing spot (more kills the longer you live) plus a
team kill-skill. Two checks on held-out games (same splits as backtest3):
  1. spread: are kills more lumpy than a plain Poisson count? (negative binomial, one extra 'lumpiness' number)
  2. style: do some teams get their kills early (fight-heavy, die early) and others late? (per-team slope on placement)
Score = held-out log-likelihood of the kill counts per game, gain over the current model. Writes out/kills_joint.json."""
import json
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln
import backtest3 as BT, model as M


def nb_ll(y, mu, r):
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu))).sum()


def pois_ll(y, mu):
    return (y * np.log(mu) - mu - gammaln(y + 1)).sum()


def fit_style(df, base, ks, sd=0.05):
    """per-team slope s_t on (placement-10.5)/10, shrunk to 0, with base and kill skill held fixed."""
    teams = sorted(df.team.unique()); idx = {t: i for i, t in enumerate(teams)}
    T = df.team.map(idx).values; z = (df.placement.values - 10.5) / 10
    eta0 = base[df.placement.values] + df.team.map(ks).fillna(0).values; y = df.kills.values

    def f(s):
        eta = eta0 + s[T] * z; mu = np.exp(eta)
        return (mu - y * eta).sum() + (s ** 2).sum() / (2 * sd ** 2), np.bincount(T, (mu - y) * z, len(teams)) + s / sd ** 2
    return dict(zip(teams, minimize(f, np.zeros(len(teams)), jac=True, method="L-BFGS-B").x))


if __name__ == "__main__":
    df, stages, reg = BT.prep_all()
    use = df.source.isin(["lan", "pro_league", "pl_s1"])
    testA = df[(df.order >= 2.8) & (df.source == "lan")]; trainA = df[(df.order < 2.8) & use]
    pl = df[df.source == "pro_league"]; day = pl.stage.str.extract(r"_D(\d)")[0].astype(float)
    testB = pl[(day >= 7) | pl.stage.str.endswith("FINALS")]
    trainB = df[use & ((df.order < 2.5) | ((df.source == "pro_league") & df.game_id.isin(pl[day <= 6].game_id)))]
    out = {}
    for name, tr, te in [("lan", trainA, testA), ("pl", trainB, testB)]:
        base, ks = M.fit_kills(tr, sigma=0.15)
        mu_tr = np.exp(base[tr.placement.values] + tr.team.map(ks).fillna(0).values)
        mu_te = np.exp(base[te.placement.values] + te.team.map(ks).fillna(0).values)
        y_tr, y_te = tr.kills.values, te.kills.values
        r = minimize_scalar(lambda lr: -nb_ll(y_tr, mu_tr, np.exp(lr)), bounds=(-1, 6), method="bounded").x
        r = float(np.exp(r)); n = te.game_id.nunique()
        p0 = pois_ll(y_te, mu_te)
        res = dict(games=n, var_over_mean_train=round(float(((y_tr - mu_tr) ** 2).sum() / mu_tr.sum()), 3), nb_r=round(r, 2),
                   nb_gain=round((nb_ll(y_te, mu_te, r) - p0) / n, 4))
        for sd in (0.05, 0.1, 0.2):
            s = fit_style(tr, base, ks, sd)
            z = (te.placement.values - 10.5) / 10
            mu2 = mu_te * np.exp(te.team.map(s).fillna(0).values * z)
            res[f"style_gain_sd{sd}"] = round((pois_ll(y_te, mu2) - p0) / n, 4)
        out[name] = res; print(name, res, flush=True)
    json.dump(out, open("out/kills_joint.json", "w"), indent=1)
