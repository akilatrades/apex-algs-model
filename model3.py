"""
Model v3: one Plackett-Luce fit with extra pieces, each shrunk toward "no effect" unless the data supports it.

  utility of team t in a game = group mean            (region x LAN/regional, as in v2)
                              + team deviation         (sd SIG_TEAM)
                              + mean of its players'   (sd SIG_PLAYER; follows players across teams)
                                strengths
                              + team-on-this-map       (sd SIG_MAP)
                              + beta_clash x [landed on a spot shared with another team this game]
                              + beta_mp    x [on match point at the start of this game]
  finishing order ~ Plackett-Luce(utilities)

Every sd/beta is chosen or fitted on data before the Y6 Split 2 Playoffs and checked on that event (analysis3.py).
"""
import numpy as np
from scipy.optimize import minimize

FEATURES = ["clash", "mp"]


def prepare(df, teams, players, groups, maps):
    """Pre-pack games for fast likelihood evaluation.
    df needs: game_id, placement, team, map, order, weight (optional), roster (tuple of player ids or ()), clash, mp."""
    ti = {t: i for i, t in enumerate(teams)}
    pi = {p: i for i, p in enumerate(players)}
    mi = {m: i for i, m in enumerate(maps)}
    glab = sorted(set(groups.values()))
    gidx = np.array([glab.index(groups[t]) for t in teams])
    G = []
    for gid, g in df.groupby("game_id", sort=False):
        g = g.sort_values("placement")
        w = float(g["weight"].iloc[0]) if "weight" in g and g["weight"].notna().iloc[0] else 1.0
        T = np.array([ti[t] for t in g.team])
        Mx = np.array([mi.get(m, -1) for m in g["map"]])
        X = g[FEATURES].to_numpy(float) if all(f in g for f in FEATURES) else np.zeros((len(g), len(FEATURES)))
        # players: (row, player) pairs with weight 1/k
        pr, pp, pw = [], [], []
        for r, ros in enumerate(g["roster"] if "roster" in g else [()] * len(g)):
            ids = [pi[p] for p in ros if p in pi]
            for p in ids:
                pr.append(r); pp.append(p); pw.append(1.0 / len(ids))
        G.append((T, Mx, X, np.array(pr, int), np.array(pp, int), np.array(pw), w))
    return dict(G=G, nT=len(teams), nP=len(players), nM=len(maps), nG=len(glab), gidx=gidx, glab=glab, teams=teams,
                players=players, maps=maps)


def fit(P, sig_team=0.15, sig_player=0.0, sig_map=0.0, tau=1.0, fit_beta=(True, True), beta_sd=1.0, x0=None):
    nT, nP, nM, nG, gidx = P["nT"], P["nP"], P["nM"], P["nG"], P["gidx"]
    use_p, use_m = sig_player > 0 and nP > 0, sig_map > 0 and nM > 0
    nF = len(FEATURES)
    sizes = [nT, nG, nP if use_p else 0, nT * nM if use_m else 0, nF]
    cut = np.cumsum([0] + sizes)

    def unpack(x):
        return [x[cut[i]:cut[i + 1]] for i in range(5)]

    fb = np.array(fit_beta, bool)

    def negpost(x):
        d, mu, p, m, b = unpack(x)
        b = np.where(fb, b, 0.0)
        theta = d + mu[gidx]
        gd = np.zeros(nT); gp = np.zeros(nP) if use_p else None
        gm = np.zeros(nT * nM) if use_m else None; gb = np.zeros(nF)
        nll = 0.0
        for T, Mx, X, pr, pp, pw, w in P["G"]:
            u = theta[T] + X @ b
            if use_m:
                mm = Mx >= 0
                mk = T * nM + np.where(mm, Mx, 0)
                u = u + np.where(mm, m[mk], 0.0)
            if use_p and len(pr):
                u = u + np.bincount(pr, pw * p[pp], len(T))
            mx = u.max(); e = np.exp(u - mx)
            suf = np.cumsum(e[::-1])[::-1]
            nll -= w * (u[:-1].sum() - (np.log(suf[:-1]) + mx).sum())
            inv = np.cumsum(1.0 / suf[:-1]); inv = np.append(inv, inv[-1])
            gk = np.ones(len(u)); gk[-1] = 0; gk -= e * inv
            gk = -w * gk                                   # d nll / d u
            np.add.at(gd, T, gk)
            gb += X.T @ gk
            if use_m:
                np.add.at(gm, mk[mm], gk[mm])
            if use_p and len(pr):
                np.add.at(gp, pp, gk[pr] * pw)
        gmu = np.bincount(gidx, gd, nG)
        nll += (d ** 2).sum() / (2 * sig_team ** 2) + (mu ** 2).sum() / (2 * tau ** 2) + (b[fb] ** 2).sum() / (2 * beta_sd ** 2)
        grad = [gd + d / sig_team ** 2, gmu + mu / tau ** 2]
        if use_p:
            nll += (p ** 2).sum() / (2 * sig_player ** 2); grad.append(gp + p / sig_player ** 2)
        else:
            grad.append(np.zeros(0))
        if use_m:
            nll += (m ** 2).sum() / (2 * sig_map ** 2); grad.append(gm + m / sig_map ** 2)
        else:
            grad.append(np.zeros(0))
        grad.append(np.where(fb, gb + b / beta_sd ** 2, 0.0))
        return nll, np.concatenate(grad)

    res = minimize(negpost, np.zeros(cut[-1]) if x0 is None else x0, jac=True, method="L-BFGS-B", options=dict(maxiter=3000))
    d, mu, p, m, b = unpack(res.x)
    out = dict(x=res.x, ok=bool(res.success), team=d + mu[gidx], group=dict(zip(P["glab"], mu)), dev=d,
               player=dict(zip(P["players"], p)) if use_p else {},
               map=(m.reshape(nT, nM) if use_m else np.zeros((nT, nM))), beta=dict(zip(FEATURES, np.where(fb, b, 0.0))),
               teams=P["teams"], maps=P["maps"])
    return out


def utilities(F, df, roster_override=None):
    """Utility per row of df (one game per game_id) under fit F. Unknown teams -> 0 + their players."""
    ti = {t: i for i, t in enumerate(F["teams"])}; mi = {m: i for i, m in enumerate(F["maps"])}
    u = np.zeros(len(df))
    for k, r in enumerate(df.itertuples()):
        i = ti.get(r.team)
        v = F["team"][i] if i is not None else 0.0
        if i is not None and r.map in mi:
            v += F["map"][i, mi[r.map]]
        ros = getattr(r, "roster", ()) or ()
        ps = [F["player"][p] for p in ros if p in F["player"]]
        if ps:
            v += np.mean(ps)
        v += F["beta"]["clash"] * getattr(r, "clash", 0) + F["beta"]["mp"] * getattr(r, "mp", 0)
        u[k] = v
    return u


def loglik(F, df, shift=None):
    """Held-out Plackett-Luce log-likelihood. shift: optional dict team -> extra utility (e.g. debut discount)."""
    df = df.copy(); df["u"] = utilities(F, df)
    if shift:
        df["u"] += df.team.map(shift).fillna(0.0)
    ll = 0.0
    for _, g in df.groupby("game_id"):
        u = g.sort_values("placement").u.to_numpy()
        mx = u.max(); suf = np.cumsum(np.exp(u - mx)[::-1])[::-1]
        ll += u[:-1].sum() - (np.log(suf[:-1]) + mx).sum()
    return ll
