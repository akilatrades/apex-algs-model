"""
Apex Legends team model:
  1. Plackett-Luce team strengths (full placement order of each 20-team lobby),
     MAP-estimated with a Gaussian prior (shrinkage) and recency weighting.
  2. Kills model: Poisson GLM, log E[kills] = baseline(placement) + team kill-skill,
     with a Gaussian prior on the team effects.
  3. Monte Carlo simulator: sample placements from Plackett-Luce (Gumbel trick),
     kills from the kills model, score with ALGS points, play Match Point finals.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from load import PLACE_PTS

PTS = np.array([0] + [PLACE_PTS.get(p, 0) for p in range(1, 21)])  # index by placement


# ---------- 1. Plackett-Luce ----------
def fit_pl(df, sigma=0.6, half_life=None, teams=None, groups=None, tau=1.0, return_groups=False):
    """Return dict team -> strength (log-worth). sigma = prior sd; half_life in stages.
    groups: optional dict team -> group label. Then strength = group mean + team deviation, with the deviation
    shrunk toward 0 (sd sigma) and group means weakly shrunk (sd tau). A team with little data is pulled toward
    the average of ITS group (e.g. its region and level), not toward the global average."""
    teams = teams or sorted(df.team.unique())
    idx = {t: i for i, t in enumerate(teams)}
    if groups is not None:
        glabels = sorted({groups.get(t, "other") for t in teams})
        gidx = np.array([glabels.index(groups.get(t, "other")) for t in teams])
        nT, nG = len(teams), len(glabels)
    max_order = df.order.max()
    games = []
    for gid, g in df.groupby("game_id"):
        g = g.sort_values("placement")
        w = 1.0 if not half_life else 0.5 ** ((max_order - g.order.iloc[0]) / half_life)
        if "weight" in g and pd.notna(g["weight"].iloc[0]):
            w *= float(g["weight"].iloc[0])
        games.append((np.array([idx[t] for t in g.team]), w))

    def negpost(theta_in):
        if groups is not None:
            delta, mu = theta_in[:nT], theta_in[nT:]
            theta = delta + mu[gidx]
        else:
            theta = theta_in
        nll, grad = 0.0, np.zeros_like(theta)
        for order, w in games:
            th = theta[order]
            # log-sum-exp over the "remaining" sets (suffixes)
            m = th.max()
            e = np.exp(th - m)
            suf = np.cumsum(e[::-1])[::-1]            # sum_{j>=i} e_j
            nll -= w * (th[:-1].sum() - (np.log(suf[:-1]) + m).sum())
            # gradient: d/dθ_k = 1[k not last] - sum_{i<=pos(k), i<n-1} e_k/suf_i
            inv = np.cumsum(1.0 / suf[:-1])          # sum over stages i up to k
            inv = np.append(inv, inv[-1])
            gk = np.ones(len(th)); gk[-1] = 0
            gk -= e * inv
            np.add.at(grad, order, -w * gk)
        if groups is not None:
            nll += (delta ** 2).sum() / (2 * sigma ** 2) + (mu ** 2).sum() / (2 * tau ** 2)
            return nll, np.concatenate([grad + delta / sigma ** 2, np.bincount(gidx, grad, nG) + mu / tau ** 2])
        nll += (theta ** 2).sum() / (2 * sigma ** 2)
        grad += theta / sigma ** 2
        return nll, grad

    if groups is not None:
        res = minimize(negpost, np.zeros(nT + nG), jac=True, method="L-BFGS-B")
        th = res.x[:nT] + res.x[nT:][gidx]
        shift = th.mean()
        if return_groups:
            return dict(zip(teams, th - shift)), dict(zip(glabels, res.x[nT:] - shift))
        return dict(zip(teams, th - shift))
    res = minimize(negpost, np.zeros(len(teams)), jac=True, method="L-BFGS-B")
    th = res.x - res.x.mean()
    return dict(zip(teams, th))


def pl_loglik(df, strength):
    """Log-likelihood of observed full orderings under given strengths (unknown teams -> 0)."""
    ll = 0.0
    for _, g in df.groupby("game_id"):
        th = np.array([strength.get(t, 0.0) for t in g.sort_values("placement").team])
        m = th.max(); suf = np.cumsum(np.exp(th - m)[::-1])[::-1]
        ll += th[:-1].sum() - (np.log(suf[:-1]) + m).sum()
    return ll


# ---------- 2. Kills model ----------
SMOOTH = 5.0   # sd of the smoothness penalty on the placement baseline; large = follow the data (0.3 over-smoothed 1st place)
def fit_kills(df, sigma=0.3, teams=None):
    """Poisson: log mu = b[placement] + k[team]. Returns (baseline array idx by placement, team dict)."""
    teams = teams or sorted(df.team.unique())
    idx = {t: i for i, t in enumerate(teams)}
    P = df.placement.values - 1
    T = df.team.map(idx).values
    y = df.kills.values
    nP, nT = 20, len(teams)

    def negpost(x):
        b, k = x[:nP], x[nP:]
        eta = b[P] + k[T]
        mu = np.exp(eta)
        nll = (mu - y * eta).sum() + (k ** 2).sum() / (2 * sigma ** 2)
        # light smoothness prior on baseline across adjacent placements
        db = np.diff(b)
        nll += (np.diff(db) ** 2).sum() / (2 * SMOOTH ** 2)
        r = mu - y
        gb = np.bincount(P, r, nP)
        gk = np.bincount(T, r, nT) + k / sigma ** 2
        # gradient of smoothness term
        d2 = np.diff(db)
        gs = np.zeros(nP)
        gs[:-2] += d2; gs[1:-1] -= 2 * d2; gs[2:] += d2
        gb += gs / SMOOTH ** 2
        return nll, np.concatenate([gb, gk])

    x0 = np.zeros(nP + nT); x0[:nP] = np.log(df.groupby("placement").kills.mean().reindex(range(1, 21)).fillna(0.5).clip(0.2).values)
    res = minimize(negpost, x0, jac=True, method="L-BFGS-B")
    b, k = res.x[:nP], res.x[nP:]
    return np.concatenate([[0], b]), dict(zip(teams, k))


# ---------- 3. Simulation ----------
def sample_games(theta, kskill, base, n, rng):
    """theta, kskill: arrays for the lobby. Returns placements (n,T), kills (n,T)."""
    T = len(theta)
    g = theta[None, :] + rng.gumbel(size=(n, T))
    ranks = np.argsort(-g, axis=1)
    place = np.empty_like(ranks)
    place[np.arange(n)[:, None], ranks] = np.arange(1, T + 1)[None, :]
    mu = np.exp(base[place] + kskill[None, :])
    kills = rng.poisson(mu)
    return place, kills


def simulate_match_point(theta, kskill, base, n_sims=20000, threshold=50, max_games=30, seed=0):
    """Returns P(win), expected final standing points after 6 games, P(reach match point)."""
    rng = np.random.default_rng(seed)
    T = len(theta)
    wins = np.zeros(T)
    reached = np.zeros(T)
    games_played = np.zeros(n_sims)
    pts_total = np.zeros((n_sims, T))
    alive = np.ones(n_sims, bool)
    for gi in range(max_games):
        idx = np.where(alive)[0]
        if len(idx) == 0:
            break
        place, kills = sample_games(theta, kskill, base, len(idx), rng)
        eligible = pts_total[idx] >= threshold          # on match point at start of game
        winner_mask = (place == 1) & eligible
        has_w = winner_mask.any(1)
        wteam = winner_mask.argmax(1)
        wins += np.bincount(wteam[has_w], minlength=T)
        pts_total[idx] += PTS[place] + kills
        games_played[idx] += 1
        alive[idx[has_w]] = False
    reached = (pts_total >= threshold).mean(0)
    return dict(p_win=wins / n_sims, p_reach_mp=reached,
                avg_games=games_played.mean(), unresolved=alive.mean())


def expected_points_per_game(theta, kskill, base, n=40000, seed=1):
    rng = np.random.default_rng(seed)
    place, kills = sample_games(theta, kskill, base, n, rng)
    return dict(exp_pts=(PTS[place] + kills).mean(0), exp_place=place.mean(0),
                p_win_game=(place == 1).mean(0), p_top5=(place <= 5).mean(0),
                exp_kills=kills.mean(0))
