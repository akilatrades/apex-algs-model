"""Boom-or-bust: does a team's finish swing more (or less) than its rating alone says?

For every game a team played, simulate that exact lobby from the fitted model and see where its actual placement points
land against what the model expected: z = (actual - expected) / spread. If the model's spread is right, z^2 averages 1.
A team averaging well above 1 has booms and busts beyond the model; below 1 is steadier than the model.
Averages over few games are noisy, so each team's figure is shrunk toward 1 by how much real team-to-team difference
there is (estimated from the data). The result is turned into a "noise scale" for the team in the simulations,
with the team's strength re-centred so its average points per game do not change - only how spread out its results are.
"""
import numpy as np
import pandas as pd
import model as M

PP = M.PTS.astype(float)   # placement points by place


def lobby_moments(u, n, rng, scale=None):
    """u: utilities of one lobby. Returns per-team mean and sd of placement points, and sims of placements."""
    T = len(u); s = np.ones(T) if scale is None else scale
    z = u[None, :] + s[None, :] * rng.gumbel(size=(n, T))
    place = np.empty((n, T), int)
    place[np.arange(n)[:, None], np.argsort(-z, 1)] = np.arange(1, T + 1)
    pts = PP[place]
    return pts.mean(0), pts.std(0) + 1e-9, place


def team_z2(df_u, n=400, seed=3):
    """df_u: games table with utility column 'u'. Returns per team: games, mean z^2, and the model's own sd of mean z^2."""
    rng = np.random.default_rng(seed)
    rows = []
    for gid, g in df_u.groupby("game_id"):
        mu, sd, place = lobby_moments(g.u.to_numpy(), n, rng)
        z = (PP[g.placement.to_numpy()] - mu) / sd
        zsim = (PP[place] - mu) / sd                      # what z looks like when the model is exactly right
        rows.append(pd.DataFrame(dict(team=g.team.to_numpy(), z2=z ** 2, z4=(zsim ** 4).mean(0))))
    r = pd.concat(rows)
    out = r.groupby("team").agg(n=("z2", "size"), m=("z2", "mean"), k4=("z4", "mean"))
    out["noise_var"] = (out.k4 - 1) / out.n            # variance of a mean of n z^2 values under the model
    return out


def shrink(t):
    """Empirical Bayes: how much of the spread in teams' mean z^2 is real?"""
    e = t.m - 1
    w = t.n / t.n.sum()
    tau2 = max(0.0, float((w * e ** 2).sum() - (w * t.noise_var).sum()))
    t = t.copy()
    t["excess"] = tau2 / (tau2 + t.noise_var) * e
    return t, tau2


def scale_curve(u_lobby, focus, grid, n=6000, seed=5):
    """For a team at position `focus` in a typical lobby: mean z^2 (vs the s=1 model) as its noise scale changes,
    with its strength shifted to keep expected placement points the same. Returns arrays (grid, ez2, theta_shift)."""
    rng = np.random.default_rng(seed)
    mu0, sd0, _ = lobby_moments(u_lobby, n, rng)
    target = mu0[focus]
    ez2, shifts = [], []
    for s in grid:
        sc = np.ones(len(u_lobby)); sc[focus] = s
        lo, hi = -2.0, 2.0                                 # find shift keeping mean placement points
        for _ in range(18):
            mid = (lo + hi) / 2; u = u_lobby.copy(); u[focus] += mid
            m, _, _ = lobby_moments(u, n, np.random.default_rng(seed + 1), sc)
            lo, hi = (mid, hi) if m[focus] < target else (lo, mid)
        sh = (lo + hi) / 2; u = u_lobby.copy(); u[focus] += sh
        _, _, place = lobby_moments(u, n, np.random.default_rng(seed + 2), sc)
        ez2.append(float((((PP[place[:, focus]] - mu0[focus]) / sd0[focus]) ** 2).mean())); shifts.append(sh)
    return np.asarray(grid), np.asarray(ez2), np.asarray(shifts)


def to_scale(excess, grid, ez2):
    """Map shrunk excess (mean z^2 - 1) to a noise scale by interpolating the curve."""
    return np.interp(1 + np.asarray(excess), ez2, grid)


def label(excess, tau, center=0.0):
    """Plain label from the shrunk excess, relative to the field's average (center). tau = sd of real team-to-team differences."""
    if tau <= 0:
        return "Typical"
    z = (excess - center) / tau
    return "Boom or bust" if z > 0.75 else ("Steady" if z < -0.75 else "Typical")
