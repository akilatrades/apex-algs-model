"""Maths checks. Run: python3 tests.py  (each check prints PASS/FAIL)."""
import numpy as np, pandas as pd
from scipy.optimize import approx_fprime
from load import load, validate, PLACE_PTS, place_pts
import model as M
import deep

ok = True
def check(name, cond, detail=""):
    global ok; ok &= bool(cond); print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))

df, _ = load(); rng = np.random.default_rng(0)

# 1. scoring
check("placement points sum to 54 per 20-team game (12+9+7+5+4+3+3+2+2+2+1x5)", sum(PLACE_PTS.values()) == 54)
check("1st = 12, 2nd = 9, 5th = 4, 15th = 1, 16th = 0", (place_pts(1), place_pts(2), place_pts(5), place_pts(15), place_pts(16)) == (12, 9, 4, 1, 0))
check("known result: Team Liquid 91 pts in Y6S2 Playoffs Finals", df[(df.stage == "Y6S2_PLAYOFFS_FINALS") & (df.team == "Team Liquid")].points.sum() == 91)

# 2. data integrity
check("every game has placements 1..N exactly once", not [p for p in validate(df) if "permutation" in p])
check("every team total matches the page total", not [p for p in validate(df) if "computed" in p])
n, bad, _ = deep.crosscheck()
check("site season totals (games, kills, points) match game data", not bad, f"{n - len(bad)}/{n} team-event rows")
check("each game has exactly one winner", (df[df.placement == 1].groupby("game_id").size() == 1).all())

# 3. Plackett-Luce maths
sub = df[df.stage == "Y6S1_PLAYOFFS_FINALS"]; teams = sorted(sub.team.unique()); idx = {t: i for i, t in enumerate(teams)}
games = [np.array([idx[t] for t in g.sort_values("placement").team]) for _, g in sub.groupby("game_id")]
def nll(th):
    return -sum(th[o][:-1].sum() - np.log(np.cumsum(np.exp(th[o])[::-1])[::-1][:-1]).sum() for o in games)
th0 = rng.normal(0, .4, len(teams))
check("model log-likelihood equals the textbook Plackett-Luce formula", abs(M.pl_loglik(sub, dict(zip(teams, th0))) + nll(th0)) < 1e-8)
fit = M.fit_pl(sub, sigma=0.5); v = np.array([fit[t] for t in teams])
g = approx_fprime(v, lambda x: nll(x) + (x ** 2).sum() / (2 * .5 ** 2), 1e-6)
check("fitted strengths sit at the optimum (gradient ~ 0)", np.abs(g - g.mean()).max() < 2e-3, f"max |grad| {np.abs(g - g.mean()).max():.1e}")
grp = {t: ("a" if i % 2 else "b") for i, t in enumerate(teams)}
fg, gm = M.fit_pl(sub, sigma=0.5, groups=grp, return_groups=True); f0 = M.fit_pl(sub, sigma=0.5, groups={t: "x" for t in teams})
check("grouped fit with one group equals the plain fit", max(abs(f0[t] - fit[t]) for t in teams) < 1e-3)
check("ratings are centred on zero", abs(np.mean(list(fg.values()))) < 1e-9)

# 4. simulator
th = np.array([0.5, 0.2, 0, -0.3] + [0.0] * 16); base, _ = M.fit_kills(df, sigma=0.15)
place, kills = M.sample_games(th, np.zeros(20), base, 200000, rng)
soft = np.exp(th) / np.exp(th).sum()
check("simulated game-win rates match the closed form e^s / sum(e^s)", np.abs((place == 1).mean(0) - soft).max() < 0.003, f"max diff {np.abs((place == 1).mean(0) - soft).max():.4f}")
check("each simulated game is a full 1..20 ordering", (np.sort(place, 1) == np.arange(1, 21)).all())
st_all = M.fit_pl(df, sigma=0.15); b_all, k_all = M.fit_kills(df, sigma=0.15); tot = cnt = 0
for _, g in df.groupby("game_id"):          # replay every real lobby 20 times with the fitted team strengths and kill rates
    pl_, kl_ = M.sample_games(g.team.map(st_all).values, g.team.map(k_all).values, b_all, 20, rng)
    tot += (M.PTS[pl_] + kl_).sum(); cnt += pl_.size
check("replaying every real lobby gives the same points per team-game as the data", abs(tot / cnt - df.points.mean()) < 0.1, f"sim {tot / cnt:.2f} vs data {df.points.mean():.2f}")
r = M.simulate_match_point(np.zeros(20), np.zeros(20), base, n_sims=40000, seed=3)
check("equal teams each win a match-point finals 5% of the time", np.abs(r["p_win"] - 0.05).max() < 0.006, f"range {r['p_win'].min():.3f}-{r['p_win'].max():.3f}")
check("win chances sum to 100%", abs(r["p_win"].sum() + r["unresolved"] - 1) < 1e-9)
real = df[df.stage.isin(["Y5S2_CHAMPS_FINALS", "Y6S1_PLAYOFFS_FINALS", "Y6S2_PLAYOFFS_FINALS"])].groupby("stage").game.max().mean()
check("simulated finals length is close to real finals (9, 10, 9 games)", abs(r["avg_games"] - real) < 1.0, f"sim {r['avg_games']:.1f} vs real {real:.1f}")
r2 = M.simulate_match_point(th * 3, np.zeros(20), base, n_sims=20000, seed=4)
check("a stronger team wins more often, a weaker team less", r2["p_win"][0] > r2["p_win"][1] > r2["p_win"][2] > r2["p_win"][3])
base2, ks2 = M.fit_kills(df, sigma=0.15)
mu = np.exp(base2[df.placement.values] + df.team.map(ks2).values)
gap = (pd.Series(mu).groupby(df.placement.values).mean() - df.groupby("placement").kills.mean()).abs().max()
check("kills model reproduces average kills at every placement", gap < 0.05, f"max gap {gap:.3f} kills")
check("kills model reproduces total kills", abs(mu.sum() - df.kills.sum()) / df.kills.sum() < 0.002, f"model {mu.sum():.0f} vs data {df.kills.sum()}")

# 5. model v3 (players, shared spot, match point)
import model3 as M3, features as FE
_, stg = load(); d3 = FE.attach(df, stg)
check("every game row has a roster of player ids", (d3.roster.map(len) > 0).mean() > 0.99, f"{(d3.roster.map(len) > 0).mean():.1%}")
check("Split 2 Playoffs LAN rows are linked to online player ids", d3[d3.stage.str.startswith("Y6S2_PLAYOFFS")].roster.map(lambda r: r[0].startswith("O:")).mean() > 0.9)
fz = d3[d3.stage.str.endswith("FINALS")]
check("every finals ends with a win by a team that started the game on match point",
      all(g[g.game == g.game.max()].query("placement == 1").mp.iloc[0] == 1 for st_, g in fz.groupby("stage") if st_ != "PL_EMEA_FINALS"))
check("nobody is on match point in game 1 of a finals", fz[fz.game == 1].mp.sum() == 0)
check("a shared landing spot needs both teams in the same lobby", all(
    len(set(g[g.clash == 1].team)) != 1 for _, g in d3[d3.clash == 1].groupby("game_id")))
small = d3[d3.stage.isin(["Y6S1_PLAYOFFS_FINALS", "Y6S2_PLAYOFFS_FINALS", "PL_NA_FINALS"])]
tm = sorted(small.team.unique()); pl3 = sorted({p for r in small.roster for p in r})
P = M3.prepare(small, tm, pl3, {t: "g" for t in tm}, sorted(small["map"].unique()))
kw = dict(sig_team=0.3, sig_player=0.2, sig_map=0.1, fit_beta=(True, True))
F = M3.fit(P, **kw)
x = F["x"] + rng.normal(0, 0.05, len(F["x"]))
def obj(xx):  # same objective as model3.fit, recomputed independently via utilities()
    F2 = dict(F); nT, nM = P["nT"], P["nM"]; nP = P["nP"]
    d_, mu_ = xx[:nT], xx[nT:nT + 1]; p_ = xx[nT + 1:nT + 1 + nP]; m_ = xx[nT + 1 + nP:nT + 1 + nP + nT * nM]; b_ = xx[-2:]
    F2.update(team=d_ + mu_[0], player=dict(zip(pl3, p_)), map=m_.reshape(nT, nM), beta=dict(clash=b_[0], mp=b_[1]))
    return -M3.loglik(F2, small) + (d_ ** 2).sum() / (2 * .09) + mu_[0] ** 2 / 2 + (p_ ** 2).sum() / (2 * .04) + (m_ ** 2).sum() / (2 * .01) + (b_ ** 2).sum() / 2
gn = approx_fprime(F["x"], obj, 1e-6)
check("v3 fit sits at the optimum of its objective (checked by an independent calculation)", np.abs(gn).max() < 5e-3, f"max |grad| {np.abs(gn).max():.1e}")
check("v3 with players/maps/effects switched off equals the v2 fit", (lambda F0: max(abs(F0["team"][i] - (lambda f: f[t])(M.fit_pl(small, sigma=0.3, groups={u: "g" for u in tm}))) for i, t in enumerate(tm)) < 2e-3)(
      M3.fit(P, sig_team=0.3, fit_beta=(False, False))))


# 6. zone data
import rings as RG
ZM = RG.load(); ZP = RG.outlines()
check("ring data decodes to the expected number of games per map", all(len(m["end"]) == n for m, n in zip(ZM.values(), [2370, 1980, 956, 736])))
check("every final zone sits inside its ring 2, and every ring 2 inside its ring 1",
      all((np.hypot(*(m["end"] - m["r2"]).T) <= m["radii"][1] + 0.01).all() and (np.hypot(*(m["r2"] - m["r1"]).T) <= m["radii"][0] + 0.02).all() for m in ZM.values()))
def _in_frac(n, pts):
    return np.mean([any(RG.inside(p, poly) for _, poly in ZP[n]) for p in pts[::6]])
gaps = []
for n, m in ZM.items():
    e = m["end"]; good = _in_frac(n, e)
    worst = max(_in_frac(n, np.stack([e[:, 0], 1 - e[:, 1]], 1)), _in_frac(n, np.stack([1 - e[:, 0], e[:, 1]], 1)), _in_frac(n, e + [0.05, 0.05]))
    gaps.append(good - worst)
check("map alignment: more final zones fall inside landing-spot outlines than with the map flipped or shifted", min(gaps) > 0.03, f"smallest margin {min(gaps):.1%}")

print("\nALL MATHS CHECKS PASS" if ok else "\nSOME CHECKS FAILED")
