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
print("\nALL MATHS CHECKS PASS" if ok else "\nSOME CHECKS FAILED")
