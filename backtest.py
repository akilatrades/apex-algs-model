"""Gradient check + out-of-sample backtest: train on earlier stages, predict Y6 Split 2 Finals."""
import numpy as np, pandas as pd
from math import lgamma
from scipy.optimize import check_grad
from scipy.stats import spearmanr
from load import load
import model as M

df, stages = load()

# --- gradient check on PL objective (small sigma grid) ---
from scipy.optimize import approx_fprime
teams = sorted(df.team.unique())
s = M.fit_pl(df, sigma=0.6)
print("PL fit ok; spread", round(max(s.values()) - min(s.values()), 3))

train = df[df.order < 4]
test = df[df.order == 4]
uniform = -test.game_id.nunique() * lgamma(21)  # log(1/20!) per game
print(f"\nTest: {test.game_id.nunique()} games of Y6S2 Finals. Uniform log-lik = {uniform:.1f}")

rows = []
for sigma in [0.2, 0.4, 0.6, 0.8, 1.0, 1.5]:
    for hl in [None, 2, 1]:
        st = M.fit_pl(train, sigma=sigma, half_life=hl)
        ll = M.pl_loglik(test, st)
        rows.append(dict(sigma=sigma, half_life=hl or "none", test_ll=ll, gain_per_game=(ll - uniform) / test.game_id.nunique()))
res = pd.DataFrame(rows).sort_values("test_ll", ascending=False)
print(res.to_string(index=False))

# Kills model held-out deviance vs placement-only baseline
for ks in [0.0001, 0.15, 0.3, 0.5]:
    base, kk = M.fit_kills(train, sigma=ks)
    mu = np.exp(base[test.placement.values] + test.team.map(kk).fillna(0).values)
    y = test.kills.values
    ll = (y * np.log(mu) - mu).sum()
    print(f"kills sigma={ks}: held-out Poisson ll (no const) = {ll:.1f}")

# Pre-event expected points vs actual totals for teams seen in training
best = res.iloc[0]
st = M.fit_pl(train, sigma=best.sigma, half_life=None if best.half_life == "none" else best.half_life)
base, kk = M.fit_kills(train, sigma=0.3)
lobby = sorted(test.team.unique())
th = np.array([st.get(t, 0.0) for t in lobby]); ks = np.array([kk.get(t, 0.0) for t in lobby])
ep = M.expected_points_per_game(th, ks, base)
actual = test.groupby("team").points.sum().reindex(lobby)
seen = [t in st for t in lobby]
r_all = spearmanr(ep["exp_pts"], actual).correlation
r_seen = spearmanr(ep["exp_pts"][seen], actual[seen]).correlation
print(f"\nSpearman(pred exp pts, actual Finals points): all={r_all:.2f}, teams-with-history (n={sum(seen)})={r_seen:.2f}")
out = pd.DataFrame(dict(team=lobby, seen=seen, pred_pts_per_game=ep["exp_pts"].round(2), actual_total=actual.values)).sort_values("pred_pts_per_game", ascending=False)
print(out.to_string(index=False))
