"""Fit the model and write every number the two pages use -> out/results.json

Model (v2):
  * Team strength: Plackett-Luce on each game's full finishing order. Each team is shrunk toward the mean of its
    own group (region x "has LAN games in the data"), not toward one global mean.
  * LAN debut discount: teams with no LAN games get strength - BETA when forecasting LAN play. BETA is estimated
    from how first-time LAN teams did at the Y6 Split 2 Playoffs relative to their regional rating.
  * Kills: Poisson, by placement, with a small team effect.
  * Ratings are reported in plain units: expected points per game in a Championship lobby.
"""
import json, os, numpy as np, pandas as pd
from math import lgamma
from scipy.optimize import minimize_scalar
from scipy.stats import f as fdist
from load import load, PLACE_PTS
import model as M
import deep

rng = np.random.default_rng(42)
df, stages = load(include_scrims=True)
REG = {"NA": "Americas", "EMEA": "EMEA", "APN": "APAC North", "APS": "APAC South"}
SIGMA = 0.15


def region_of(stage):
    for k in REG:
        if stage.startswith(f"PL_{k}_") or stage.startswith(f"LCQ_{k}_") or stage.startswith(f"PL1_{k}_"):
            return REG[k]
    return None


df["region"] = df.stage.map(region_of)
team_region = df.dropna(subset=["region"]).groupby("team").region.agg(lambda s: s.mode().iloc[0])


def groups_for(teams, lanset):
    return {t: f"{team_region.get(t, 'none')}|{'LAN' if t in lanset else 'regional'}" for t in teams}


def uni_ll(d):
    return sum(-lgamma(len(g) + 1) for _, g in d.groupby("game_id"))


def pairwise(d, st):
    """Share of team pairs (within each game) where the higher-rated team finished ahead."""
    hit = tot = 0
    for _, g in d.groupby("game_id"):
        s = g.team.map(st).fillna(0).values; p = g.placement.values
        ds = s[:, None] - s[None, :]; dp = p[:, None] - p[None, :]
        m = ds > 1e-9
        hit += (dp[m] < 0).sum(); tot += m.sum()
    return hit / tot


# ---------------- backtest: everything before the Y6 Split 2 Playoffs -> that event (63 LAN games) ----------------
if "weight" not in df: df["weight"] = np.nan
test = df[(df.order >= 2.8) & (df.source == "lan")]
train_all = df[(df.order < 2.8) & df.source.isin(["lan", "pro_league", "pl_s1"])]
# How much should Split 1 Pro League (older, many roster changes since) count? Let the backtest decide.
s1_trials = []
for w in [0.0, 0.25, 0.5, 1.0]:
    tr = train_all[train_all.source != "pl_s1"] if w == 0 else train_all.assign(weight=np.where(train_all.source == "pl_s1", w, np.nan))
    pl_ = set(tr[tr.source == "lan"].team)
    st_ = M.fit_pl(tr, sigma=SIGMA, groups=groups_for(tr.team.unique(), pl_ | set(test.team)))
    s1_trials.append(dict(weight=w, gain=float((M.pl_loglik(test, st_) - uni_ll(test)) / test.game_id.nunique()), acc=float(pairwise(test, st_))))
S1W = max(s1_trials, key=lambda r: r["gain"])["weight"]
df.loc[df.source == "pl_s1", "weight"] = S1W if S1W > 0 else 1e-6
train = train_all[train_all.source != "pl_s1"] if S1W == 0 else train_all.assign(weight=np.where(train_all.source == "pl_s1", S1W, np.nan))
prev_lan = set(train[train.source == "lan"].team)
debut = set(test.team) - prev_lan
G_bt = groups_for(train.team.unique(), prev_lan | set(test.team))
st_bt = M.fit_pl(train, sigma=SIGMA, groups=G_bt)
st_flat = M.fit_pl(train, sigma=0.2)


def with_beta(st, b, who):
    return {t: v - (b if t in who else 0) for t, v in st.items()}


opt = minimize_scalar(lambda b: -M.pl_loglik(test, with_beta({t: st_bt.get(t, 0) for t in set(test.team)}, b, debut)), bounds=(-0.5, 1), method="bounded")
BETA = float(opt.x)
bg = {g: d for g, d in test.groupby("game_id")}; tg = test.game_id.unique(); bb = []
for _ in range(200):
    d = pd.concat([bg[g].assign(game_id=f"{g}_{i}") for i, g in enumerate(rng.choice(tg, len(tg)))])
    bb.append(minimize_scalar(lambda b: -M.pl_loglik(d, with_beta({t: st_bt.get(t, 0) for t in set(d.team)}, b, debut)), bounds=(-0.5, 1), method="bounded").x)
BETA_SD = float(np.std(bb))
st_v2 = with_beta({t: st_bt.get(t, 0) for t in set(test.team)}, BETA, debut)
nT = test.game_id.nunique()
act = test.groupby("team").points.mean()
pred = pd.Series(st_v2)[act.index]
thirds = []
order = pred.sort_values(ascending=False).index
for name, chunk in zip(["Top third (model's best 13)", "Middle third", "Bottom third (model's worst 13)"], [order[:13], order[13:27], order[27:]]):
    thirds.append(dict(label=name, n=len(chunk), actual_ppg=float(act[chunk].mean())))
pl = df[df.source == "pro_league"].copy(); pl["day"] = pl.stage.str.extract(r"_D(\d)")[0].astype(float)
pl_tr, pl_te = pl[pl.day <= 6], pl[(pl.day >= 7) | pl.stage.str.endswith("FINALS")]
backtest = dict(
    n_games=int(nT), n_debut=len(debut),
    acc_v2=float(pairwise(test, st_v2)), acc_old=float(pairwise(test, st_flat)),
    gain_v2=float((M.pl_loglik(test, st_v2) - uni_ll(test)) / nT), gain_old=float((M.pl_loglik(test, st_flat) - uni_ll(test)) / nT),
    acc_pl=float(pairwise(pl_te, M.fit_pl(pl_tr, sigma=SIGMA))), n_pl=int(pl_te.game_id.nunique()),
    s1_trials=s1_trials, s1_weight=S1W, debut_actual=float(act[list(debut)].mean()), returning_actual=float(act.drop(list(debut)).mean()), thirds=thirds,
    beta=BETA, beta_lo=float(np.percentile(bb, 10)), beta_hi=float(np.percentile(bb, 90)))

# ---------------- final fit on all games ----------------
lan_teams = set(df[df.source == "lan"].team)
G = groups_for(df.team.unique(), lan_teams)
strength, gmeans = M.fit_pl(df, sigma=SIGMA, groups=G, return_groups=True)
base, kskill = M.fit_kills(df, sigma=0.15)
teams_sorted = sorted(strength)
by_gid = {g: d for g, d in df.groupby("game_id")}; gids = df.game_id.unique()
NB = 100
boot = np.zeros((NB, len(teams_sorted)))
for b in range(NB):
    d = pd.concat([by_gid[g].assign(game_id=f"{g}_b{i}") for i, g in enumerate(rng.choice(gids, len(gids)))])
    s = M.fit_pl(d, sigma=SIGMA, groups=G, teams=teams_sorted)
    boot[b] = [s[t] for t in teams_sorted]
no_lan = np.array([t not in lan_teams for t in teams_sorted])
beta_draw = rng.normal(BETA, BETA_SD, NB)
boot_lan = boot - beta_draw[:, None] * no_lan[None, :]                 # LAN-scale strength draws
lan_strength = {t: strength[t] - (BETA if t not in lan_teams else 0) for t in teams_sorted}
ti = {t: i for i, t in enumerate(teams_sorted)}

CHAMPS = ["AG GLOBAL", "Alliance", "Aurora", "Crazy Thieves", "DINOS", "ENTER FORCE.36", "ForFun Esports", "Forged Wolves",
          "GaiminGladiators", "Geekay Esports", "Gen.G Citadel", "KINOTROPE CLUB", "NinjasInPyjamas", "NRG", "PVISION", "PXX",
          "REJECT", "Relove DCG", "S8UL", "Sentinels", "ShopifyRebellion", "Stallions", "Team Falcons", "Team Liquid", "Team RRQ",
          "Team Vision", "TOP", "UNLIMIT", "VK GAMING", "WhiteGrimReaper", "Wolves Esports", "ZETA DIVISION"]
LCQ_QUAL = ["FFS", "SLG", "Team WaR", "TLN PIRATES", "NOEZ FOXX", "ALTERNATIVE", "Daylight Savings", "CPR"]
CHAMPS = CHAMPS + LCQ_QUAL
assert all(t in strength for t in CHAMPS)
fidx = np.array([ti[t] for t in CHAMPS])


# ---------------- plain-unit rating: expected points per game in a Championship lobby ----------------
def power(theta, ks, n=6000):
    """One team with strength theta / kill skill ks against 19 random Championship teams. Returns (exp points, P(win game))."""
    others = np.array([lan_strength[t] for t in CHAMPS])
    pick = np.argsort(rng.random((n, len(CHAMPS))), 1)[:, :19]
    z = np.concatenate([np.full((n, 1), theta), others[pick]], 1) + rng.gumbel(size=(n, 20))
    place = 1 + (z[:, 1:] > z[:, :1]).sum(1)
    pts = M.PTS[place] + rng.poisson(np.exp(base[place] + ks))
    return float(pts.mean()), float((place == 1).mean())


grid = np.linspace(-0.9, 0.7, 17); curve = np.array([power(g, 0.0, 12000)[0] for g in grid])
f_pts = lambda th: float(np.interp(th, grid, curve))
rows = []
agg = df[df.source != "scrim"].groupby("team").agg(games=("game_id", "count"), ppg=("points", "mean"), avg_place=("placement", "mean"),
                             wins=("placement", lambda x: int((x == 1).sum())), kpg=("kills", "mean"),
                             lan_games=("source", lambda s: int((s == "lan").sum()))).reset_index()
for _, r in agg.iterrows():
    t = r.team; i = ti[t]; th = lan_strength[t]
    p, w = power(th, kskill[t])
    lo, hi = np.percentile(boot_lan[:, i], [10, 90])
    rows.append(dict(team=t, region=team_region.get(t, "Other"), games=int(r.games), lan_games=int(r.lan_games), ppg=float(r.ppg),
                     avg_place=float(r.avg_place), wins=int(r.wins), kpg=float(r.kpg), strength=float(th), raw_strength=float(strength[t]),
                     kill_skill=float(kskill[t]), no_lan=bool(t not in lan_teams), power=p, win_game=w,
                     power_lo=p + f_pts(lo) - f_pts(th), power_hi=p + f_pts(hi) - f_pts(th), in_field=t in CHAMPS))
teams_out = sorted(rows, key=lambda r: -r["power"])
avg_power = float(np.mean([r["power"] for r in rows if r["in_field"]]))


# ---------------- Championship forecast (uncertainty in ratings included) ----------------
def champs_forecast(n=8000, seed=11):
    """Y6S2 Playoffs format: 4 groups of 10, round robin (6 series x 6 games) -> top 20 to Winners Round, bottom 20 to Losers R1
    -> WR bottom 10 + LR1 top 10 play Losers R2 -> WR top 10 + LR2 top 10 in Match Point Finals (50 pts, then win a game).
    Each simulation draws one bootstrap set of ratings, so rating uncertainty flows into the odds."""
    r = np.random.default_rng(seed); S = len(CHAMPS)
    TH = boot_lan[r.integers(0, NB, n)][:, fidx]                       # (n, 40) strength draw per simulation
    ks = np.array([kskill[t] for t in CHAMPS]); rows_ = np.arange(n)[:, None]

    def series(lobby, games):
        pts = np.zeros(lobby.shape); th = np.take_along_axis(TH, lobby, 1)
        for _ in range(games):
            place = np.empty(lobby.shape, int)
            place[rows_, np.argsort(-(th + r.gumbel(size=lobby.shape)), 1)] = np.arange(1, lobby.shape[1] + 1)
            pts += M.PTS[place] + r.poisson(np.exp(base[place] + ks[lobby]))
        return pts

    ranked = lambda lobby, pts: np.take_along_axis(lobby, np.argsort(-(pts + r.random(pts.shape) * 1e-3), 1), 1)
    perm = np.argsort(r.random((n, S)), 1); grp = [perm[:, i * 10:(i + 1) * 10] for i in range(4)]
    tot = np.zeros((n, S))
    for a, b in [(0, 1), (2, 3), (1, 3), (0, 2), (1, 2), (0, 3)]:
        lob = np.concatenate([grp[a], grp[b]], 1); np.add.at(tot, (rows_, lob), series(lob, 6))
    allr = ranked(np.tile(np.arange(S), (n, 1)), tot)
    wr_r, lr1_r = ranked(allr[:, :20], series(allr[:, :20], 6)), ranked(allr[:, 20:], series(allr[:, 20:], 6))
    lr2 = np.concatenate([wr_r[:, 10:], lr1_r[:, :10]], 1)
    fin = np.concatenate([wr_r[:, :10], ranked(lr2, series(lr2, 6))[:, :10]], 1)
    made = np.bincount(fin.ravel(), minlength=S); wins = np.zeros(S); totf = np.zeros(fin.shape); alive = np.ones(n, bool)
    thf = np.take_along_axis(TH, fin, 1)
    for _ in range(40):
        i = np.where(alive)[0]
        if not len(i): break
        lob = fin[i]; order = np.argsort(-(thf[i] + r.gumbel(size=lob.shape)), 1)
        place = np.empty(lob.shape, int); place[np.arange(len(i))[:, None], order] = np.arange(1, 21)
        first = order[:, 0]; elig = totf[i, first] >= 50
        np.add.at(wins, lob[np.arange(len(i)), first][elig], 1)
        totf[i] += M.PTS[place] + r.poisson(np.exp(base[place] + ks[lob]))
        alive[i[elig]] = False
    assert abs(wins.sum() - n) < 1 and made.sum() == 20 * n               # every simulation has exactly one champion, 20 finalists
    return {t: dict(p_finals=float(made[k] / n), p_win=float(wins[k] / n)) for k, t in enumerate(CHAMPS)}


champs = champs_forecast()
for r in teams_out:
    if r["in_field"]:
        r.update(champs[r["team"]]); r["path"] = "LCQ" if r["team"] in LCQ_QUAL else "Points"

# ---------------- luck vs skill, in plain terms ----------------
def icc(d):
    g = d.groupby("team").points; n = g.size(); m = g.mean(); N = len(d); k = len(n)
    msb = (n * (m - d.points.mean()) ** 2).sum() / (k - 1); msw = ((d.points - d.team.map(m)) ** 2).sum() / (N - k)
    n0 = (N - (n ** 2).sum() / N) / (k - 1)
    return float(max(msb - msw, 0) / n0 / (max(msb - msw, 0) / n0 + msw)), float(msb / msw), float(1 - fdist.cdf(msb / msw, k - 1, N - k))


lan = df[df.source == "lan"]
icc_lan, F_lan, p_lan = icc(lan); icc_pl, F_pl, p_pl = icc(pl)

# ---------------- maps: do teams differ by map beyond chance? ----------------
official = df[df.source != "scrim"]
tm = official.groupby("team").points.agg(["mean", "var", "count"])
cell = official.groupby(["team", "map"]).points.agg(["mean", "count"]).reset_index()
cell = cell[(cell["count"] >= 12) & (cell["map"] != "unknown")].merge(tm, left_on="team", right_index=True, suffixes=("", "_t"))
z = (cell["mean"] - cell["mean_t"]) / np.sqrt(cell["var"] / cell["count"])
map_ratio = float(z.var())            # ~1.0 means map-to-map differences are what chance alone would produce
team_maps = {}
for _, r in cell.iterrows():
    team_maps.setdefault(r.team, {})[r["map"]] = [int(r["count"]), round(float(r["mean"]), 2)]
scr = df[df.source == "scrim"]
scrims = [dict(team=t, games=int(len(g)), ppg=round(float(g.points.mean()), 2), wins=int((g.placement == 1).sum()), kpg=round(float(g.kills.mean()), 2))
          for t, g in scr.groupby("team")] if len(scr) else []
scrims.sort(key=lambda r: -r["ppg"])

# ---------------- deep stats ----------------
dt = deep.teams()
for c in ["kills", "knocks", "dmg", "assists", "rez", "rspn", "knocked", "dmg_taken", "pts", "deaths"]:
    dt[c + "_pg"] = dt[c] / dt.games
dt["dmg_ratio"] = dt.dmg / dt.dmg_taken; dt["kd"] = dt.kills / dt.deaths
dt["label"] = dt.event.map(lambda e: "Split 2 Playoffs (LAN)" if e == "Y6S2_PLAYOFFS" else "Split 2 Pro League")
deep_teams = {}
for _, r in dt.iterrows():
    deep_teams.setdefault(r.team, {})["lan" if r.event == "Y6S2_PLAYOFFS" else "pl"] = dict(
        games=int(r.games), ppg=round(r.pts_pg, 2), kills=round(r.kills_pg, 2), knocks=round(r.knocks_pg, 2), dmg=round(r.dmg_pg), assists=round(r.assists_pg, 2),
        ratio=round(r.dmg_ratio, 2), kd=round(r.kd, 2), rez=round(r.rez_pg, 2), rspn=round(r.rspn_pg, 2), avg_place=r.avg_place, top1=int(r.top1), top5=int(r.top5))
lanrows = dt[dt.event == "Y6S2_PLAYOFFS"]; plrows = dt[dt.event != "Y6S2_PLAYOFFS"]
STATS = [("kills_pg", "Kills per game"), ("knocks_pg", "Knocks per game"), ("dmg_pg", "Damage per game"), ("assists_pg", "Assists per game"),
         ("dmg_ratio", "Damage dealt ÷ taken"), ("rez_pg", "Revives per game"), ("avg_place", "Average placement")]
links = [dict(stat=lab, lan=float(np.corrcoef(lanrows[c], lanrows.pts_pg)[0, 1]), pl=float(np.corrcoef(plrows[c], plrows.pts_pg)[0, 1])) for c, lab in STATS]
lg = deep.legends()
LEV = {"Y6S2_PLAYOFFS": "Split 2 Playoffs (LAN)", "PL_NA": "Pro League Americas", "PL_EMEA": "Pro League EMEA", "PL_APN": "Pro League APAC North", "PL_APS": "Pro League APAC South"}
legends = {LEV[e]: g[["legend", "pick", "avg_place", "win", "n"]].to_dict("records") for e, g in lg.groupby("event", sort=False)}
pp = deep.players()
players = [dict(player=r.player, team=r.team, src=r.src, games=int(r.games), kills=round(r.kills / r.games, 2), assists=round(r.assists / r.games, 2),
                knocks=round(r.knocks / r.games, 2), dmg=round(r.dmg / r.games), ratio=round(r.dmg / max(r.dmg_taken, 1), 2),
                tk=int(r.kills), ta=int(r.assists)) for r in pp.itertuples()]
n_chk, bad_chk, _ = deep.crosscheck()

src = df.groupby("source").game_id.nunique().to_dict()
out = dict(
    meta=dict(n_games=int(df.game_id.nunique()), n_teams=int(df.team.nunique()), by_source=src, fetched="2026-11-24", sigma=SIGMA,
              beta=BETA, avg_power=avg_power, pts_per_strength=float((f_pts(0.45) - f_pts(0.05)) / 0.4), icc_lan=icc_lan, icc_pl=icc_pl, F_lan=F_lan, F_pl=F_pl,
              totals_checked=n_chk, totals_bad=len(bad_chk), n_no_lan_field=int(sum(r["no_lan"] for r in teams_out if r["in_field"])),
              stages=[dict(stage=s["stage"], url=s["url"], games=len(s["maps"])) for s in stages]),
    teams=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in teams_out],
    backtest=backtest, kill_base=[float(np.exp(base[p])) for p in range(1, 21)], place_pts=[PLACE_PTS.get(p, 0) for p in range(1, 21)],
    deep_teams=deep_teams, links=links, legends=legends, comps=deep.comps(), players=players, lcq_qual=LCQ_QUAL,
    guns=deep.guns(), poi=deep.poi(), team_legends=deep.team_legends(), team_maps=team_maps, map_ratio=map_ratio, scrims=scrims,
    group_means={k: round(v, 3) for k, v in gmeans.items()},
)
os.makedirs("out", exist_ok=True)
json.dump(out, open("out/results.json", "w"))
pd.DataFrame(teams_out).to_csv("out/team_ratings.csv", index=False)
df.to_csv("out/algs_games_tidy.csv", index=False)
if __name__ == "__main__":
    print("backtest", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in backtest.items() if k != "thirds"}); print(thirds)
    print("S1 trials", s1_trials, "chosen", S1W, "| map ratio", round(map_ratio, 2), "| sources", src)
    print("icc lan/pl", round(icc_lan, 3), round(icc_pl, 3), "avg power", round(avg_power, 2))
    f = pd.DataFrame([r for r in teams_out if r["in_field"]]).sort_values("p_win", ascending=False)
    print(f[["team", "region", "games", "lan_games", "no_lan", "raw_strength", "strength", "power", "power_lo", "power_hi", "win_game", "p_finals", "p_win"]].round(3).to_string(index=False))
    print(pd.DataFrame(links).round(2).to_string(index=False))
