"""Fit the model and write every number the two pages use -> out/results.json

Model v3 (see model3.py) = v2 plus: players (ratings follow players across teams), a penalty for landing on a spot
shared with another team, and a match-point effect. Each piece was kept only because it predicted held-out games
better (backtest3.py). Map-by-map team strength and boom-or-bust were tested too and made predictions worse, so they are
reported as information, not used in the odds.

Model (v2):
  * Team strength: Plackett-Luce on each game's full finishing order. Each team is shrunk toward the mean of its
    own group (region x "has LAN games in the data"), not toward one global mean.
  * LAN debut discount: teams with no LAN games get strength - BETA when forecasting LAN play. BETA is estimated
    from how first-time LAN teams did at the Y6 Split 2 Playoffs relative to their regional rating.
  * Kills: Poisson, by placement, with a small team effect.
  * Ratings are reported in plain units: expected points per game in a Championship lobby.
"""
import json, os, numpy as np, pandas as pd
from pathlib import Path
from math import lgamma
from scipy.optimize import minimize_scalar
from scipy.stats import f as fdist
from load import load, PLACE_PTS
import model as M
import model3 as M3
import features as FE
import swing as SW
import deep
from backtest3 import run_fit, groups as groups3, players_of

rng = np.random.default_rng(42)
df, stages = load(include_scrims=True)
df = FE.attach(df, stages)
KW3 = dict(sig_team=0.15, sig_player=0.20, fit_beta=(True, True))     # chosen in backtest3.py
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


def ll_rows(d, ucol="u"):
    """Plackett-Luce log-likelihood from a per-row utility column."""
    ll = 0.0
    for _, g in d.groupby("game_id"):
        u = g.sort_values("placement")[ucol].to_numpy(); mx = u.max(); suf = np.cumsum(np.exp(u - mx)[::-1])[::-1]
        ll += u[:-1].sum() - (np.log(suf[:-1]) + mx).sum()
    return ll


def pairwise_rows(d, ucol="u"):
    hit = tot = 0
    for _, g in d.groupby("game_id"):
        s = g[ucol].values; p = g.placement.values
        ds = s[:, None] - s[None, :]; dp = p[:, None] - p[None, :]; m = ds > 1e-9
        hit += (dp[m] < 0).sum(); tot += m.sum()
    return hit / tot


nT = test.game_id.nunique()
# previous model (v2), kept for the before/after comparison
opt2 = minimize_scalar(lambda b: -M.pl_loglik(test, with_beta({t: st_bt.get(t, 0) for t in set(test.team)}, b, debut)), bounds=(-0.5, 1), method="bounded")
st_prev = with_beta({t: st_bt.get(t, 0) for t in set(test.team)}, float(opt2.x), debut)
# model v3 on the same split
F_bt = run_fit(train, prev_lan | set(test.team), team_region, **KW3)
tu = test.copy(); tu["u0"] = M3.utilities(F_bt, tu); tu["deb"] = tu.team.isin(debut).astype(float)
ll_b = lambda d, b: ll_rows(d.assign(u=d.u0 - b * d.deb))
BETA = float(minimize_scalar(lambda b: -ll_b(tu, b), bounds=(-0.5, 1), method="bounded").x)
bg = {g: d for g, d in tu.groupby("game_id")}; tg = tu.game_id.unique(); bb = []
for _ in range(200):
    d = pd.concat([bg[g].assign(game_id=f"{g}_{i}") for i, g in enumerate(rng.choice(tg, len(tg)))])
    bb.append(minimize_scalar(lambda b: -ll_b(d, b), bounds=(-0.5, 1), method="bounded").x)
BETA_SD = float(np.std(bb))
tu["u"] = tu.u0 - BETA * tu.deb
act = test.groupby("team").points.mean()
pred = tu.groupby("team").u.mean()[act.index]
thirds = []
order = pred.sort_values(ascending=False).index
for name, chunk in zip(["Top third (model's best 13)", "Middle third", "Bottom third (model's worst 13)"], [order[:13], order[13:27], order[27:]]):
    thirds.append(dict(label=name, n=len(chunk), actual_ppg=float(act[chunk].mean())))
pl = df[df.source == "pro_league"].copy(); pl["day"] = pl.stage.str.extract(r"_D(\d)")[0].astype(float)
pl_tr, pl_te = pl[pl.day <= 6], pl[(pl.day >= 7) | pl.stage.str.endswith("FINALS")]
trainB = df[df.source.isin(["lan", "pro_league", "pl_s1"]) & ((df.order < 2.5) | df.game_id.isin(pl_tr.game_id))]
F_pl = run_fit(trainB, set(trainB[trainB.source == "lan"].team), team_region, **KW3)
pu = pl_te.copy(); pu["u"] = M3.utilities(F_pl, pu)
bt3 = json.load(open("out/backtest3.json")) if os.path.exists("out/backtest3.json") else []
backtest = dict(
    n_games=int(nT), n_debut=len(debut),
    acc_v2=float(pairwise_rows(tu)), acc_old=float(pairwise(test, st_flat)), acc_prev=float(pairwise(test, st_prev)),
    gain_v2=float((ll_rows(tu) - uni_ll(test)) / nT), gain_old=float((M.pl_loglik(test, st_flat) - uni_ll(test)) / nT),
    gain_prev=float((M.pl_loglik(test, st_prev) - uni_ll(test)) / nT),
    acc_pl=float(pairwise_rows(pu)), n_pl=int(pl_te.game_id.nunique()), gain_pl=float((ll_rows(pu) - uni_ll(pl_te)) / pl_te.game_id.nunique()),
    s1_trials=s1_trials, s1_weight=S1W, debut_actual=float(act[list(debut)].mean()), returning_actual=float(act.drop(list(debut)).mean()), thirds=thirds,
    beta=BETA, beta_lo=float(np.percentile(bb, 10)), beta_hi=float(np.percentile(bb, 90)), trials=bt3,
    swing_check=json.load(open("out/swing_check.json")) if os.path.exists("out/swing_check.json") else None)

# ---------------- final fit on all games (model v3) ----------------
lan_teams = set(df[df.source == "lan"].team)
teams_sorted = sorted(df.team.unique())
ti = {t: i for i, t in enumerate(teams_sorted)}
PLAYERS = players_of(df)
G = groups3(teams_sorted, lan_teams, team_region)
P3 = M3.prepare(df, teams_sorted, PLAYERS, G, [])
F3 = M3.fit(P3, **KW3)
assert F3["ok"]
B_CLASH, B_MP = float(F3["beta"]["clash"]), float(F3["beta"]["mp"])
latest = {t: FE.latest_roster(t, df) for t in teams_sorted}


def team_strength(F, t):
    """Strength for a team's next game: team part + average of its current players."""
    ps = [F["player"].get(p, 0.0) for p in latest[t]]
    return float(F["team"][ti[t]] + (np.mean(ps) if ps else 0.0))


strength = {t: team_strength(F3, t) for t in teams_sorted}
gmeans = F3["group"]
base, kskill = M.fit_kills(df, sigma=0.15)
# kills are lumpier than a plain count (some games a team finds three fights, some none): negative binomial, see kills_joint.py
from kills_joint import nb_ll as _nbll
from scipy.optimize import minimize_scalar as _ms
_mu = np.exp(base[df.placement.values] + df.team.map(kskill).fillna(0).values)
KILL_R = float(np.exp(_ms(lambda lr: -_nbll(df.kills.values, _mu, np.exp(lr)), bounds=(-1, 6), method="bounded").x))


def kdraw(rng, mu):
    """kills: count with lumpiness (gamma-mixed Poisson = negative binomial, shape KILL_R)"""
    return rng.poisson(mu * rng.gamma(KILL_R, 1 / KILL_R, np.shape(mu)))
by_gid = {g: d for g, d in df.groupby("game_id")}; gids = df.game_id.unique()
NB = 100
boot = np.zeros((NB, len(teams_sorted)))
for b in range(NB):
    d = pd.concat([by_gid[g].assign(game_id=f"{g}_b{i}") for i, g in enumerate(rng.choice(gids, len(gids)))])
    Fb = M3.fit(M3.prepare(d, teams_sorted, PLAYERS, G, []), x0=F3["x"], **KW3)
    boot[b] = [team_strength(Fb, t) for t in teams_sorted]
no_lan = np.array([t not in lan_teams for t in teams_sorted])
beta_draw = rng.normal(BETA, BETA_SD, NB)
boot_lan = boot - beta_draw[:, None] * no_lan[None, :]                 # LAN-scale strength draws
lan_strength = {t: strength[t] - (BETA if t not in lan_teams else 0) for t in teams_sorted}

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
    pts = M.PTS[place] + kdraw(rng, np.exp(base[place] + ks))
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
                     power_lo=p + f_pts(lo) - f_pts(th), power_hi=p + f_pts(hi) - f_pts(th), in_field=t in CHAMPS, sd=float(boot_lan[:, i].std())))
teams_out = sorted(rows, key=lambda r: -r["power"])
avg_power = float(np.mean([r["power"] for r in rows if r["in_field"]]))


# ---------------- Championship forecast (uncertainty in ratings included) ----------------
MAP_LIST = ["Storm Point", "World's Edge", "E-District", "Olympus"]
STG = {x["stage"]: x["maps"] for x in stages}
ROT = dict(groups=[STG[f"Y6S2_PLAYOFFS_GRP_{k}"] for k in ["AvB", "CvD", "BvD", "AvC", "BvC", "AvD"]],
           wr=STG["Y6S2_PLAYOFFS_WR1"], lr1=STG["Y6S2_PLAYOFFS_LR1"], lr2=STG["Y6S2_PLAYOFFS_LR2"],
           finals=STG["Y6S2_PLAYOFFS_FINALS"][:8])                    # finals cycle SP OL ED WE WE ED OL SP, repeated
DROPS = FE.champs_drops()
SPOT_NAMES = sorted({v for d in DROPS.values() for v in d.values()})
SPOT = np.full((len(CHAMPS), len(MAP_LIST)), -1)
for k, t in enumerate(CHAMPS):
    for j, m in enumerate(MAP_LIST):
        if m in DROPS.get(t, {}):
            SPOT[k, j] = SPOT_NAMES.index(DROPS[t][m]) + 1000 * j


# ---- zone: distance from each team's usual spot to where past ALGS final zones ended (placement.gg, see zone_effect.py) ----
import json as _json, rings as _RG
ZONE = _json.load(open("out/zone_effect.json")) if Path("out/zone_effect.json").exists() else None
B_ZONE = ZONE["d_end"]["beta"] if ZONE else 0.0
_RD, _PO = _RG.load(), _RG.outlines()
ZD = {}            # map -> (k past end zones, 40 teams) distances, nan if spot unknown
for j, m in enumerate(MAP_LIST):
    ends = _RD[m]["end"][_RD[m]["algs"]] if _RD[m]["algs"].sum() >= 50 else _RD[m]["end"]
    cen = {n: p.mean(0) for n, p in _PO[m]}
    c = np.array([cen.get(DROPS.get(t, {}).get(m), (np.nan, np.nan)) for t in CHAMPS])
    ZD[m] = np.hypot(ends[:, None, 0] - c[None, :, 0], ends[:, None, 1] - c[None, :, 1])


def zone_x(lobby, mapname, r):
    """Per simulated game: draw a final zone from this map's ALGS history; x = how much closer than the lobby average the
    team's spot is (per 10% of map width), minus what that team gets on average -> zone luck only, no double counting."""
    D = ZD[mapname]; d = D[r.integers(0, len(D), len(lobby))[:, None], lobby]; dbar = np.nanmean(D, 0)[lobby]
    x = (np.nanmean(d, 1, keepdims=True) - d) - (np.nanmean(dbar, 1, keepdims=True) - dbar)
    return np.nan_to_num(x / 0.1)


def zone_pull(t):
    """How often the zone ends close (within 15% of the map width) to this team's usual spot, per map."""
    k = CHAMPS.index(t); o = {}
    for m in MAP_LIST:
        col = ZD[m][:, k]
        if not np.isnan(col).all(): o[m] = round(float((col < 0.15).mean()), 3)
    return o


def clash_mask(lobby, mapname):
    """(n, L) array of lobby team indices -> True where the team's usual spot on this map is also another lobby team's."""
    sp = SPOT[lobby, MAP_LIST.index(mapname)]
    eq = (sp[:, :, None] == sp[:, None, :]) & (sp[:, :, None] >= 0)
    return eq.sum(2) > 1


def clash_partners():
    out = {}
    for k, t in enumerate(CHAMPS):
        for j, m in enumerate(MAP_LIST):
            if SPOT[k, j] < 0: continue
            others = [CHAMPS[q] for q in range(len(CHAMPS)) if q != k and SPOT[q, j] == SPOT[k, j]]
            if others: out.setdefault(t, []).append(dict(map=m, spot=DROPS[t][m], with_=others))
    return out


# ---- real Champs groups, once announced: data/champs_groups.txt, lines "A|Team;Team;..." (10 teams each) ----
def read_groups():
    p = Path("data/champs_groups.txt")
    if not p.exists(): return None
    g = {}
    for l in p.read_text().splitlines():
        if "|" in l and not l.startswith("#"):
            k, v = l.split("|", 1); g[k.strip()] = [x.strip() for x in v.split(";") if x.strip()]
    bad = [t for v in g.values() for t in v if t not in CHAMPS]
    ok = sorted(g) == ["A", "B", "C", "D"] and all(len(v) == 10 for v in g.values()) and not bad and len({t for v in g.values() for t in v}) == 40
    if not ok:
        print("WARNING data/champs_groups.txt ignored (needs groups A-D, 10 Champs teams each); unknown names:", bad)
        return None
    return g


GROUPS = read_groups()


def champs_forecast(n=8000, seed=11, use_clash=True, use_mp=True, use_zone=True):
    """Y6S2 Playoffs format: 4 groups of 10, round robin (6 series x 6 games) -> top 20 to Winners Round, bottom 20 to Losers R1
    -> WR bottom 10 + LR1 top 10 play Losers R2 -> WR top 10 + LR2 top 10 in Match Point Finals (50 pts, then win a game).
    Maps follow the Split 2 Playoffs rotation. Teams whose usual landing spot is shared with another team in the lobby get the
    fitted shared-spot penalty for that game; in the finals, teams on match point get the fitted match-point effect.
    Each simulation draws one bootstrap set of ratings, so rating uncertainty flows into the odds."""
    r = np.random.default_rng(seed); S = len(CHAMPS)
    TH = boot_lan[r.integers(0, NB, n)][:, fidx]                       # (n, 40) strength draw per simulation
    ks = np.array([kskill[t] for t in CHAMPS]); rows_ = np.arange(n)[:, None]
    bc = B_CLASH if use_clash else 0.0; bm = B_MP if use_mp else 0.0; bz = B_ZONE if use_zone else 0.0

    def series(lobby, maps):
        pts = np.zeros(lobby.shape); th = np.take_along_axis(TH, lobby, 1)
        for m in maps:
            u = th + bc * clash_mask(lobby, m) + bz * zone_x(lobby, m, r)
            place = np.empty(lobby.shape, int)
            place[rows_, np.argsort(-(u + r.gumbel(size=lobby.shape)), 1)] = np.arange(1, lobby.shape[1] + 1)
            pts += M.PTS[place] + kdraw(r, np.exp(base[place] + ks[lobby]))
        return pts

    ranked = lambda lobby, pts: np.take_along_axis(lobby, np.argsort(-(pts + r.random(pts.shape) * 1e-3), 1), 1)
    if GROUPS:
        grp = [np.tile(np.array([CHAMPS.index(t) for t in GROUPS[k]]), (n, 1)) for k in "ABCD"]
    else:
        perm = np.argsort(r.random((n, S)), 1); grp = [perm[:, i * 10:(i + 1) * 10] for i in range(4)]
    tot = np.zeros((n, S))
    for (a, b), maps in zip([(0, 1), (2, 3), (1, 3), (0, 2), (1, 2), (0, 3)], ROT["groups"]):
        lob = np.concatenate([grp[a], grp[b]], 1); np.add.at(tot, (rows_, lob), series(lob, maps))
    allr = ranked(np.tile(np.arange(S), (n, 1)), tot)
    top20 = np.bincount(allr[:, :20].ravel(), minlength=S)
    wr_r, lr1_r = ranked(allr[:, :20], series(allr[:, :20], ROT["wr"])), ranked(allr[:, 20:], series(allr[:, 20:], ROT["lr1"]))
    lr2 = np.concatenate([wr_r[:, 10:], lr1_r[:, :10]], 1)
    fin = np.concatenate([wr_r[:, :10], ranked(lr2, series(lr2, ROT["lr2"]))[:, :10]], 1)
    made = np.bincount(fin.ravel(), minlength=S); wins = np.zeros(S); totf = np.zeros(fin.shape); alive = np.ones(n, bool)
    thf = np.take_along_axis(TH, fin, 1)
    for gno in range(40):
        i = np.where(alive)[0]
        if not len(i): break
        lob = fin[i]; m = ROT["finals"][gno % len(ROT["finals"])]
        u = thf[i] + bc * clash_mask(lob, m) + bm * (totf[i] >= 50) + bz * zone_x(lob, m, r)
        order = np.argsort(-(u + r.gumbel(size=lob.shape)), 1)
        place = np.empty(lob.shape, int); place[np.arange(len(i))[:, None], order] = np.arange(1, 21)
        first = order[:, 0]; elig = totf[i, first] >= 50
        np.add.at(wins, lob[np.arange(len(i)), first][elig], 1)
        totf[i] += M.PTS[place] + kdraw(r, np.exp(base[place] + ks[lob]))
        alive[i[elig]] = False
    assert abs(wins.sum() - n) < 1 and made.sum() == 20 * n               # every simulation has exactly one champion, 20 finalists
    return {t: dict(p_finals=float(made[k] / n), p_win=float(wins[k] / n), p_top20=float(top20[k] / n)) for k, t in enumerate(CHAMPS)}


champs = champs_forecast(use_mp=False)        # match point: placement effect is real, but no extra wins (see mp_all) -> not in the title odds
for r in teams_out:
    if r["in_field"]:
        r.update(champs[r["team"]]); r["path"] = "LCQ" if r["team"] in LCQ_QUAL else "Points"

# ---------------- legend use per Champs team (Split 2 games), for ban talking points ----------------
def legend_use():
    import bans_comps as BCm
    if not BCm.SRC.exists(): return {}
    d = BCm.attach(df[df.stage.str.match(r"(PL_|LCQ_|Y6S2_)")], BCm.read())
    out = {}
    for t in CHAMPS:
        g = d[(d.team == t) & d.comp.notna()]
        if len(g):
            c = pd.Series([l for comp in g.comp for l in comp]).value_counts() / len(g)
            out[t] = dict(games=int(len(g)), top=[[k, round(float(v), 2)] for k, v in c.head(5).items()])
    return out


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

# ---------------- v3 extras: boom-or-bust, match point, rosters, shared spots, calibration ----------------
official = df[df.source != "scrim"].copy()
official["u"] = M3.utilities(F3, official)
zt, tau2 = SW.shrink(SW.team_z2(official, n=300))
TAU = float(np.sqrt(tau2))
halves = []
for rep in range(3):                                                    # does swing carry over? random split-half check
    hh = set(rng.choice(gids, len(gids) // 2, replace=False))
    a_ = SW.team_z2(official[official.game_id.isin(hh)], n=150, seed=rep); b_ = SW.team_z2(official[~official.game_id.isin(hh)], n=150, seed=rep + 9)
    j_ = a_.join(b_, lsuffix="_a", rsuffix="_b").dropna(); j_ = j_[(j_.n_a >= 20) & (j_.n_b >= 20)]
    halves.append(float(np.corrcoef(j_.m_a, j_.m_b)[0, 1]))
top3 = official.groupby("team").placement.apply(lambda x: float((x <= 3).mean())); bot = official.groupby("team").placement.apply(lambda x: float((x >= 14).mean()))
zf = zt.loc[[t for t in CHAMPS if t in zt.index]]                      # labels compare each team with the rest of the Champs field
CENTER = float(zf.excess.mean()); SPREAD = float(zf.excess.std())
swing = {t: dict(label=SW.label(r.excess, SPREAD, CENTER), excess=round(float(r.excess), 3), n=int(r.n), top3=round(top3[t], 3), bottom=round(bot[t], 3))
         for t, r in zt.iterrows()}

# match point: every finals game a team started on 50+
fz = official[official.stage.str.endswith("FINALS")].copy()
fz["p_first"] = fz.groupby("game_id").u.transform(lambda u: np.exp(u - B_MP * 0) / np.exp(u).sum())
fz["u_nomp"] = fz.u - B_MP * fz.mp
fz["p_nomp"] = fz.groupby("game_id").u_nomp.transform(lambda u: np.exp(u) / np.exp(u).sum())
on = fz[fz.mp == 1]
mp_all = dict(games=int(len(on)), wins=int((on.placement == 1).sum()), expected=round(float(on.p_nomp.sum()), 2),
              finals=int(fz.stage.nunique()), beta=B_MP, odds=float(np.exp(B_MP) - 1))
mp_team = {t: dict(games=int(len(g)), wins=int((g.placement == 1).sum()), expected=round(float(g.p_nomp.sum()), 2),
                   finals=int(g.stage.nunique())) for t, g in on.groupby("team")}

# rosters: who is back from Split 1, where newcomers came from
o_hist = df[df.roster.map(lambda r: bool(r) and r[0].startswith("O:"))]


def built_from(t):
    now = set(latest[t])
    if not now or not next(iter(now)).startswith("O:"):
        return None
    s1 = df[df.team.isin([t, t + " (S1 roster)"]) & (df.order < 2.0) & df.roster.map(lambda r: bool(r) and r[0].startswith("O:"))]
    s1set = set(s1.roster.iloc[-1]) if len(s1) else set()
    new = []
    own = df[df.team.isin([t, t + " (S1 roster)"]) & (df.order < 2.0)].roster.explode()
    s1set |= set(own.dropna()) & now                      # anyone who played for this team in Split 1 counts as returning
    for p in sorted(now - s1set):
        prev = o_hist[(o_hist.team != t) & (o_hist.order < 2.5) & o_hist.roster.map(lambda r: p in r)]
        new.append(prev.groupby("team").size().idxmax().replace(" (S1 roster)", "") if len(prev) else None)
    return dict(returning=len(now & s1set), new_from=new, had_s1=bool(len(s1)))


built = {t: built_from(t) for t in CHAMPS}
partners = clash_partners()
champs_plain = champs_forecast(n=8000, seed=11, use_clash=False, use_mp=False)   # same draws as the main forecast, minus shared spots


def calib(d, ucol="u", n=3000, seed=21):
    """Predicted chance of winning the game / finishing top 5 vs what happened, in bins."""
    rr = np.random.default_rng(seed); rows = []
    for _, g in d.groupby("game_id"):
        u = g[ucol].to_numpy(); L = len(u)
        z = u[None, :] + rr.gumbel(size=(n, L)); pl_ = np.empty((n, L), int); pl_[np.arange(n)[:, None], np.argsort(-z, 1)] = np.arange(1, L + 1)
        rows.append(pd.DataFrame(dict(pw=np.exp(u) / np.exp(u).sum(), p5=(pl_ <= 5).mean(0), w=(g.placement.to_numpy() == 1), t5=(g.placement.to_numpy() <= 5))))
    return pd.concat(rows)


def bins(r, k=5):
    out = {}
    for col, hit in [("pw", "w"), ("p5", "t5")]:
        r = r.assign(bin=pd.qcut(r[col].rank(method="first"), k, labels=False))
        out[col] = [dict(said=float(g[col].mean()), happened=float(g[hit].mean()), n=int(len(g)),
                         se=float(np.sqrt(g[col].mean() * (1 - g[col].mean()) / len(g)))) for _, g in r.groupby("bin")]
    return out


cal_rows = pd.concat([calib(tu), calib(pu)])
calibration = dict(all=bins(cal_rows), n_games=int(nT + pl_te.game_id.nunique()), n_lan=int(nT), n_pl=int(pl_te.game_id.nunique()),
                   brier_win=float(((cal_rows.pw - cal_rows.w) ** 2).mean()), brier_win_luck=float(((0.05 - cal_rows.w) ** 2).mean()))

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
    bans_comps=json.load(open('out/bans_comps.json')) if os.path.exists('out/bans_comps.json') else None,
    legend_use=legend_use(),
    track_record=json.load(open('out/track_record.json')) if os.path.exists('out/track_record.json') else None,
    groups=dict(announced=bool(GROUPS), teams=GROUPS),
    win_sharp=json.load(open('out/win_sharpness.json')) if os.path.exists('out/win_sharpness.json') else None,
    form_drift=json.load(open('out/form_drift.json')) if os.path.exists('out/form_drift.json') else None,
    scrim_check=json.load(open('out/scrim_check.json')) if os.path.exists('out/scrim_check.json') else None,
    kills_model=dict(r=KILL_R, check=json.load(open('out/kills_joint.json')) if os.path.exists('out/kills_joint.json') else None),
    zone_model=dict(beta=B_ZONE, quint=__import__('zone_effect').quintiles(), fit=ZONE, pull={t: zone_pull(t) for t in CHAMPS},
                    spot_dist={m: [None if np.isnan(v) else round(float(v), 3) for v in np.nanmean(ZD[m], 0)] for m in MAP_LIST}),
    meta=dict(kill_r=KILL_R, b_zone=B_ZONE, n_games=int(df.game_id.nunique()), n_teams=int(df.team.nunique()), by_source=src, fetched="2026-11-24", sigma=SIGMA,
              beta=BETA, avg_power=avg_power, pts_per_strength=float((f_pts(0.45) - f_pts(0.05)) / 0.4), icc_lan=icc_lan, icc_pl=icc_pl, F_lan=F_lan, F_pl=F_pl,
              totals_checked=n_chk, totals_bad=len(bad_chk), n_no_lan_field=int(sum(r["no_lan"] for r in teams_out if r["in_field"])),
              stages=[dict(stage=s["stage"], url=s["url"], games=len(s["maps"])) for s in stages]),
    teams=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in teams_out],
    backtest=backtest, kill_base=[float(np.exp(base[p])) for p in range(1, 21)], place_pts=[PLACE_PTS.get(p, 0) for p in range(1, 21)],
    deep_teams=deep_teams, links=links, legends=legends, comps=deep.comps(), players=players, lcq_qual=LCQ_QUAL,
    guns=deep.guns(), poi=deep.poi(), team_legends=deep.team_legends(), team_maps=team_maps, map_ratio=map_ratio, scrims=scrims,
    group_means={k: round(float(v), 3) for k, v in gmeans.items()},
    v3=dict(clash=B_CLASH, mp=B_MP, pts_clash=float(B_CLASH * (f_pts(0.45) - f_pts(0.05)) / 0.4), clash_odds=float(np.exp(B_CLASH) - 1),
            sig_player=KW3["sig_player"], n_players=len(PLAYERS), n_clash_rows=int(official.clash.sum()), tau_swing=TAU, swing_halves=halves,
            rot={k: v for k, v in ROT.items() if k != "groups"}, rot_groups=ROT["groups"]),
    swing=swing, mp_all=mp_all, mp_team=mp_team, built=built, clash_partners=partners, calibration=calibration,
    champs_plain={t: round(v["p_win"], 4) for t, v in champs_plain.items()},
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
