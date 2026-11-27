"""Track record: how the model would have done at past LANs, using only games played BEFORE each event.

For each past LAN we refit the current model (players + shared spot + match point) on earlier games only, then:
  * Split 2 Playoffs: simulate the whole event in its real format with the real groups and map rotation
    -> each team's chance to reach the final and to win, before a single Playoffs game.
  * Both Playoffs: simulate just the Match Point final with the 20 teams that actually made it.
Ratings are point estimates (no rating-uncertainty draws) and no LAN-debut discount is applied, so nothing fitted on these
events leaks in. The Year 5 Championship can't be checked: we have no games from before it. Writes out/track_record.json."""
import json
import numpy as np
import backtest3 as BT, features as FE, model as M

KW = BT.CONFIGS["New model (players + spot + match point)"]
N = 20000


def strengths(train, reg):
    lanset = set(train[train.source == "lan"].team)
    F = BT.run_fit(train, lanset, reg, **KW)
    teams = sorted(train.team.unique()); ti = {t: i for i, t in enumerate(teams)}

    def s(t):
        if t not in ti: return float(np.mean(F["team"]))
        ps = [F["player"].get(p, 0.0) for p in FE.latest_roster(t, train)]
        return float(F["team"][ti[t]] + (np.mean(ps) if ps else 0.0))
    base, ks = M.fit_kills(train, sigma=0.15)
    return s, base, ks


class Sim:
    def __init__(self, field, s, base, ks, seed=5):
        self.field = list(field); self.r = np.random.default_rng(seed)
        self.th = np.array([s(t) for t in field]); self.ks = np.array([ks.get(t, 0.0) for t in field]); self.base = base

    def series(self, lobby, games):
        r = self.r; n, k = lobby.shape; pts = np.zeros((n, k)); th = self.th[lobby]; rows = np.arange(n)[:, None]
        for _ in range(games):
            place = np.empty((n, k), int); place[rows, np.argsort(-(th + r.gumbel(size=(n, k))), 1)] = np.arange(1, k + 1)
            pts += M.PTS[place] + r.poisson(np.exp(self.base[place] + self.ks[lobby]))
        return pts

    def ranked(self, lobby, pts):
        return np.take_along_axis(lobby, np.argsort(-(pts + self.r.random(pts.shape) * 1e-3), 1), 1)

    def finals(self, fin):
        r = self.r; n = len(fin); wins = np.zeros(len(self.field)); tot = np.zeros(fin.shape); alive = np.ones(n, bool)
        for _ in range(40):
            i = np.where(alive)[0]
            if not len(i): break
            lob = fin[i]; order = np.argsort(-(self.th[lob] + r.gumbel(size=lob.shape)), 1)
            place = np.empty(lob.shape, int); place[np.arange(len(i))[:, None], order] = np.arange(1, 21)
            first = order[:, 0]; elig = tot[i, first] >= 50
            np.add.at(wins, lob[np.arange(len(i)), first][elig], 1)
            tot[i] += M.PTS[place] + r.poisson(np.exp(self.base[place] + self.ks[lob])); alive[i[elig]] = False
        return wins / n


def s2_groups(df):
    m = {k: set(df[df.stage == f"Y6S2_PLAYOFFS_GRP_{k}"].team) for k in ["AvB", "CvD", "BvD", "AvC", "BvC", "AvD"]}
    return dict(A=m["AvB"] & m["AvC"], B=m["AvB"] & m["BvC"], C=m["CvD"] & m["AvC"], D=m["CvD"] & m["BvD"])


def summarise(field, p_win, p_fin, winner, finalists):
    n = len(field); order = np.argsort(-p_win)
    out = dict(teams=n, winner=winner, winner_p=round(float(p_win[field.index(winner)]), 4),
               winner_rank=int(1 + (p_win > p_win[field.index(winner)]).sum()), even=round(1 / n, 4),
               top=[dict(team=field[k], p_win=round(float(p_win[k]), 4), **({"p_finals": round(float(p_fin[k]), 4)} if p_fin is not None else {}),
                         made_final=field[k] in finalists, won=field[k] == winner) for k in order[:10]])
    if p_fin is not None:
        F = np.array([t in finalists for t in field]); k = int(F.sum())
        top = set(np.argsort(-p_fin)[:k])
        out.update(finals_hits=int(sum(F[j] for j in top)), finals_k=k, finals_hits_even=round(k * k / n, 1),
                   brier=round(float(((p_fin - F) ** 2).mean()), 4), brier_even=round(float(((k / n - F) ** 2).mean()), 4))
    return out


def finals_only(df, stage, s, base, ks):
    d = df[df.stage == stage]; field = sorted(d.team.unique())
    tot = d.groupby("team").points.sum()
    last = d[d.game_id == d.sort_values("game").game_id.iloc[-1]]; winner = last[last.placement == 1].team.iloc[0]
    S = Sim(field, s, base, ks); pw = S.finals(np.tile(np.arange(len(field)), (N, 1)))
    o = summarise(field, pw, None, winner, set(field)); o["note"] = "final only, with the 20 teams that made it"
    return o


if __name__ == "__main__":
    df, stages, reg = BT.prep_all()
    use = df.source.isin(["lan", "pro_league", "pl_s1", "lcq"])
    out = {}
    # ---- Split 1 Playoffs: only the final can be simulated faithfully (its earlier stages had a different format) ----
    tr1 = df[use & (df.order < 1.9)]
    s, base, ks = strengths(tr1, reg)
    out["Split 1 Playoffs"] = dict(final=finals_only(df, "Y6S1_PLAYOFFS_FINALS", s, base, ks), trained_games=int(tr1.game_id.nunique()))
    print(out["Split 1 Playoffs"], flush=True)
    # ---- Split 2 Playoffs: whole event, real groups and maps ----
    tr2 = df[use & (df.order < 2.8)]
    s, base, ks = strengths(tr2, reg)
    G = s2_groups(df); field = sorted(set().union(*G.values())); fi = {t: i for i, t in enumerate(field)}
    S = Sim(field, s, base, ks); rows = np.arange(N)[:, None]
    grp = {k: np.tile(np.array([fi[t] for t in sorted(v)]), (N, 1)) for k, v in G.items()}
    tot = np.zeros((N, len(field)))
    for a, b in [("A", "B"), ("C", "D"), ("B", "D"), ("A", "C"), ("B", "C"), ("A", "D")]:
        lob = np.concatenate([grp[a], grp[b]], 1); np.add.at(tot, (rows, lob), S.series(lob, 6))
    allr = S.ranked(np.tile(np.arange(len(field)), (N, 1)), tot)
    wr = S.ranked(allr[:, :20], S.series(allr[:, :20], 6)); lr1 = S.ranked(allr[:, 20:], S.series(allr[:, 20:], 6))
    lr2 = np.concatenate([wr[:, 10:], lr1[:, :10]], 1)
    fin = np.concatenate([wr[:, :10], S.ranked(lr2, S.series(lr2, 6))[:, :10]], 1)
    p_fin = np.bincount(fin.ravel(), minlength=len(field)) / N
    p_win = S.finals(fin)
    d = df[df.stage == "Y6S2_PLAYOFFS_FINALS"]; finalists = set(d.team)
    last = d[d.game_id == d.sort_values("game").game_id.iloc[-1]]; winner = last[last.placement == 1].team.iloc[0]
    out["Split 2 Playoffs"] = dict(event=summarise(field, p_win, p_fin, winner, finalists),
                                   final=finals_only(df, "Y6S2_PLAYOFFS_FINALS", s, base, ks), trained_games=int(tr2.game_id.nunique()))
    print(out["Split 2 Playoffs"], flush=True)
    json.dump(out, open("out/track_record.json", "w"), indent=1)
