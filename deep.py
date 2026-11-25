"""Load deep stats (team damage/knocks/assists, players, legends) and cross-check them against the game data."""
from pathlib import Path
import pandas as pd
from load import load, norm_team
D = Path(__file__).parent / "data" / "deep"
HINT = {"Y6S2_PLAYOFFS": "Y6S2_", "PL_NA": "PL_NA_", "PL_EMEA": "PL_EMEA_", "PL_APN": "PL_APN_", "PL_APS": "PL_APS_"}
TCOLS = ["team", "games", "avg_place", "kills", "pts", "knocks", "knocked", "dmg", "dmg_taken", "rez", "rspn", "deaths", "top1", "top5", "assists"]


def _blocks(path):
    out, cur = {}, None
    for line in Path(path).read_text().splitlines():
        if line.startswith("## "): cur = line[3:].strip(); out[cur] = []
        elif line and not line.startswith("#") and cur: out[cur].append(line.split("|"))
    return out


def teams():
    rows = []
    for ev, rs in _blocks(D / "teams.txt").items():
        for r in rs:
            d = dict(zip(TCOLS, [norm_team(r[0], HINT.get(ev, ""))] + [float(x) for x in r[1:]])); d["event"] = ev; rows.append(d)
    return pd.DataFrame(rows)


def legends():
    return pd.DataFrame([dict(event=ev, legend=r[0], pick=float(r[1]), avg_place=float(r[2]), win=float(r[3]), n=int(r[4]))
                         for ev, rs in _blocks(D / "legends.txt").items() for r in rs])


def players():
    rows = [l.split("|") for l in (D / "players.txt").read_text().splitlines() if l and not l.startswith("#")]
    p = pd.DataFrame(rows, columns=["player", "team", "games", "kills", "assists", "knocks", "dmg", "dmg_taken"])
    p["team"] = p.team.map(lambda t: norm_team(t, "Y6S2_")); p["src"] = "Split 2 Playoffs (LAN)"
    rows2 = [l.split("|") for l in (D / "players_pl.txt").read_text().splitlines() if l and not l.startswith("#")]
    q = pd.DataFrame(rows2, columns=p.columns[:8]); q["src"] = "Split 2 Pro League"
    q = q[~q.team.isin(set(p.team))]                      # LAN lines win where a team has both
    p = pd.concat([p, q], ignore_index=True)
    for c in ["games", "kills", "assists", "knocks", "dmg", "dmg_taken"]: p[c] = p[c].astype(int)
    return p


def guns():
    rows = [l.split("|") for l in (D / "guns.txt").read_text().splitlines() if l and not l.startswith("#")]
    return [dict(weapon=r[0], kills=int(r[1]), dmg=int(r[2]), knocks=int(r[3]), fwr=float(r[4]), time=r[5]) for r in rows]


def poi():
    return {m: [dict(poi=r[0], wins=int(r[1]), pts=float(r[2]), kills=float(r[3]), place=float(r[4]), picks=int(r[5])) for r in rs]
            for m, rs in _blocks(D / "poi.txt").items()}


def team_legends():
    out = {}
    for blk, rs in _blocks(D / "team_legends.txt").items():
        kind, src = blk.split()
        for r in rs:
            t = norm_team(r[0], "Y6S2_" if src == "LAN" else "")
            d = out.setdefault(t, {})
            if kind == "LEGENDS" and ("legends" not in d or src == "LAN"):
                d["legends"] = [[x.split(":")[0], int(x.split(":")[1])] for x in r[1].split(",")]; d["src"] = "Split 2 Playoffs (LAN)" if src == "LAN" else "Split 2 Pro League"
            if kind == "COMPS" and ("comps" not in d or src == "LAN"):
                d["comps"] = [[c.split(":")[0].replace("+", ", "), int(c.split(":")[1]), float(c.split(":")[2])] for c in r[1].split(";")]
    return out


def comps():
    rows = [l.split("|") for l in (D / "comps.txt").read_text().splitlines() if l and not l.startswith("#")]
    return [dict(comp=r[0], pick=float(r[1]), avg_place=float(r[2]), win=float(r[3]), top5=float(r[4]), n=int(r[5])) for r in rows]


def crosscheck():
    """Compare the site's team totals (games, kills, points) with sums from our per-game data."""
    df, _ = load(); t = teams(); bad = []; n = 0
    sel = {"Y6S2_PLAYOFFS": df.stage.str.startswith("Y6S2_"), "PL_NA": df.stage.str.startswith("PL_NA_"), "PL_EMEA": df.stage.str.startswith("PL_EMEA_"),
           "PL_APN": df.stage.str.startswith("PL_APN_"), "PL_APS": df.stage.str.startswith("PL_APS_")}
    for ev, m in sel.items():
        g = df[m].groupby("team").agg(games=("game_id", "count"), kills=("kills", "sum"), pts=("points", "sum"))
        for _, r in t[t.event == ev].iterrows():
            n += 1
            if r.team not in g.index: bad.append((ev, r.team, "not in game data")); continue
            o = g.loc[r.team]
            if (o.games, o.kills, o.pts) != (r.games, r.kills, r.pts): bad.append((ev, r.team, f"site g/k/p {int(r.games)}/{int(r.kills)}/{int(r.pts)} vs ours {o.games}/{o.kills}/{o.pts}"))
    # Split 1 Pro League season totals
    for ev, rs in _blocks(D / "s1_totals.txt").items():
        g = df[df.stage.str.startswith(ev + "_")].groupby("team").agg(games=("game_id", "count"), kills=("kills", "sum"), pts=("points", "sum"))
        for r in rs:
            n += 1; tn = norm_team(r[0][:-1], ev + "_"); o = g.loc[tn] if tn in g.index else None
            if o is None or (o.games, o.kills, o.pts) != (int(r[1]), int(r[2]), int(r[3])):
                bad.append((ev, tn, f"site g/k/p {r[1]}/{r[2]}/{r[3]} vs ours " + ("missing" if o is None else f"{o.games}/{o.kills}/{o.pts}")))
    # players sum to team kills
    pp = players(); p = pp[pp.src.str.contains("LAN")].groupby("team").kills.sum(); tk = t[t.event == "Y6S2_PLAYOFFS"].set_index("team").kills
    pk = [(k, int(p.get(k, 0)), int(v)) for k, v in tk.items() if p.get(k, 0) != v]
    return n, bad, pk


if __name__ == "__main__":
    n, bad, pk = crosscheck()
    print(f"{n} team rows checked against game data; {len(bad)} differ")
    for b in bad: print("  ", b)
    print("player kills != team kills:", pk)
