"""Per-game features for model v3: roster (player ids), shared landing spot, on match point."""
from pathlib import Path
import numpy as np
import pandas as pd
from load import norm_team

DATA = Path(__file__).parent / "data" / "deep"
MAPN = {"SP": "Storm Point", "WE": "World's Edge", "ED": "E-District", "OL": "Olympus"}
LAN_PREFIX = ("Y5S2_", "Y6S1_PLAYOFFS", "Y6S2_PLAYOFFS")
MP_THRESHOLD = 50


def _read():
    R, D, sec = [], [], None
    for line in (DATA / "rosters_drops.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line in ("R", "D"):
            sec = line; continue
        (R if sec == "R" else D).append(line.split("|"))
    return R, D


def rosters():
    """dict (stage, team) -> tuple of player keys. Player keys carry the id space: 'L:' LAN events, 'O:' online events."""
    R, _ = _read()
    stage_order = list(dict.fromkeys(r[0] for r in R))
    by_team = {}                                     # raw team -> list of (stage position, ids)
    for st, team, ids in R:
        by_team.setdefault(team, []).append((stage_order.index(st), st, ids))
    return by_team, stage_order


def attach(df, stages):
    """Add roster, clash, mp columns to the games table. `stages` = list of stage codes in scrape order (from load())."""
    R, D = _read()
    order = [s["stage"] for s in stages if "apexlegendsstatus" in s.get("url", "")]
    pos = {s: i for i, s in enumerate(order)}
    # The scrape writes a roster line only when a site name's roster changes (in scrape order), so lines must be replayed
    # per SITE name. Several site names can map to one team (renames); use the name the team had at that event.
    by_raw = {}                                       # (team, raw name) -> [(pos, stage, ids)]
    for st, raw, ids in R:
        by_raw.setdefault((norm_team(raw, st), raw), []).append((pos.get(st, -1), st, ids))
    raws_of = {}
    for (t, raw), v in by_raw.items():
        raws_of.setdefault(t, []).append(raw); v.sort()
    event = lambda st: "_".join(st.split("_")[:2])

    def roster_for(stage, team):
        p = pos.get(stage)
        if p is None or team not in raws_of:
            return ()
        lan = stage.startswith(LAN_PREFIX)
        cands = [(raw, [x for x in by_raw[(team, raw)] if x[0] <= p and x[1].startswith(LAN_PREFIX) == lan]) for raw in raws_of[team]]
        cands = [(raw, v) for raw, v in cands if v]
        if not cands:
            return ()
        same = [(raw, v) for raw, v in cands if any(event(x[1]) == event(stage) for x in by_raw[(team, raw)])]
        raw, v = max(same or cands, key=lambda c: c[1][-1][0])
        tag = "L:" if lan else "O:"
        return tuple(tag + i for i in v[-1][2].split(",") if i)

    df = df.copy()
    df["roster"] = [roster_for(s, t) for s, t in zip(df.stage, df.team)]
    df["roster"] = _link_lan_to_online(df)

    shared = {}
    for st, mp, n, groups in D:
        if mp not in MAPN:
            continue
        for g in filter(None, groups.split(";")):
            names = {norm_team(x, st) for x in g.split("~")}
            shared.setdefault((st, MAPN[mp]), []).append(names)
    lobby = df.groupby("game_id").team.apply(set).to_dict()

    def clash(r):
        for grp in shared.get((r.stage, r.map), []):
            if r.team in grp and len((grp - {r.team}) & lobby[r.game_id]) > 0:
                return 1
        return 0

    df["clash"] = [clash(r) for r in df.itertuples()]

    df["mp"] = 0
    for st, g in df[df.stage.str.endswith("FINALS")].groupby("stage"):
        tot = {}
        for gm in sorted(g.game.unique()):
            rows = g[g.game == gm]
            for i, t in zip(rows.index, rows.team):
                df.at[i, "mp"] = int(tot.get(t, 0) >= MP_THRESHOLD)
            for t, p in zip(rows.team, rows.points):
                tot[t] = tot.get(t, 0) + p
    return df


SPLIT_PL = {"Y6S1_PLAYOFFS": "PL1_", "Y6S2_PLAYOFFS": "PL_"}


def _link_lan_to_online(df):
    """LAN events use a different player-id system from online events. Link them through teams: at a split's LAN, a team's
    LAN roster is mapped to that team's latest online roster from the same split's Pro League (same three people in almost
    every case). Any LAN roster seen with that exact trio elsewhere (e.g. the year before) gets the same online ids.
    LAN rosters that cannot be linked keep their LAN ids."""
    link = {}
    for lan, pl in SPLIT_PL.items():
        lan_rows = df[df.stage.str.startswith(lan)]
        onl = df[df.stage.str.startswith(pl) & (df.roster.map(len) == 3)].sort_values("order")
        last_online = onl.groupby("team").roster.last()
        for t, ros in lan_rows.groupby("team").roster.last().items():
            if len(ros) == 3 and t in last_online.index:
                link.setdefault(frozenset(ros), last_online[t])
    out = []
    for ros in df.roster:
        out.append(link.get(frozenset(ros), ros) if ros and ros[0].startswith("L:") else ros)
    return out


def champs_drops():
    """dict team -> {map name: spot} for Championship teams (normalised names)."""
    out = {}
    for line in (DATA / "champs_drops.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        t, m, spot = line.split("|")
        out.setdefault(norm_team(t, "Y6S2_"), {})[MAPN[m]] = spot
    return out


def latest_roster(team, df):
    """Most recent roster of a team in the games table (online roster preferred, as it is the newest event)."""
    g = df[(df.team == team) & (df.roster.map(len) > 0)].sort_values("order")
    return g.roster.iloc[-1] if len(g) else ()
