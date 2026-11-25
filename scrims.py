"""
Scrim import. Drop a CSV at data/scrims.csv with one row per team per game:

    date,lobby,game,map,team,placement,kills
    2026-11-02,EEC-T1,1,Storm Point,Team Liquid,3,6

- `lobby` + `date` + `game` identify a game. Lobbies can have fewer than 20 teams.
- Team names are normalised with the same aliases as ALGS data (load.norm_team),
  so use the names apexlegendsstatus.com uses where you can.
- Scrim games enter the rating with weight SCRIM_WEIGHT (default 0.5): teams
  try things in scrims, so one scrim game counts as half an official game.
Run `python3 scrims.py data/scrims.csv` to validate a file without fitting anything.
"""
import sys
import pandas as pd
from load import norm_team, place_pts

SCRIM_WEIGHT = 0.5
SCRIM_ORDER = 3.5   # recency slot: after Y6S2 Pro League, before the latest LAN finals; edit if needed


def load_scrims(path, weight=SCRIM_WEIGHT, order=SCRIM_ORDER):
    s = pd.read_csv(path)
    need = {"date", "lobby", "game", "team", "placement", "kills"}
    missing = need - set(s.columns)
    if missing:
        raise ValueError(f"scrims file missing columns: {sorted(missing)}")
    s["team"] = s.team.astype(str).map(norm_team)
    s["placement"] = s.placement.astype(int)
    s["kills"] = s.kills.astype(int)
    s["stage"] = "SCRIM_" + s.date.astype(str) + "_" + s.lobby.astype(str)
    s["game_id"] = s.stage + "_G" + s.game.astype(str)
    s["order"] = float(order)
    s["map"] = s.get("map", "unknown")
    s["page_total"] = None
    s["points"] = s.placement.map(place_pts) + s.kills
    s["source"] = "scrim"
    s["weight"] = weight
    bad = [gid for gid, g in s.groupby("game_id") if sorted(g.placement) != list(range(1, len(g) + 1))]
    if bad:
        raise ValueError(f"{len(bad)} scrim games have placements that are not 1..N, e.g. {bad[:3]}")
    return s[["stage", "order", "game", "game_id", "map", "team", "placement", "kills", "page_total", "points", "source", "weight"]]


if __name__ == "__main__":
    df = load_scrims(sys.argv[1] if len(sys.argv) > 1 else "data/scrims_template.csv")
    print(f"OK: {df.game_id.nunique()} scrim games, {df.team.nunique()} teams")
