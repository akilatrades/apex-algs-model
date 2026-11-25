"""
Turn scraped EEC scrim blocks into data/scrims.csv.

    python3 build_scrims.py incoming.json   # merge a fresh scrape (output of scrape/eec_all.js), then rebuild
    python3 build_scrims.py                 # just rebuild from data/scrims_json/

Each block is saved as data/scrims_json/<id>-<tag>.json. Only finished games are kept, and a
block with fewer than FULL_BLOCK games is listed as "re-check", so the next run scrapes it again.
Every game is checked: placements must be 1..N and points must equal placement points + kills.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from load import place_pts

DATA = Path(__file__).parent / "data"
JDIR = DATA / "scrims_json"
FULL_BLOCK = 6

# EEC display names -> the names the rest of the model uses
EEC_NAMES = {
    "Gaimin Gladiators": "GaiminGladiators",
    "Falcons": "Team Falcons",
    "Rex Regum Qeon": "Team RRQ",
    "Relove Deep Cross Gaming": "Relove DCG",
}


def block_problems(b):
    """Return a list of reasons the block can't be used yet (empty list = good)."""
    if b.get("error"):
        return [b["error"]]
    probs = []
    rows, games = b["rows"], b["games"]
    if not games or len(rows) < 10:
        return ["no games yet"]
    for gi in range(len(games)):
        pl = [r["placements"][gi] for r in rows]
        if any(p is None for p in pl):
            probs.append(f"game {gi + 1} not finished")
            continue
        if sorted(pl) != list(range(1, len(rows) + 1)):
            probs.append(f"game {gi + 1} placements are not 1..{len(rows)}")
        for r in rows:
            if r["points"][gi] - place_pts(r["placements"][gi]) < 0:
                probs.append(f"game {gi + 1} {r['team']}: points below placement points")
    for r in rows:
        if sum(p for p in r["points"] if p is not None) != r["total"]:
            probs.append(f"{r['team']}: game points don't add up to total")
        kills = sum(r["points"][g] - place_pts(r["placements"][g]) for g in range(len(games)) if r["placements"][g])
        if kills != r["kills"]:
            probs.append(f"{r['team']}: kills {kills} vs page {r['kills']}")
    return probs


def trim_unfinished(b):
    """Drop trailing games where some team has no placement yet (game in progress)."""
    if b.get("error") or not b.get("games"):
        return b
    n = len(b["games"])
    while n and any(r["placements"][n - 1] is None for r in b["rows"]):
        n -= 1
    b = dict(b, games=b["games"][:n])
    b["rows"] = [dict(r, placements=r["placements"][:n], points=r["points"][:n],
                      total=sum(r["points"][:n]),
                      kills=sum(r["points"][g] - place_pts(r["placements"][g]) for g in range(n)))
                 for r in b["rows"]]
    return b


def recheck_keys():
    """Blocks saved with fewer than FULL_BLOCK games - scrape these again."""
    return sorted(f.stem for f in JDIR.glob("*.json") if len(json.loads(f.read_text())["games"]) < FULL_BLOCK)


def save_new(incoming):
    saved, waiting = [], []
    JDIR.mkdir(exist_ok=True)
    if isinstance(incoming, dict):
        incoming = [incoming]
    for b in incoming:
        b = trim_unfinished(b)
        probs = block_problems(b)
        if probs:
            waiting.append((b["key"], probs[:3]))
            continue
        (JDIR / f"{b['key']}.json").write_text(json.dumps(b, indent=1))
        saved.append(b["key"])
    return saved, waiting


def rebuild():
    out = []
    for f in sorted(JDIR.glob("*.json")):
        b = json.loads(f.read_text())
        date = datetime.strptime(b["date"], "%b %d, %Y").strftime("%Y-%m-%d")
        lobby = "EEC-" + b["key"]
        for gi, g in enumerate(b["games"]):
            for r in b["rows"]:
                p = r["placements"][gi]
                out.append(dict(date=date, lobby=lobby, game=g["game"], map=g["map"],
                                team=EEC_NAMES.get(r["team"], r["team"]), placement=p,
                                kills=r["points"][gi] - place_pts(p)))
    df = pd.DataFrame(out).sort_values(["date", "lobby", "game", "placement"])
    df.to_csv(DATA / "scrims.csv", index=False)
    return df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        incoming = json.loads(Path(sys.argv[1]).read_text())
        if isinstance(incoming, str):
            incoming = json.loads(incoming)
        saved, waiting = save_new(incoming)
        print("saved:", saved or "none")
        for k, p in waiting:
            print("not saved yet:", k, "-", "; ".join(p))
    df = rebuild()
    print("re-check next run:", recheck_keys() or "none")
    print(f"scrims.csv: {df.groupby(['lobby', 'game']).ngroups} games, {df.lobby.nunique()} blocks, "
          f"{df.team.nunique()} teams, latest {df.date.max()}")
