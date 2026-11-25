"""Parse raw ALGS stage data into a tidy long table and validate it."""
from pathlib import Path
import pandas as pd

# ALGS placement points (Year 4+ scoring), +1 per kill
PLACE_PTS = {1: 12, 2: 9, 3: 7, 4: 5, 5: 4, 6: 3, 7: 3, 8: 2, 9: 2, 10: 2,
             11: 1, 12: 1, 13: 1, 14: 1, 15: 1}


def place_pts(p):
    return PLACE_PTS.get(int(p), 0)


import re

ALIASES = {"SR": "ShopifyRebellion", "NinjasinPyjamas": "NinjasInPyjamas", "SeventhLetterGg": "SLG",
           "HABNW": "HAB", "AURORA": "Aurora",
           "TLN Pirates": "TLN PIRATES", "GenG Esports": "Gen.G Citadel", "AG Global": "AG GLOBAL", "WGR NEO": "WhiteGrimReaper"}


# Names that the site shows for a team on certain Pro League days only. Each mapping was confirmed by arithmetic:
# with it, the team's games, kills and points match the site's season totals exactly (see deep.py crosscheck).
SCOPED = {"PL_NA_": {"JD GAMING": "FFS", "SSS": "NRG"},
          "PL_EMEA_": {"BANDITS": "PVISION", "D4C": "PlayFor2nd", "Majin": "JOB", "GEI": "Fear the Lamb"},
          "PL_APN_": {"O2esports": "SLEEPY", "ORB GARDEN": "4QuattroFormaggi", "REIGA": "UNFAZED", "kawaiipaimonster": "ETiAS"},
          "PL_APS_D8": {"TitanEsportsClub": "TEC-Academy", "TEC Academy": "TEC Black", "GoaT": "DreamFire", "Finishers": "KAKKOII"},
          "PL_APS_D9": {"TitanEsportsClub": "TEC-Academy", "SMP": "TitanEsportsClub", "Comeback": "Outside", "AELIX": "ASC3NT"},
          # Roster-confirmed: the same three players appear under both names in the site's player tables.
          "Y6S2_": {"7VEN": "Daylight Savings"},
          "Y6S1_": {"ZEDI ESPORTS": "Geekay Esports", "Geekay Esports": "DKF", "JD GAMING": "FFS", "Team Nemesis": "Stallions",
                    "ELITE Esports": "PVISION", "TriniTY": "SBI eSports", "Virtus.pro": "TOP"},
          "PL1_EMEA": {"TLN PIRATES": "Aurora", "EX TLN Pirates": "TLN PIRATES", "Aurora": "Aurora (S1 roster)", "Virtus.pro": "TOP"},
          "PL1_APS": {"CPR": "CPR (S1 roster)"}}
MAPCODE = {"S": "Storm Point", "W": "World's Edge", "E": "E-District", "O": "Olympus", "B": "Broken Moon", "K": "Kings Canyon", "?": "unknown"}


def norm_team(name, stage=""):
    name = re.sub(r" [ABCD]$", "", name.strip())   # strip Pro League group letter suffix
    for pre, m in SCOPED.items():
        if stage.startswith(pre) and name in m:
            return m[name]
    return ALIASES.get(name, name)


DATA = Path(__file__).parent / "data"
FILES = ["raw_stages.txt", "lan_group_stages.txt", "lcq.txt", "pro_league_na.txt", "pro_league_emea.txt", "pro_league_apac_n.txt", "pro_league_apac_s.txt"]


def load(paths=None, include_scrims=False):
    default = paths is None
    paths = paths or [DATA / f for f in FILES if (DATA / f).exists()]
    rows, stages = [], []
    for p in paths:
        r, s = _load_one(p)
        rows += r; stages += s
    if default:
        for p in sorted((DATA / "enc").glob("*.txt")):
            r, s = _load_enc(p)
            rows += r; stages += s
    df = pd.DataFrame(rows)
    df["points"] = df.placement.map(place_pts) + df.kills
    df["source"] = df.stage.map(lambda s: "pro_league" if s.startswith("PL_") else ("pl_s1" if s.startswith("PL1_") else ("lcq" if s.startswith("LCQ_") else "lan")))
    if include_scrims and (DATA / "scrims.csv").exists():
        from scrims import load_scrims
        sc = load_scrims(DATA / "scrims.csv")
        df = pd.concat([df, sc], ignore_index=True)
    return df, stages


def _load_enc(path):
    """Compact files written from the browser: '@PREFIX|order|baseurl', then '#page|map letters' and
    'team|placements as letters (a=1st)|kills in base 36|page total' lines."""
    rows, stages = [], []
    for line in Path(path).read_text().splitlines():
        line = line.rstrip()
        if not line: continue
        if line.startswith("@"):
            prefix, order, base = line[1:].split("|"); continue
        if line.startswith("#"):
            page, maps = line[1:].split("|")
            tag = page.split("/")[0].replace("Day", "D") if "Finals" not in page else "FINALS"
            if page.endswith("Survivor"): tag = "SURVIVOR"
            o = float(order) + (0.2 if tag == "FINALS" else 0)
            stage = dict(stage=f"{prefix}_{tag}", order=o, url=base + page, maps=[MAPCODE[c] for c in maps]); stages.append(stage); continue
        team, pl, kl, tot = line.split("|")
        team = norm_team(team, stage["stage"])
        assert len(pl) == len(kl) == len(stage["maps"]), (stage["stage"], team)
        for g, (p, k) in enumerate(zip(pl, kl), 1):
            if p == "-": continue
            rows.append(dict(stage=stage["stage"], order=stage["order"], game=g, game_id=f'{stage["stage"]}_G{g}', map=stage["maps"][g - 1],
                             team=team, placement=ord(p) - 96, kills=int(k, 36), page_total=int(tot)))
    return rows, stages


def _load_one(path):
    rows, stages = [], []
    stage = None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or (line.startswith("#") and not line.startswith("## ")):
            continue
        if line.startswith("## "):
            sid, order, url, maps = line[3:].split("|")
            stage = dict(stage=sid, order=float(order), url=url, maps=[m.strip() for m in maps.split(",")])
            stages.append(stage)
            continue
        team, pl, kl, tot = line.split("|")
        team = norm_team(team, stage["stage"])
        pl, kl = pl.replace(",", " ").split(), kl.replace(",", " ").split()   # "-" = team has no result for that game
        assert len(pl) == len(kl) == len(stage["maps"]), (stage["stage"], team)
        for g, (p, k) in enumerate(zip(pl, kl), 1):
            if p == "-":
                continue
            p, k = int(p), int(k)
            rows.append(dict(stage=stage["stage"], order=stage["order"], game=g,
                             game_id=f'{stage["stage"]}_G{g}', map=stage["maps"][g - 1],
                             team=team, placement=p, kills=k, page_total=(None if tot.strip() == "?" else int(tot))))
    return rows, stages


def validate(df):
    problems = []
    for gid, g in df.groupby("game_id"):
        if sorted(g.placement) != list(range(1, len(g) + 1)):
            problems.append(f"{gid}: placements not a permutation {sorted(g.placement)}")
    tot = df.groupby(["stage", "team"]).agg(calc=("points", "sum"), page=("page_total", "first"))
    bad = tot[tot.page.notna() & (tot.calc != tot.page)]
    for (s, t), r in bad.iterrows():
        problems.append(f"{s} {t}: computed {r.calc} vs page {r.page}")
    return problems


if __name__ == "__main__":
    df, stages = load()
    probs = validate(df)
    print(f"{df.game_id.nunique()} games, {df.team.nunique()} teams, {len(df)} team-games")
    print("\n".join(probs) if probs else "ALL CHECKS PASS")
