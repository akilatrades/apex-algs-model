"""How much should a scrim game count? Runs itself once enough scrims have piled up (MIN_GAMES).
Hold out the newest third of scrim days; fit the ratings on all official games plus the older scrims at several weights;
score how well each predicts the held-out scrim games. Before then it just reports how many games it is waiting for.
Writes out/scrim_check.json. The weight in use stays scrims.SCRIM_WEIGHT until a human changes it; the report says
which weight the data prefers."""
import json
import numpy as np, pandas as pd
from math import lgamma
from load import load
import features as FE, model3 as M3, backtest3 as BT, scrims as SC

MIN_GAMES = 60
WEIGHTS = [0.0, 0.25, 0.5, 1.0]


def main():
    df, stages = load(include_scrims=True)
    sc = df[df.source == "scrim"]
    n = int(sc.game_id.nunique())
    out = dict(games=n, need=MIN_GAMES, in_use=SC.SCRIM_WEIGHT, status="waiting")
    if n >= MIN_GAMES:
        df = FE.attach(df, stages)
        if "roster" not in df: df["roster"] = [()] * len(df)
        df["roster"] = [r if isinstance(r, tuple) else () for r in df.roster]
        df["region"] = df.stage.map(BT.region_of)
        reg = df.dropna(subset=["region"]).groupby("team").region.agg(lambda s: s.mode().iloc[0])
        days = sorted(sc.stage.str.slice(6, 16).unique()); cut = days[int(len(days) * 2 / 3)]
        sd = df.stage.str.slice(6, 16)
        test = df[(df.source == "scrim") & (sd >= cut)]
        test = test[test.team.isin(set(df[df.source != "scrim"].team) | set(df[(df.source == "scrim") & (sd < cut)].team))]
        test = test[test.groupby("game_id").team.transform("size") >= 2]
        base = df[df.source.isin(["lan", "pro_league", "pl_s1"])]
        old = df[(df.source == "scrim") & (sd < cut)]
        uni = sum(-lgamma(len(g) + 1) for _, g in test.groupby("game_id"))
        lanset = set(base[base.source == "lan"].team)
        trials = []
        for w in WEIGHTS:
            tr = base if w == 0 or not len(old) else pd.concat([base, old.assign(weight=w)])
            F = BT.run_fit(tr, lanset, reg, **BT.CONFIGS["New model (players + spot + match point)"])
            trials.append(dict(weight=w, gain=round((M3.loglik(F, test) - uni) / test.game_id.nunique(), 4)))
            print(trials[-1], flush=True)
        best = max(trials, key=lambda r: r["gain"])
        out.update(status="done", test_games=int(test.game_id.nunique()), cut_day=cut, trials=trials, best=best["weight"],
                   agrees=bool(best["weight"] == SC.SCRIM_WEIGHT))
    json.dump(out, open("out/scrim_check.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    print(main())
