"""
Nightly refresh. Usage:

    python3 update.py [scrape1.json scrape2.json ...]

1. merges any freshly scraped scrim blocks (output of scrape/eec_extract.js) into data/scrims_json/
2. rebuilds data/scrims.csv
3. runs the maths checks (tests.py) - stops if any fail
4. refits the model (analysis.py) and rebuilds both pages into out/
5. writes out/update_report.txt: what changed, so a human can glance at it

Exit code is non-zero if anything failed; the pages are not rebuilt in that case.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

import build_scrims as B
from load import load

ROOT = Path(__file__).parent
OUT = ROOT / "out"


def run(cmd):
    r = subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return r.stdout


def top_power():
    p = OUT / "results.json"
    if not p.exists():
        return {}
    res = json.loads(p.read_text())
    return {r["team"]: r for r in res["teams"] if r.get("in_field")}


def main(paths):
    report = []
    before_games = pd.read_csv(ROOT / "data/scrims.csv").groupby(["lobby", "game"]).ngroups
    before = top_power()

    for p in paths:
        saved, waiting = B.save_new(json.loads(Path(p).read_text()))
        report += [f"saved block {k}" for k in saved]
        report += [f"NOT saved {k}: {'; '.join(w)}" for k, w in waiting]
    df = B.rebuild()

    known = set(load()[0].team)
    from load import norm_team
    unknown = sorted({t for t in df.team if norm_team(t) not in known})
    if unknown:
        report.append("WARNING new scrim names not matched to ALGS teams (rated on scrims alone): " + ", ".join(unknown))

    after_games = df.groupby(["lobby", "game"]).ngroups
    report.append(f"scrim games: {before_games} -> {after_games} (latest {df.date.max()})")
    report.append("re-check next run: " + (", ".join(B.recheck_keys()) or "none"))

    report.append(run(["tests.py"]).strip().splitlines()[-1])
    run(["analysis.py"])
    run(["build_page.py"])
    run(["build_broadcast.py"])

    after = top_power()
    moves = []
    for t, r in after.items():
        a, b = r.get("p_win"), (before.get(t) or {}).get("p_win")
        if a is not None and b is not None and abs(a - b) >= 0.01:
            moves.append((abs(a - b), f"{t}: Champs win chance {b:.1%} -> {a:.1%}"))
    report += [m for _, m in sorted(moves, reverse=True)[:8]] or ["no team's Champs win chance moved by 1 point or more"]

    (OUT / "update_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main(sys.argv[1:])
