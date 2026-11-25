# Apex ALGS team model

Team ratings, Championship forecast and broadcast tool for the ALGS Year 6 Championship.

    python3 load.py             # parse + validate every game in data/
    python3 deep.py             # cross-check site season totals (games, kills, points) against the game data
    python3 tests.py            # 21 maths checks: scoring, model formula, simulator vs closed-form answers
    python3 analysis.py         # fit, backtest, forecast, deep stats -> out/results.json
    python3 build_page.py       # analysis page -> out/apex_model.html
    python3 build_broadcast.py  # broadcast desk -> out/broadcast.html
    python3 scrims.py data/scrims.csv   # validate a scrim file (see data/scrims_template.csv)

Data: 665 official games plus 12 scrims, all validated. LAN (128): 2025 Champs finals; Y6 Split 1 Playoffs (groups, survivor
stage, finals); Y6 Split 2 Playoffs (every stage). Pro League (500): Y6 Split 1 and Split 2, four regions. LCQ finals (37).
Scrims (12): Eternal Esports Club "Year 6 Champs" season. Sources: apexlegendsstatus.com, eternalesports.club.
Deep stats (data/deep/): team damage/knocks/assists, player lines, legend picks and comps (event and per team), weapons, drop spots.
data/enc/*.txt are compact files read in the browser (format in load._load_enc).

Model: Plackett-Luce strength from each game's finishing order; each team is shrunk toward the mean of its region and
level; teams with no LAN games get a measured "LAN debut" discount; Poisson kills by placement. Ratings are reported as
expected points per game in a Championship lobby.

Team names: group suffixes are stripped; ALIASES and SCOPED in load.py merge alternate names (each SCOPED mapping is
confirmed by the totals cross-check). Requires numpy, scipy, pandas.

## Nightly scrim refresh
1. In the browser, open any https://eternalesports.club page and run `scrape/list_blocks.js` to list played blocks.
2. For each block that is new or listed under "re-check" (fewer than 6 games), open
   `/lobbies/<id>/<tag>` and run `scrape/eec_extract.js`; save each result as a .json file.
3. `python3 update.py scrape1.json scrape2.json ...` merges the blocks, rebuilds data/scrims.csv,
   runs the maths checks, refits, and rebuilds out/apex_model.html and out/broadcast.html.
   out/update_report.txt says what changed. Nothing is rebuilt if a check fails.
