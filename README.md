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

## Model v3 (25 Nov 2026)
`model3.py` extends the team ratings with: player strengths (ratings follow players; ids from `data/deep/rosters_drops.txt`,
LAN and online id systems linked through each team's same-split roster), a penalty for sharing a landing spot with another
team in the game, and a match-point effect (used when fitting ratings, not in the title odds: teams on match point win no
more often than their rating says). `features.py` builds those columns. `backtest3.py` scores each idea on held-out games
(results in `out/backtest3.json`); team-by-map ratings and wider spreads for boom-or-bust teams (`swing.py`, `swing_check.py`)
made predictions worse and are shown as information only. Champs landing spots: `data/deep/champs_drops.txt`
(`scrape/als_champs_drops.js`). Rosters/drops per stage: `scrape/als_rosters_drops.js`.
The broadcast desk's live tracker updates ratings from games entered during the event and applies shared-spot penalties by map.

## Zone map (25 Nov 2026)
`rings.py` + `zone_widget.js`: where final zones end on each Champs map and where they're likely to end once ring 1 / ring 2
are known (kernel-weighted past games). Data: placement.gg's public ring archive (`data/rings/ring_sequences.txt`, built with
`scrape/placement_rings.js`; 6,042 games on current ring settings, ~420 tagged ALGS Year 6) on the frame of the
apexlegendsstatus.com landing-spot outlines (`data/rings/poi_outlines.txt`). Held-out check in `out/rings_check.json`:
after ring 2 the likeliest spot is right 63-82% of the time and the top three 97-99%. Credit placement.gg on anything public.

## Zone, kills, scrim weight, prediction ledger (25 Nov 2026)
- `zone_effect.py` / `zone_backtest.py`: links 308 ALGS games (apexlegendsstatus game ids, `data/rings/als_game_ids.txt`) to their ring sequences (`data/rings/algs_tagged.txt`, placement.gg) and every team's landing spot (`data/rings/als_game_drops.txt`). Landing closer to the zone helps; the effect sizes are in `out/zone_effect.json`. The Champs sim draws a final zone per game ("zone luck", centred so it does not double count ratings); the Broadcast Desk applies ring 1 / ring 2 to "Wins next game".
- `kills_joint.py`: kills are lumpier than Poisson (negative binomial, r ≈ 3.7), now used in all sims. A per-team "early vs late kills" slope did not help.
- `scrim_check.py`: tests the scrim weight automatically once 60 scrim games exist (nightly).
- `ledger.py`: locks the Champs odds on the first nightly run on/after 26 Jan 2027 (`data/ledger/y6_champs.json`, never edited after) and scores them once `data/ledger/y6_champs_results.txt` exists.
