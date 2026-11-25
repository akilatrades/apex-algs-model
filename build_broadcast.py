"""Broadcast desk page for the Y6 Championship -> out/broadcast.html"""
import json, datetime
from pathlib import Path
from load import load
import features as FE
R = json.load(open("out/results.json"))
DROPS = FE.champs_drops()
df, stages = load(include_scrims=True)
T = {t["team"]: t for t in R["teams"]}
field = [t["team"] for t in R["teams"] if t["in_field"]]
EV = [("2025 Champs", lambda s: s.startswith("Y5S2")), ("Y6 S1 Pro League", lambda s: s.startswith("PL1_")), ("Y6 S1 Playoffs", lambda s: s.startswith("Y6S1")),
      ("Y6 S2 Pro League", lambda s: s.startswith("PL_")), ("Y6 S2 Playoffs", lambda s: s.startswith("Y6S2")),
      ("Y6 S2 LCQ final", lambda s: s.startswith("LCQ_")), ("Champs scrims", lambda s: s.startswith("SCRIM_"))]
df["event"] = df.stage.map(lambda s: next(n for n, f in EV if f(s)))
hist = {}
for (team, ev), g in df.groupby(["team", "event"]):
    hist.setdefault(team, {})[ev] = [len(g), round(g.points.mean(), 2), round(g.placement.mean(), 1), int((g.placement == 1).sum()), round(g.kills.mean(), 2)]
fin = {}
for st, label in [("Y5S2_CHAMPS_FINALS", "2025 Champs"), ("Y6S1_PLAYOFFS_FINALS", "Y6 S1 Playoffs"), ("Y6S2_PLAYOFFS_FINALS", "Y6 S2 Playoffs")]:
    tot = df[df.stage == st].groupby("team").points.sum().sort_values(ascending=False)
    for i, (t, p) in enumerate(tot.items(), 1): fin.setdefault(t, []).append([label, i, int(p)])
champions = {"2025 Champs": "Oblivion", "Y6 S1 Playoffs": "UNLIMIT", "Y6 S2 Playoffs": "Team Liquid"}
lcq_fin = {}
for st in ["LCQ_NA_FINALS", "LCQ_EMEA_FINALS", "LCQ_APN_FINALS", "LCQ_APS_FINALS"]:
    for i, t in enumerate(dict.fromkeys(df[df.stage == st].team), 1): lcq_fin[t] = i     # file order = site standings order
piv = df[df.team.isin(field)].pivot_table(index="game_id", columns="team", values="placement")
h2h = {}
for a in field:
    for b in field:
        if a < b:
            m = piv[[a, b]].dropna()
            if len(m): h2h[f"{a}|{b}"] = [len(m), int((m[a] < m[b]).sum())]
pl_by_team = {}
for p in R["players"]: pl_by_team.setdefault(p["team"], []).append(p)
def row(t):
    d = T[t]
    return dict(team=t, region=d["region"], s=d["strength"], ks=d["kill_skill"], pow=d["power"], plo=d["power_lo"], phi=d["power_hi"], wg=d["win_game"],
                g=d["games"], lg=d["lan_games"], ppg=round(d["ppg"], 2), ap=round(d["avg_place"], 1), w=d["wins"], kpg=round(d["kpg"], 2), nolan=d["no_lan"],
                hist=hist.get(t, {}), fin=fin.get(t, []), lcq=lcq_fin.get(t), deep=R["deep_teams"].get(t, {}), players=pl_by_team.get(t, []),
                tl=R["team_legends"].get(t), maps=R["team_maps"].get(t, {}), sd=round(d.get("sd", 0.1), 4),
                sw=R["swing"].get(t), mp=R["mp_team"].get(t), bu=R["built"].get(t), cp=R["clash_partners"].get(t, []), spots=DROPS.get(t, {}),
                plain=R["champs_plain"].get(t))
F = []
for t in field:
    r = row(t); r.update(pw=T[t]["p_win"], pf=T[t]["p_finals"], path=T[t]["path"]); F.append(r)
recent = set(df[df.stage.str.startswith("PL_") | df.stage.str.startswith("LCQ_") | df.stage.str.startswith("Y6S2")].team)
missed = [row(t["team"]) for t in R["teams"] if t["team"] in recent and t["team"] not in field]
for m in missed:
    for k in ["players", "hist", "fin", "tl", "maps", "sw", "mp", "bu", "cp", "spots", "plain"]: m.pop(k)
data = dict(field=F, missed=missed, h2h=h2h, kb=R["kill_base"], pts=R["place_pts"], champions=champions, legends=R["legends"], comps=R["comps"], guns=R["guns"], poi=R["poi"],
            meta=dict(games=R["meta"]["n_games"], avg=R["meta"]["avg_power"], fetched=R["meta"]["fetched"], pps=R["meta"]["pts_per_strength"]),
            v3=dict(clash=R["v3"]["clash"], pts_clash=R["v3"]["pts_clash"], rot=R["v3"]["rot"], rot_groups=R["v3"]["rot_groups"]), mp_all=R["mp_all"])
js = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
Path("out/broadcast.html").write_text(Path("broadcast_template.html").read_text().replace("__DATA__", js).replace("__DATE__", datetime.date.today().strftime("%-d %b %Y")))
print("field", len(F), "missed", len(missed), "bytes", len(js))
print([(m["team"], round(m["pow"], 2), m["lcq"]) for m in missed[:8]])
