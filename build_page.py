"""Inject results.json into the analysis page template -> out/apex_model.html"""
import json, datetime
from pathlib import Path
R = json.load(open("out/results.json"))
R = {k: v for k, v in R.items()}
R["meta"] = {k: v for k, v in R["meta"].items() if k != "stages"}
import rings, features as FE
R["zone"] = rings.page_data()
R["zone_teams"] = FE.champs_drops()
import ledger
R["ledger"] = ledger.status()
js = json.dumps(R, separators=(",", ":")).replace("</", "<\\/")
Path("out/apex_model.html").write_text(Path("page_template.html").read_text().replace("__ZONE_JS__", Path("zone_widget.js").read_text()).replace("__DATA__", js).replace("__DATE__", datetime.date.today().strftime("%-d %b %Y")))
print("wrote out/apex_model.html", len(js))
