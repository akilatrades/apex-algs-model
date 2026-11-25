"""Inject results.json into the analysis page template -> out/apex_model.html"""
import json
from pathlib import Path
R = json.load(open("out/results.json"))
R = {k: v for k, v in R.items()}
R["meta"] = {k: v for k, v in R["meta"].items() if k != "stages"}
js = json.dumps(R, separators=(",", ":")).replace("</", "<\\/")
Path("out/apex_model.html").write_text(Path("page_template.html").read_text().replace("__DATA__", js))
print("wrote out/apex_model.html", len(js))
