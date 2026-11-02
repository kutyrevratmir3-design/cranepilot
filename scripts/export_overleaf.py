"""Flatten the manuscript for safe, additive upload into the existing Overleaf project."""
from pathlib import Path
import hashlib
import json
import re
import shutil


def main():
    root=Path(__file__).resolve().parents[1]
    paper=root/"paper"; out=root/".local/overleaf_sync"
    out.mkdir(parents=True,exist_ok=True)
    main=(paper/"main.tex").read_text()
    body=(paper/"sections/body.tex").read_text()
    for name in ["forecast_table","control_table","all_family_neural_table","passive_rollout_table"]:
        body=body.replace("\\input{"+name+"}",(paper/f"{name}.tex").read_text())
    main=main.replace("\\input{sections/body}",body).replace("\\bibliography{refs}","\\bibliography{cranepilot_refs}")
    figures=re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",main)
    for rel in figures:
        source=paper/rel
        dest="cp_"+source.name
        shutil.copyfile(source,out/dest)
        main=main.replace("{"+rel+"}","{"+dest+"}")
    (out/"cranepilot_main.tex").write_text(main,encoding="utf-8")
    shutil.copyfile(paper/"refs.bib",out/"cranepilot_refs.bib")
    files=[out/"cranepilot_main.tex",out/"cranepilot_refs.bib",*[out/("cp_"+Path(p).name) for p in figures]]
    manifest={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out/"export_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps({"directory":str(out),"files":[str(p) for p in files]},indent=2))


if __name__=="__main__":
    main()

