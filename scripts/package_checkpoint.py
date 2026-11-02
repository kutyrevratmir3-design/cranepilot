"""Compact, inspectable handoff archive; excludes credentials, environments and Git internals."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib
import json


def main():
    root=Path(__file__).resolve().parents[1]
    output=root.parent/"output/CranePilot_Checkpoint_2026-11-02.zip"
    selected=[]
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel=path.relative_to(root)
        if any(p in {".git",".venv",".local","__pycache__","cranepilot_research.egg-info"} for p in rel.parts):
            continue
        if path.suffix in {".pyc",".zip",".bundle",".aux",".log",".out"}:
            continue
        if rel.as_posix()=="local_settings.json":
            continue
        if rel.parts[:2]==("artifacts","runs"):
            # Retain all logs/metrics and the initial forecasting/control traces.
            if path.suffix in {".pt",".npz"} and rel.parts[2] not in {"extrapolation_pilot","control_pilot"}:
                continue
        selected.append(path)
    manifest={"scope":"source, processed CC BY data, metrics and selected pilot traces",
              "files":{str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in selected}}
    output.parent.mkdir(exist_ok=True)
    with ZipFile(output,"w",ZIP_DEFLATED,compresslevel=6) as z:
        for path in selected:
            z.write(path,"cranepilot/"+path.relative_to(root).as_posix())
        z.writestr("cranepilot/CHECKPOINT_MANIFEST.json",json.dumps(manifest,indent=2))
    with ZipFile(output) as z:
        bad=z.testzip()
        if bad:
            raise RuntimeError(f"Corrupt archive member: {bad}")
    print(json.dumps({"archive":str(output),"files":len(selected),"MiB":round(output.stat().st_size/2**20,2)}))


if __name__=="__main__":
    main()

