"""Package frozen history-backtest research; retain upstream sources separately."""
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from cranepilot.data import sha256


def main():
    root=Path(__file__).resolve().parents[1];out=root/'results/history_backtest_20261107';run=root/'artifacts/runs/history_backtest_20261107'
    prov=json.loads((run/'provenance.json').read_text())
    if prov['status']!='complete':raise ValueError('Completed study required')
    for mapping,prefix in [('guarded_hashes',root),('result_hashes',run)]:
        for name,digest in prov[mapping].items():
            if sha256(prefix/name)!=digest:raise ValueError(f'Changed evidence: {name}')
    for name in ['source_run','broad_run']:
        folder=root/prov['config'][name];old=json.loads((folder/'provenance.json').read_text())
        for relative,digest in old['checkpoint_hashes'].items():
            if sha256(folder/relative.replace('\\','/'))!=digest:raise ValueError('Parent weights changed')
    paths=[p for p in run.rglob('*') if p.is_file()]
    paths.extend(p for p in out.rglob('*') if p.is_file() and p.name not in ['archive.json','archive_manifest.json','evidence.zip'])
    for pattern in ['scripts/*history_backtest*.py','tests/test_history_backtest*.py','configs/history_backtest*.json','docs/HISTORY_BACKTEST*.md']:
        paths.extend(root.glob(pattern))
    paths=sorted(set(paths));manifest={'scope':'This local stage. Restored upstream code/data/weights required for complete replay.',
        'files':{str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in paths}}
    m=out/'archive_manifest.json';m.write_text(json.dumps(manifest,indent=2)+'\n');paths.append(m)
    dest=out/'evidence.zip'
    with ZipFile(dest,'x',ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths:z.write(p,str(p.relative_to(root)))
    with ZipFile(dest) as z:
        if z.testzip() is not None:raise ValueError('Archive integrity failed')
    info={'file':dest.name,'files':len(paths),'bytes':dest.stat().st_size,'sha256':sha256(dest),'crc_verified':True}
    (out/'archive.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))


if __name__=='__main__':main()
