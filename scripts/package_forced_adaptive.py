"""Archive frozen forced-adaptation evidence, keeping upstream project separate."""
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from cranepilot.data import sha256


def main():
    root=Path(__file__).resolve().parents[1];out=root/'results/forced_adaptive_20261107';run=root/'artifacts/runs/forced_adaptive_20261107'
    prov=json.loads((run/'provenance.json').read_text())
    if prov['status']!='complete':raise ValueError('Completed run required')
    for mapping,prefix in [('guarded_hashes',root),('result_hashes',run)]:
        for name,digest in prov[mapping].items():
            if sha256(prefix/name)!=digest:raise ValueError(f'Changed evidence: {name}')
    for name in ['source_run','broad_run']:
        folder=root/prov['config'][name];old=json.loads((folder/'provenance.json').read_text())
        for relative,digest in old['checkpoint_hashes'].items():
            if sha256(folder/relative.replace('\\','/'))!=digest:raise ValueError('Parent weights changed')
    paths=[p for p in run.rglob('*') if p.is_file()]
    paths.extend(p for p in out.rglob('*') if p.is_file() and p.name not in ['archive.json','archive_manifest.json','evidence.zip'])
    for pattern in ['scripts/*forced_adaptive*.py','tests/test_forced_adaptive*.py','configs/forced_adaptive*.json','docs/FORCED_ADAPTIVE*.md']:
        paths.extend(root.glob(pattern))
    paths=sorted(set(paths));manifest={'scope':'This research stage only; restored upstream project/data/weights required for full replay. Local only.',
        'files':{str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in paths}}
    m=out/'archive_manifest.json';m.write_text(json.dumps(manifest,indent=2)+'\n');paths.append(m)
    dest=out/'evidence.zip'
    with ZipFile(dest,'x',ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths:z.write(p,str(p.relative_to(root)))
    with ZipFile(dest) as z:
        if z.testzip() is not None:raise ValueError('Archive CRC failed')
    info={'file':dest.name,'files':len(paths),'bytes':dest.stat().st_size,'sha256':sha256(dest),'crc_verified':True}
    (out/'archive.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))


if __name__=='__main__':main()
