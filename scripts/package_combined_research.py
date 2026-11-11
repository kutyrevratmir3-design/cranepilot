"""Package this local research stage, including its retained failed attempt."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'results/combined_stress_20261107'
    runs = ['combined_stress_20261107', 'combined_stress_20261107_replayfix', 'combined_offgrid_20261107']
    for name in runs[1:]:
        directory = root/'artifacts/runs'/name
        provenance = json.loads((directory/'provenance.json').read_text())
        assert provenance['status'] == 'complete'
        for relative,digest in provenance['guarded_hashes'].items():
            assert sha(root/relative) == digest, relative
        for relative,digest in provenance['result_hashes'].items():
            assert sha(directory/relative) == digest, relative
    paths = []
    for name in runs:
        paths.extend(p for p in (root/'artifacts/runs'/name).rglob('*') if p.is_file())
    for pattern in ['scripts/combined_*.py','scripts/*combined*.py','tests/test_combined_*.py',
                    'configs/combined_*.json','docs/COMBINED_*.md']:
        paths.extend(root.glob(pattern))
    paths.extend(p for p in out.rglob('*') if p.is_file() and p.name not in ['evidence.zip','archive.json','archive_manifest.json'])
    paths = sorted(set(paths))
    manifest = {'scope':'New stage, failed attempt and off-grid follow-up. Full replay also needs restored upstream project/data/weights. No external publication.',
                'files':{str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths}}
    m = out/'archive_manifest.json';m.write_text(json.dumps(manifest,indent=2)+'\n');paths.append(m)
    archive = out/'evidence.zip'
    with ZipFile(archive,'x',ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths: z.write(p,str(p.relative_to(root)))
    with ZipFile(archive) as z: assert z.testzip() is None
    metadata = {'file':archive.name,'files':len(paths),'bytes':archive.stat().st_size,'sha256':sha(archive),'crc_verified':True}
    (out/'archive.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata,indent=2))


if __name__ == '__main__': main()
