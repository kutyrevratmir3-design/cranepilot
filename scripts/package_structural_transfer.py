"""Package verified control evidence, synthetic raw traces and frozen model bundle."""
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from cranepilot.data import sha256


def main():
    out=Path('results/structural_transfer_20261107');run=out/'frozen_run';report=json.loads((out/'report.json').read_text())
    prov=json.loads((run/'provenance.json').read_text())
    if prov['status']!='complete':raise ValueError('Completed run required')
    for key,base in [('guarded_hashes',Path('.')),('result_hashes',run)]:
        for name,digest in prov[key].items():
            if sha256(base/name)!=digest:raise ValueError(f'Changed {name}')
    checks=[(run/'provenance.json','provenance_sha256'),(Path(report['document']),'document_sha256'),
        (Path('scripts/report_structural_transfer.py'),'report_source_sha256'),(out/'verification.json','verification_sha256'),
        (Path('scripts/report_angular_control.py'),'aggregation_helper_sha256'),(out/'runtime_batch1.csv','runtime_batch1_sha256'),(out/'independent_metrics.json','independent_metrics_sha256')]
    for path,key in checks:
        if sha256(path)!=report[key]:raise ValueError(f'Changed report input: {path}')
    if not json.loads((out/'verification.json').read_text())['passed']:raise ValueError('Verification failed')
    exclude={'evidence.zip','archive.json','archive_manifest.json'}
    if any((out/name).exists() for name in exclude):raise FileExistsError('Preserve earlier archive')
    files=[p for p in out.rglob('*') if p.is_file() and p.name not in exclude]
    for pattern in ['scripts/*structural_transfer*.py','scripts/adaptive_actuator.py','scripts/run_adaptive_actuator.py','scripts/actuator_identification.py','scripts/angular_control.py','tests/test_structural_transfer.py',
        'configs/structural_transfer_20261107.json','docs/STRUCTURAL_TRANSFER*.md','RESEARCH_STATUS_2026_11_07.md',
        'artifacts/research_bundle_20261107/**/*']:
        files.extend(p for p in Path('.').glob(pattern) if p.is_file())
    files=sorted(set(files));manifest=out/'archive_manifest.json'
    manifest.write_text(json.dumps({'scope':'New acceleration-command ablation, complete synthetic traces and frozen weights; repository code still required',
        'files':{str(p):{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in files}},indent=2)+'\n')
    files.append(manifest);target=out/'evidence.zip'
    with ZipFile(target,'x',ZIP_DEFLATED,compresslevel=6) as z:
        for path in files:z.write(path,str(path))
    with ZipFile(target) as z:
        assert z.testzip() is None and set(z.namelist())==set(map(str,files))
    info=dict(file=target.name,files=len(files),bytes=target.stat().st_size,sha256=sha256(target),manifest_sha256=sha256(manifest),crc_verified=True)
    (out/'archive.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))

if __name__=='__main__':main()
