"""Package own reproducible evidence; original external MAT files stay on their source host."""
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from cranepilot.data import sha256

OUT=Path('results/external_benchmark_20261107')

def main():
    out=OUT;run=out/'frozen_run';p=json.loads((run/'provenance.json').read_text());r=json.loads((out/'report.json').read_text())
    assert p['status']=='complete' and json.loads((out/'verification.json').read_text())['passed']
    for mapping,base in [('guarded_hashes',Path('.')),('result_hashes',run),('training_hashes',run)]:
        for name,digest in p[mapping].items():assert sha256(base/name)==digest,name
    for path,key in [(run/'provenance.json','provenance_sha256'),(Path(r['document']),'document_sha256'),(Path('scripts/report_external_benchmark.py'),'report_source_sha256'),(out/'verification.json','verification_sha256'),(out/'posthoc_diagnostics.json','posthoc_diagnostics_sha256'),(Path('scripts/report_angular_control.py'),'aggregation_helper_sha256')]:
        assert sha256(path)==r[key],path
    excluded={'evidence.zip','archive.json','archive_manifest.json'}
    if any((out/n).exists() for n in excluded):raise FileExistsError('Retain previous evidence archive')
    files=[p for p in out.rglob('*') if p.is_file() and p.name not in excluded]
    for pattern in ['scripts/*external*.py','tests/test_external_benchmark.py','configs/external_benchmark_20261107.json','docs/EXTERNAL_BENCHMARK*.md','RESEARCH_STATUS_2026_11_07.md','artifacts/research_bundle_20261107/**/*']:
        files.extend(p for p in Path('.').glob(pattern) if p.is_file())
    files=sorted(set(files));assert not any('data/raw' in str(p) for p in files)
    manifest=out/'archive_manifest.json';manifest.write_text(json.dumps(dict(scope='Own code, checkpoints, aggregate metrics and derived numerical forecasts; original external MAT/source files excluded; repository code and verified external download required.',files={str(p):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in files}),indent=2)+'\n');files.append(manifest)
    target=out/'evidence.zip'
    with ZipFile(target,'x',ZIP_DEFLATED,compresslevel=6) as z:
        for path in files:z.write(path,str(path))
    with ZipFile(target) as z:assert z.testzip() is None and set(z.namelist())==set(map(str,files))
    info=dict(file=target.name,bytes=target.stat().st_size,files=len(files),sha256=sha256(target),manifest_sha256=sha256(manifest),crc_verified=True,original_external_mat_files_included=False)
    (out/'archive.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))

if __name__=='__main__':main()
