"""Verified code/weights/forecast evidence; no original raw measurements in archive."""
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from cranepilot.data import sha256
from scripts.verify_finetune_campaign import original_files
from scripts.run_external_benchmark import verify_hashes

OUT=Path('results/modal_campaign_20261107')
def main():
    if (OUT/'evidence.zip').exists() or (OUT/'archive_manifest.json').exists():raise FileExistsError('Retain earlier campaign archive')
    originals=original_files();assert originals['passed'];(OUT/'original_integrity.json').write_text(json.dumps(originals,indent=2)+'\n')
    for stem in ['modal_history','modal_confirmation','modal_reference_challenge']:
        out=Path(f'results/{stem}_20261107');assert json.loads((out/'verification.json').read_text())['passed'];report=json.loads((out/'report.json').read_text());assert sha256(Path(report['document']))==report['document_sha256']
    assert json.loads((OUT/'bundle_verification.json').read_text())['passed']
    for stem in ['modal_history','modal_confirmation']:
        run=Path(f'artifacts/runs/{stem}_20261107');p=json.loads((run/'provenance.json').read_text());verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],run)
    ref=Path('artifacts/runs/modal_reference_challenge_20261107');p=json.loads((ref/'development.json').read_text());q=json.loads((ref/'retrospective/provenance.json').read_text());verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],ref);verify_hashes(q['result_hashes'],ref/'retrospective')
    files=[]
    for stem in ['modal_history','modal_confirmation','modal_reference_challenge','modal_campaign']:
        files.extend(f for f in Path(f'results/{stem}_20261107').rglob('*') if f.is_file() and not f.name.startswith('.') and f.name!='package.log' and 'packaging_attempt_1' not in f.parts)
    for pattern in ['scripts/*modal*.py','tests/test_modal*.py','configs/modal*.json','docs/MODAL*.md','docs/AUTHOR_DATA_INTAKE_20261107_RU.md','src/cranepilot/modal_pilot.py','src/cranepilot/data.py','src/cranepilot/physics.py','RESEARCH_STATUS_2026_11_07.md','artifacts/modal_bundle_20261107/**/*','scripts/audit_inteco_holdouts.py','results/inteco_usage_audit_20261107/audit.json']:
        files.extend(f for f in Path('.').glob(pattern) if f.is_file())
    files=sorted(set(files));assert not any('data/raw' in str(f) or 'data/processed' in str(f) for f in files)
    manifest=OUT/'archive_manifest.json';manifest.write_text(json.dumps(dict(scope='Own code, parameters, derived predictions/metrics, exact replay evidence; original raw/processed datasets excluded. Repository dependencies and original source needed for full audit; portable bundle inference needs no training data.',final_goal_achieved=False,files={str(f):dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in files}),indent=2)+'\n');files.append(manifest);target=OUT/'evidence.zip'
    with ZipFile(target,'x',ZIP_DEFLATED,compresslevel=6) as z:
        for f in files:z.write(f,str(f))
    with ZipFile(target) as z:assert z.testzip() is None and set(z.namelist())==set(map(str,files))
    result=dict(bytes=target.stat().st_size,files=len(files),sha256=sha256(target),manifest_sha256=sha256(manifest),crc_verified=True,original_raw_data_included=False,final_goal_achieved=False)
    (OUT/'archive.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
