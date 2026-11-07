"""Freeze reliability development evidence without copying original measurements."""
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from cranepilot.data import sha256
from cranepilot.reliability_pilot import FrozenReliabilityPilot
from scripts.run_external_benchmark import verify_hashes
from scripts.verify_finetune_campaign import original_files

OUT=Path('results/modal_backtest_gate_20261107');RUN=Path('artifacts/runs/modal_backtest_gate_20261107')
def main():
    if (OUT/'archive_manifest.json').exists() or (OUT/'evidence.zip').exists():raise FileExistsError('Retain earlier evidence')
    for f in ['verification.json','bundle_verification.json']:assert json.loads((OUT/f).read_text())['passed']
    p=json.loads((RUN/'development.json').read_text());q=json.loads((RUN/'retrospective/provenance.json').read_text());verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);verify_hashes(q['result_hashes'],RUN/'retrospective');r=json.loads((OUT/'report.json').read_text());assert sha256(Path(r['document']))==r['document_sha256'];FrozenReliabilityPilot();originals=original_files();assert originals['passed'];(OUT/'original_integrity.json').write_text(json.dumps(originals,indent=2)+'\n')
    files=[]
    for base in [OUT,Path('results/modal_diagnostics_20261107'),Path('results/data_access_followup_20261107'),Path('artifacts/reliability_bundle_20261107')]:files.extend(f for f in base.rglob('*') if f.is_file() and not f.name.startswith('.'))
    for pattern in ['scripts/*modal_backtest_gate*.py','scripts/*reliability*.py','scripts/diagnose_modal_transfer.py','scripts/modal_history.py','scripts/modal_reference_challenge.py','tests/test_modal_backtest_gate.py','tests/test_reliability_pilot.py','configs/modal_backtest_gate_20261107.json','src/cranepilot/reliability_pilot.py','src/cranepilot/modal_pilot.py','src/cranepilot/data.py','src/cranepilot/physics.py','docs/MODAL_BACKTEST_GATE*.md','docs/MODAL_FAILURE_DIAGNOSIS_20261107_RU.md','docs/DATA_ACCESS_FOLLOWUP_20261107.md','docs/RELIABILITY_BUNDLE_20261107.md','docs/AUTHOR_DATA_INTAKE_20261107_RU.md','RESEARCH_STATUS_2026_11_07.md']:
        files.extend(f for f in Path('.').glob(pattern) if f.is_file())
    files=sorted(set(files));assert not any('data/raw' in str(f) or 'data/processed' in str(f) for f in files)
    manifest=OUT/'archive_manifest.json';manifest.write_text(json.dumps(dict(scope='Own frozen code/parameters/derived forecasts and retrospective diagnosis; original raw data excluded. Portable bundle inference requires main repository/dependencies but no dataset or training. Prior repository stages and original source required for full replay.',independent_confirmation=False,final_goal_achieved=False,files={str(f):dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in files}),indent=2)+'\n');files.append(manifest);target=OUT/'evidence.zip'
    with ZipFile(target,'x',ZIP_DEFLATED,compresslevel=6) as z:
        for f in files:z.write(f,str(f))
    with ZipFile(target) as z:assert z.testzip() is None and set(z.namelist())==set(map(str,files))
    result=dict(bytes=target.stat().st_size,files=len(files),sha256=sha256(target),manifest_sha256=sha256(manifest),crc_verified=True,original_measurements_included=False,independent_confirmation=False,final_goal_achieved=False);(OUT/'archive.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
