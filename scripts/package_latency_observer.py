"""Preserve timestamp-aware code, all fitted references and unreselected results."""
import json,shutil,hashlib
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from cranepilot.data import sha256
from cranepilot.latency_pilot import FrozenLatencyPilot
from scripts.run_latency_observer import RUN,OUT,LOCKED
from scripts.run_external_benchmark import verify_hashes
from scripts.verify_finetune_campaign import original_files

def main():
    if (OUT/'archive_manifest.json').exists() or (OUT/'evidence.zip').exists():raise FileExistsError('Preserve earlier archive')
    p=json.loads((RUN/'development.json').read_text());q=json.loads((RUN/'retrospective/provenance.json').read_text());v=json.loads((OUT/'verification.json').read_text());r=json.loads((OUT/'report.json').read_text());assert v['passed'] and v['development_sha256']==sha256(RUN/'development.json') and r['document_sha256']==sha256(Path(r['document']));verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);verify_hashes(q['result_hashes'],RUN/'retrospective');FrozenLatencyPilot();originals=original_files();assert originals['passed'];(OUT/'original_integrity.json').write_text(json.dumps(originals,indent=2)+'\n')
    assert json.loads((OUT/'reference_challenge/verification.json').read_text())['passed']
    for rel in ['development.json']+list(p['result_hashes']):assert (OUT/'frozen_development'/rel).read_bytes()==(RUN/rel).read_bytes()
    assert not (OUT/'frozen_replay').exists();shutil.copytree(RUN/'retrospective',OUT/'frozen_replay');files=[f for root in [OUT,Path('artifacts/latency_bundle_20261108')] for f in root.rglob('*') if f.is_file() and not f.name.startswith('.')]+LOCKED
    for pattern in ['scripts/*latency_observer*.py','scripts/build_latency_bundle.py','docs/LATENCY*.md','scripts/regime_fusion.py','scripts/modal_history.py','scripts/modal_reference_challenge.py','scripts/modal_backtest_gate.py','scripts/active_qualification.py','scripts/drive_conditioned.py','src/cranepilot/regime_pilot.py','src/cranepilot/modal_pilot.py','src/cranepilot/reliability_pilot.py','src/cranepilot/data.py','src/cranepilot/physics.py','RESEARCH_STATUS_2026_11_07.md','docs/AUTHOR_DATA_INTAKE_20261107_RU.md']:files.extend(f for f in Path('.').glob(pattern) if f.is_file())
    files=sorted(set(files));assert not any(str(f).startswith('data/') or f.suffix=='.zip' for f in files);manifest=OUT/'archive_manifest.json';manifest.write_text(json.dumps(dict(scope='Own code/fitted ARX parameters, frozen metadata-age adapter, derived forecasts and synthetic delayed-angle qualification. Original datasets/private correspondence excluded. Portable inference needs main source/dependencies but no raw or ancestral runs. Full replay requires original recordings. Known-age compensation is not unknown-lag identification.',independent_confirmation=False,final_goal_achieved=False,files={str(f):dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in files}),indent=2)+'\n');files.append(manifest);target=OUT/'evidence.zip'
    with ZipFile(target,'x',ZIP_DEFLATED,compresslevel=6) as z:
        for f in files:z.write(f,str(f))
    with ZipFile(target) as z:
        assert z.testzip() is None and set(z.namelist())==set(map(str,files))
        for f in files:assert hashlib.sha256(z.read(str(f))).hexdigest()==sha256(f)
    result=dict(bytes=target.stat().st_size,files=len(files),sha256=sha256(target),manifest_sha256=sha256(manifest),crc_and_all_entry_hashes_verified=True,original_measurements_included=False,independent_confirmation=False,final_goal_achieved=False);(OUT/'archive.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
