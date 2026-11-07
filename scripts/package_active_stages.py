"""Preserve new stage evidence; original datasets and private correspondence excluded."""
import argparse,json,shutil
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from cranepilot.data import sha256
from cranepilot.reliability_pilot import FrozenReliabilityPilot
from scripts.run_external_benchmark import verify_hashes
from scripts.verify_finetune_campaign import original_files

def main(stage):
    out=Path('results')/f'{stage}_20261107';run=Path('artifacts/runs')/f'{stage}_20261107'
    if (out/'archive_manifest.json').exists() or (out/'evidence.zip').exists():raise FileExistsError('Retain earlier evidence')
    assert json.loads((out/'verification.json').read_text())['passed'];report=json.loads((out/'report.json').read_text());assert sha256(Path(report['document']))==report['document_sha256'];originals=original_files();assert originals['passed'];(out/'original_integrity.json').write_text(json.dumps(originals,indent=2)+'\n');FrozenReliabilityPilot()
    if stage=='active_qualification':
        p=json.loads((run/'provenance.json').read_text());assert p['status']=='complete';verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],run)
        for rel in ['provenance.json']+list(p['result_hashes']):assert (out/'frozen_run'/rel).read_bytes()==(run/rel).read_bytes()
    else:
        p=json.loads((run/'development.json').read_text());assert p['status']=='complete';verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],run)
        for rel in ['development.json']+list(p['result_hashes']):assert (out/'frozen_development'/rel).read_bytes()==(run/rel).read_bytes()
        for folder,dest in [('retrospective','frozen_replay'),('attribution','frozen_attribution')]:
            q=json.loads((run/folder/'provenance.json').read_text());assert q['status']=='complete';verify_hashes(q['result_hashes'],run/folder);assert not (out/dest).exists();shutil.copytree(run/folder,out/dest)
    files=[]
    for base in [out,Path('artifacts/reliability_bundle_20261107')]:files.extend(f for f in base.rglob('*') if f.is_file() and not f.name.startswith('.'))
    patterns=['scripts/*active_qualification*.py','scripts/*drive_conditioned*.py','scripts/*drive_attribution*.py','scripts/package_active_stages.py','scripts/modal_history.py','scripts/modal_reference_challenge.py','scripts/modal_backtest_gate.py','tests/test_active_qualification.py','tests/test_drive_conditioned.py','configs/active_qualification_20261107.json','configs/drive_conditioned_20261107.json','src/cranepilot/modal_pilot.py','src/cranepilot/reliability_pilot.py','src/cranepilot/data.py','src/cranepilot/physics.py','docs/ACTIVE_QUALIFICATION*.md','docs/DRIVE_CONDITIONED*.md','docs/DRIVE_ATTRIBUTION*.md','docs/RELIABILITY_BUNDLE_20261107.md','RESEARCH_STATUS_2026_11_07.md']
    for pattern in patterns:files.extend(f for f in Path('.').glob(pattern) if f.is_file())
    files=sorted(set(files));assert not any(str(f).startswith(('data/','../../')) or f.suffix=='.zip' for f in files);manifest=out/'archive_manifest.json';manifest.write_text(json.dumps(dict(scope='Own derived results, fitted weights, frozen selection, source, tests and documentation; original measurements/private correspondence excluded. Original main repository and datasets required for full replay. New drive-conditioned models are research parameters, not a released controller or portable API.',independent_confirmation=False,final_goal_achieved=False,files={str(f):dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in files}),indent=2)+'\n');files.append(manifest);target=out/'evidence.zip'
    with ZipFile(target,'x',ZIP_DEFLATED,compresslevel=6) as z:
        for f in files:z.write(f,str(f))
    with ZipFile(target) as z:
        assert z.testzip() is None and set(z.namelist())==set(map(str,files))
        for f in files:
            import hashlib
            assert hashlib.sha256(z.read(str(f))).hexdigest()==sha256(f)
    result=dict(bytes=target.stat().st_size,files=len(files),sha256=sha256(target),manifest_sha256=sha256(manifest),crc_and_all_entry_hashes_verified=True,original_measurements_included=False,independent_confirmation=False,final_goal_achieved=False);(out/'archive.json').write_text(json.dumps(result,indent=2)+'\n');print(stage,json.dumps(result))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['active_qualification','drive_conditioned']);main(parser.parse_args().stage)
