"""Package the Nov 9 public diagnostics with their prior evidence commitment."""
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

from cranepilot.data import sha256
from scripts.run_external_benchmark import verify_hashes, write

OUT = Path('results/timing_tradeoffs_20261109')
STAGE_DIRS = [Path('results')/name for name in [
    'timestamp_offset_20261109', 'timestamp_offset_audit_20261109',
    'timestamp_offset_report_20261109', 'dense_horizons_20261109',
    'fresh_session_design_20261109', 'timestamp_offset_recovery_20261109',
]]
SOURCE = [Path(name) for name in [
    'configs/timestamp_offset_20261109.json',
    'docs/TIMESTAMP_OFFSET_DESIGN_20261109.md',
    'docs/TIMESTAMP_OFFSET_2026_11_09_RU.md',
    'docs/TIMESTAMP_OFFSET_RECOVERY_20261109.md',
    'docs/DENSE_HORIZONS_2026_11_09_RU.md',
    'docs/FRESH_SESSION_DESIGN_20261109_RU.md',
    'scripts/timestamp_offset.py', 'tests/test_timestamp_offset.py',
    'scripts/verify_timestamp_offset.py', 'scripts/report_timestamp_offset.py',
    'scripts/finalize_timestamp_offset.py',
    'scripts/report_dense_horizons.py', 'scripts/plan_fresh_sessions.py',
    'scripts/package_timing_tradeoff_evidence.py',
]]


def main():
    v = json.loads((STAGE_DIRS[1]/'verification.json').read_text())
    p = json.loads((STAGE_DIRS[0]/'provenance.json').read_text())
    old = json.loads((STAGE_DIRS[3]/'verification.json').read_text())
    planning = json.loads((STAGE_DIRS[4]/'planning.json').read_text())
    assert p['status'] == 'complete' and v['passed'] and old['passed']
    assert not p['independent_confirmation']
    assert p['recovered_metadata_only'] and p['original_producer_exit_code'] == 1
    assert p['recovery_source_sha256'] == sha256(Path('scripts/finalize_timestamp_offset.py'))
    assert planning['frozen_policy_sha256'] == sha256(Path('configs/fresh_confirmation_20261108.json'))
    verify_hashes(p['result_hashes'], STAGE_DIRS[0])
    verify_hashes(p['source_hashes'])
    prior = Path('results/dense_reference_20261108/EVIDENCE_MANIFEST.json')
    assert sha256(prior) == p['input_manifest_sha256']
    verify_hashes(json.loads(prior.read_text())['files'])
    unit = Path('../../handoff/restoration_20261107/research_nov09_unit.log')
    s = unit.read_text()
    count = int(re.search(r'Ran (\d+) tests',s).group(1))
    assert count >= 319 and s.rstrip().endswith('OK')
    OUT.mkdir()
    shutil.copyfile(unit, OUT/'unit_tests.log')
    files = sorted(set(SOURCE+[OUT/'unit_tests.log']+
                       [f for d in STAGE_DIRS for f in d.rglob('*') if f.is_file()]))
    for f in files:
        assert not f.is_absolute() and '..' not in f.parts and not f.is_symlink()
        assert f.parts[0] in ['configs','docs','scripts','tests','results']
        assert not any(part.startswith('private_confirmation') or part=='performance_exposures' for part in f.parts)
    manifest = OUT/'EVIDENCE_MANIFEST.json'
    hashes = {str(f): sha256(f) for f in files}
    write(manifest,dict(files=hashes,prior_evidence_manifest_sha256=sha256(prior),unit_tests=count,
        scope='New analyses of already opened records plus synthetic acquisition planning. No incoming private trials, correspondence or whole original datasets. Frozen predictors and original public data reside in the preceding stage.',
        independent_confirmation=False,hardware_success_claimed=False,final_goal_achieved=False))
    files.append(manifest)
    archive = OUT/'timing_tradeoff_evidence.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in files:z.write(f,str(f))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist()) == set(map(str,files))
        for f in files:assert hashlib.sha256(z.read(str(f))).hexdigest() == sha256(f), str(f)
    verify_hashes(hashes)
    write(OUT/'package_verification.json',dict(passed=True,archive=str(archive),
        archive_bytes=archive.stat().st_size,archive_sha256=sha256(archive),
        manifest_sha256=sha256(manifest),entries=len(files),unit_tests=count,
        all_crc_entry_and_live_hashes_verified=True,private_inputs_excluded=True,
        independent_confirmation=False,final_goal_achieved=False))
    print('TIMING_EVIDENCE',len(files),archive.stat().st_size,sha256(archive))


if __name__=='__main__':
    main()
