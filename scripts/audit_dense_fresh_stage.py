"""Audit old evidence and private readiness without opening new physical signals."""
import json
import re
import shutil
import subprocess
from pathlib import Path

from scripts.fresh_confirmation import bundle_commitments, verify_seen_index
from scripts.fresh_record_guard import sha256_file
from scripts.run_external_benchmark import verify_hashes, write
from scripts.verify_finetune_campaign import original_files

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/dense_reference_20261108'
READY = ROOT / 'results/fresh_confirmation_20261108'
HANDOFF = ROOT.parents[1] / 'handoff/restoration_20261107'


def main():
    destination = READY / 'readiness.json'
    if destination.exists():
        raise FileExistsError('Retain the preceding readiness audit')
    v = json.loads((OUT / 'verification.json').read_text())
    dv = json.loads((OUT / 'development_verification.json').read_text())
    assert v['passed'] and dv['passed'] and v['primary_unchanged']
    originals = original_files()
    assert originals['passed'] and originals['checked_files'] == 1281
    baseline = json.loads((ROOT / 'results/drive_conditioned_20261107/original_git_worktree_integrity.json').read_text())['paths']
    for relative in baseline:
        saved = subprocess.check_output(['git', 'show', 'HEAD:' + relative], cwd=ROOT)
        assert (ROOT / relative).read_bytes().replace(b'\r\n', b'\n') == saved.replace(b'\r\n', b'\n'), relative
    old_manifest = json.loads((ROOT / 'results/dated_forecast_20261108/EVIDENCE_MANIFEST.json').read_text())
    verify_hashes(old_manifest['files'])
    run = ROOT / 'artifacts/runs/dense_reference_20261108'
    provenance = json.loads((run / 'retrospective/provenance.json').read_text())
    verify_hashes(provenance['result_hashes'], run / 'retrospective')
    mirror = OUT / 'frozen_replay'
    mirror.mkdir(exist_ok=True)
    for relative in ['provenance.json'] + list(provenance['result_hashes']):
        source, target = run / 'retrospective' / relative, mirror / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert source.read_bytes() == target.read_bytes()
        else:
            shutil.copyfile(source, target)
    unit_path = HANDOFF / 'dense_fresh_hardened_unit_20261108.log'
    text = unit_path.read_text()
    count = int(re.search(r'Ran (\d+) tests', text).group(1))
    assert count >= 316 and text.rstrip().endswith('OK')
    for source, target in [
        (unit_path, READY / 'final_unit_tests_hardened.log'),
        (HANDOFF / 'dense_reference_full_verify_20261108.log', OUT / 'full_verification.log'),
        (HANDOFF / 'dense_reference_report_20261108.log', OUT / 'report.log'),
    ]:
        if target.exists():
            assert source.read_bytes() == target.read_bytes()
        else:
            shutil.copyfile(source, target)
    private_root = ROOT / 'data/private_confirmation_20261108'
    assert subprocess.run(['git', 'check-ignore', '-q', str(private_root.relative_to(ROOT))], cwd=ROOT).returncode == 0
    seen_path = private_root / 'known_seen.json'
    seen = verify_seen_index(seen_path)
    assert sha256_file(ROOT / 'configs/fresh_confirmation_20261108.json') == seen['policy_sha256']
    for relative, digest in seen['source_hashes'].items():
        assert sha256_file(ROOT / relative) == digest
    exposures = list((private_root / 'performance_exposures').glob('*.json'))
    locks = list((private_root / 'runs').glob('*/lock.json'))
    assert not exposures and not locks
    commitments = bundle_commitments()
    value = dict(
        passed=True,
        original_files_checked=originals['checked_files'],
        existing_eol_only_paths_preserved=len(baseline),
        preceding_dated_evidence_live_files_verified=len(old_manifest['files']),
        primary_manifest_sha256=v['primary_bundle_manifest_sha256'],
        reference_manifest_sha256=v['comparator_bundle_manifest_sha256'],
        guarded_bundle_code_train_files=len(commitments),
        unit_tests=count,
        final_unit_tests_sha256=sha256_file(READY / 'final_unit_tests_hardened.log'),
        final_unit_test_report='final_unit_tests_hardened.log',
        old_registry=dict(
            sha256=sha256_file(seen_path),
            response_file_hashes=len(seen['known_file_sha256']),
            auxiliary_file_hashes=len(seen['auxiliary_file_sha256']),
            sequence_segments=len(seen['sequences']),
            informative_blocks=sum(len(s['blocks']) for s in seen['sequences'].values()),
            policy_and_source_hashes_verified=True,
        ),
        private_intake_ignored=True,
        exact_old_registry_seal_verified=True,
        sealed_registry_sha256=sha256_file(ROOT / 'configs/seen_registry_commitment_20261108.json'),
        real_confirmation_locks=0,
        real_performance_exposures=0,
        suitable_new_physical_records_received=0,
        scope='Software readiness and retrospective reproducibility only; no candidate numerical files loaded by this audit.',
        independent_confirmation=False,
        hardware_success_claimed=False,
        final_goal_achieved=False,
    )
    write(destination, value)
    print(json.dumps({k: value[k] for k in ['passed', 'unit_tests', 'original_files_checked', 'real_confirmation_locks', 'independent_confirmation']}))


if __name__ == '__main__':
    main()
