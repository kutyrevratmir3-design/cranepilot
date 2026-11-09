"""Package public dense-reference evidence; exclude all private confirmation data."""
import hashlib
import json
import zipfile
from pathlib import Path

from cranepilot.data import sha256
from scripts.dense_reference import LOCKED as DENSE_LOCKED, RUN, OUT
from scripts.fresh_confirmation import LOCKED as FRESH_LOCKED
from scripts.run_external_benchmark import write, verify_hashes


def main():
    manifest = OUT / 'EVIDENCE_MANIFEST.json'
    archive = OUT / 'dense_fresh_evidence.zip'
    verification = OUT / 'package_verification.json'
    if any(f.exists() for f in [manifest, archive, verification]):
        raise FileExistsError('Retain the preceding evidence package')
    v = json.loads((OUT / 'verification.json').read_text())
    readiness = json.loads(Path('results/fresh_confirmation_20261108/readiness.json').read_text())
    assert v['passed'] and readiness['passed'] and v['primary_unchanged']
    p = json.loads((RUN / 'development.json').read_text())
    q = json.loads((RUN / 'retrospective/provenance.json').read_text())
    verify_hashes(p['guarded_hashes'])
    verify_hashes(p['result_hashes'], RUN)
    verify_hashes(q['result_hashes'], RUN / 'retrospective')
    for base, mirror, relatives in [
        (RUN, OUT / 'frozen_development', ['development.json'] + list(p['result_hashes'])),
        (RUN / 'retrospective', OUT / 'frozen_replay', ['provenance.json'] + list(q['result_hashes'])),
    ]:
        for relative in relatives:
            assert (base / relative).read_bytes() == (mirror / relative).read_bytes(), relative
    extra = [
        'scripts/report_dense_reference.py', 'scripts/audit_dense_fresh_stage.py',
        'scripts/package_dense_fresh_evidence.py',
        'docs/DENSE_REFERENCE_2026_11_08_RU.md', 'docs/FRESH_DATA_CANDIDATES_20261108.md',
        'docs/DATED_BUNDLE_20261108.md', 'docs/PROSPECTIVE_AUDIT_20261108.md',
    ]
    bases = [OUT, Path('results/fresh_confirmation_20261108'),
             Path('artifacts/dated_bundle_20261108'), Path('artifacts/dense_reference_bundle_20261108')]
    files = sorted(set(DENSE_LOCKED + [Path(f) for f in FRESH_LOCKED + extra] +
                       [f for base in bases for f in base.rglob('*') if f.is_file()]), key=str)
    for f in files:
        # Only explicit public trees/source files. Incoming data can never enter by recursive glob.
        assert not f.is_absolute() and '..' not in f.parts and not f.is_symlink()
        assert not str(f).startswith('data/') or str(f) == 'data/.gitignore'
        assert not any(part.startswith('private_confirmation') or part == 'performance_exposures' for part in f.parts)
    hashes = {str(f): sha256(f) for f in files}
    write(manifest, dict(
        files=hashes, file_bytes={str(f): f.stat().st_size for f in files},
        scope='Public stage code/config/tests/docs, both frozen bundles, old-data forecasts/metrics and checks. No incoming raw data, private registry, source review, trial identifiers, candidate targets or correspondence. Full replay needs original public data and previously retained repository source; bundles need main source/dependencies.',
        mutable_global_status_and_external_handoff_excluded=True,
        independent_confirmation=False, final_goal_achieved=False))
    files.append(manifest)
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            z.write(f, str(f))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == set(map(str, files))
        for f in files:
            assert hashlib.sha256(z.read(str(f))).hexdigest() == sha256(f), str(f)
    verify_hashes(hashes)
    write(verification, dict(
        passed=True, archive=str(archive), archive_bytes=archive.stat().st_size,
        archive_sha256=sha256(archive), entries=len(files), manifest_sha256=sha256(manifest),
        all_entry_crc_and_sha256_verified=True, all_manifest_live_files_verified=True,
        private_confirmation_tree_excluded=True, original_datasets_excluded=True,
        development_and_replay_mirrors_byte_identical=True,
        mutable_global_status_and_external_handoff_excluded=True,
        package_report_excluded_from_its_own_archive=True,
        independent_confirmation=False, final_goal_achieved=False))
    print('PACKAGE', archive.stat().st_size, 'bytes', len(files), 'files', sha256(archive))


if __name__ == '__main__':
    main()
