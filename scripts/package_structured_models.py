"""Package completed structured-model evidence locally with SHA and CRC checks."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from cranepilot.data import sha256
from scripts.report_structured_models import verify_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('artifacts/runs/structured_models_20261107'))
    parser.add_argument('--output', type=Path, default=Path('results/structured_models_20261107'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run, out = root/args.run, root/args.output
    _, counts = verify_evidence(root, run)
    report = json.loads((out/'report.json').read_text())
    guards = [(run/'provenance.json', 'source_provenance_sha256'),
              (root/report['document'], 'document_sha256'),
              (root/'scripts/report_structured_models.py', 'report_source_sha256'),
              (root/'scripts/report_finetune_campaign.py', 'aggregation_helper_sha256'),
              (out/'replay_verification.json', 'replay_verification_sha256')]
    if 'inference_latency_sha256' in report:
        guards.append((out/'inference_latency.json', 'inference_latency_sha256'))
    for path, key in guards:
        if sha256(path) != report[key]:
            raise ValueError(f'Changed report evidence: {path}')
    if report.get('unit_tests') is not None:
        if sha256(out/'structured_models_tests.log') != report['unit_tests']['sha256']:
            raise ValueError('Retained unit-test log changed')
    excluded = {'evidence.zip', 'archive_manifest.json', 'archive.json'}
    for name in excluded:
        if (out/name).exists():
            raise FileExistsError(f'Preserve existing archive: {out/name}')
    paths = [p for p in run.rglob('*') if p.is_file()]
    paths.extend(p for p in out.rglob('*') if p.is_file() and p.name not in excluded)
    for pattern in ['scripts/*structured*.py', 'scripts/residual_adapter.py', 'scripts/stable_latent.py',
                    'tests/test_structured*.py', 'tests/test_residual_adapter.py', 'tests/test_stable_latent.py',
                    'configs/structured_models*.json', 'docs/STRUCTURED_MODELS*.md',
                    'RESEARCH_STATUS_2026_11_07.md', 'docs/GITHUB_PUBLICATION_SCOPE_20261107.md',
                    'docs/CONTROL_INTEGRATION_AUDIT_20261107.md',
                    'scripts/export_research_bundle.py', 'scripts/predict_research_bundle.py',
                    'tests/test_research_bundle.py', 'docs/RESEARCH_BUNDLE_20261107.md',
                    'artifacts/research_bundle_20261107/**/*']:
        paths.extend(root.glob(pattern))
    paths.append(root/report['document'])
    paths = sorted({path for path in paths if path.is_file()})
    manifest = dict(scope='Structured models stage; full replay requires retained upstream code, data and weights.',
        source_provenance_sha256=sha256(run/'provenance.json'), verified_hash_counts=counts,
        files={str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in paths})
    manifest_path = out/'archive_manifest.json'
    with manifest_path.open('x') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    paths.append(manifest_path)
    destination = out/'evidence.zip'
    with ZipFile(destination, 'x', ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            archive.write(path, str(path.relative_to(root)))
    with ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC failed')
        if set(archive.namelist()) != {str(p.relative_to(root)) for p in paths}:
            raise ValueError('Archive membership mismatch')
    information = dict(file=destination.name, files=len(paths), bytes=destination.stat().st_size,
        sha256=sha256(destination), manifest_sha256=sha256(manifest_path), crc_verified=True)
    with (out/'archive.json').open('x') as handle:
        json.dump(information, handle, indent=2)
        handle.write('\n')
    print(json.dumps(information, indent=2))


if __name__ == '__main__':
    main()
