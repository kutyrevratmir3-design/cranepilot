"""Create a non-overwriting, CRC- and SHA-verified local global-research archive."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from cranepilot.data import sha256
from scripts.report_global_research import verify_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('artifacts/runs/global_research_20261107'))
    parser.add_argument('--output', type=Path, default=Path('results/global_research_20261107'))
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    run = root/args.run; out = root/args.output
    provenance, counts = verify_evidence(root, run)
    report = json.loads((out/'report.json').read_text())
    if sha256(run/'provenance.json') != report['source_provenance_sha256']:
        raise ValueError('Report and run provenance differ')
    if sha256(root/report['document']) != report['document_sha256']:
        raise ValueError('Generated research document changed')
    if sha256(root/'scripts/report_global_research.py') != report['report_source_sha256']:
        raise ValueError('Report generator changed since reporting')
    excluded = {'archive.json', 'archive_manifest.json', 'evidence.zip'}
    for name in excluded:
        if (out/name).exists():
            raise FileExistsError(f'Preserve existing archive evidence: {out/name}')
    paths = [p for p in run.rglob('*') if p.is_file()]
    paths.extend(p for p in out.rglob('*') if p.is_file() and p.name not in excluded)
    for pattern in ['scripts/global_*.py', 'scripts/plot_global*.py', 'scripts/run_global_research.py', 'scripts/report_global_research.py',
                    'scripts/package_global_research.py', 'scripts/predict_global*.py',
                    'scripts/verify_global_research.py', 'tests/test_global_*.py', 'configs/global_research*.json',
                    'docs/GLOBAL_RESEARCH*.md', 'docs/LITERATURE_CRANE_20261107.md', 'docs/LITERATURE_ADJACENT_20261107.md']:
        paths.extend(root.glob(pattern))
    paths.append(root/report['document'])
    paths = sorted(set(paths))
    manifest = {'scope': 'This local global-model stage; complete replay also requires restored upstream code, data and weights.',
                'source_provenance_sha256': sha256(run/'provenance.json'), 'verified_hash_counts': counts,
                'files': {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in paths}}
    manifest_path = out/'archive_manifest.json'
    with manifest_path.open('x') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False); handle.write('\n')
    paths.append(manifest_path); destination = out/'evidence.zip'
    with ZipFile(destination, 'x', ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            archive.write(path, str(path.relative_to(root)))
    with ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC integrity failed')
        if set(archive.namelist()) != {str(p.relative_to(root)) for p in paths}:
            raise ValueError('Archive membership mismatch')
    info = {'file': destination.name, 'files': len(paths), 'bytes': destination.stat().st_size,
            'sha256': sha256(destination), 'crc_verified': True,
            'manifest_sha256': sha256(manifest_path)}
    with (out/'archive.json').open('x') as handle:
        json.dump(info, handle, indent=2); handle.write('\n')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
