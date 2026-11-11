"""Preserve the local November 6 campaign and its exact reused checkpoints."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
from cranepilot.data import sha256


def main():
    run = Path('artifacts/runs/temporal_20261106')
    output = Path('results/latest/research_20261106_evidence.zip')
    if output.exists():
        raise FileExistsError('Do not overwrite evidence archives')
    provenance = json.loads((run/'provenance.json').read_text())
    if provenance['status'] != 'complete' or not (run/'cleanroom/provenance.json').exists():
        raise ValueError('Complete all fixed experiments first')
    for path, expected in provenance['guarded_hashes'].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f'Guarded source/input changed: {path}')
    for path, expected in provenance['checkpoint_hashes'].items():
        if sha256(run/path) != expected:
            raise ValueError(f'Checkpoint changed: {path}')
    files = [p for p in run.rglob('*') if p.is_file()]
    for info in provenance['inputs'].values():
        files.extend(Path(s) for s in [info['actor'], info['source_provenance'], *info['weights']])
    for directory in ['src/cranepilot', 'scripts', 'configs', 'tests', 'docs', 'examples']:
        files.extend(p for p in Path(directory).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.zip'))
    files.extend(Path(x) for x in ['README.md', 'LICENSE', 'CITATION.cff',
                                  'pyproject.toml', 'requirements-lock.txt'])
    files.extend(p for p in Path('results/latest').glob('*20261106*')
                 if p.is_file() and p.suffix in ('.csv', '.json', '.txt'))
    files.extend(Path('paper/figures').glob('*20261106*'))
    files = sorted(set(files))
    manifest = {p.as_posix(): sha256(p) for p in files}
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p, p.as_posix())
        archive.writestr('archive_manifest.json', json.dumps(manifest, indent=2))
        archive.writestr('ARCHIVE_README.txt',
            'Local research evidence, 6 November 2026. Extract at the project root. '
            'Contains 36 new angular checkpoints, exact reused incumbents/actors, '
            'full prediction traces, fixed configurations, source, tests and reports. '
            'No GitHub or Overleaf publication performed. Raw/processed dataset and '
            'historical GP joblib files are excluded. Obtain the public data from '
            'DOI 10.17632/x69hyyv3zw.2 (CC BY 4.0), and cite DOI 10.1016/j.dib.2025.111822. '
            'The supplied history-only inference example does not need that dataset. '
            'Run python -m scripts.predict_temporal --input examples/history_packet.npz '
            '--output examples/NEW_FORECAST.csv from the project root. '
            'Forecasting comparisons are retrospective. Synthetic free-sway results '
            'are angular-only diagnostics, not hardware validation or a SOTA claim. '
            'Read docs/RESEARCH_2026_11_06.md and the model card before using results.')
    with ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Archive CRC failure')
        for path, expected in manifest.items():
            if hashlib.sha256(archive.read(path)).hexdigest() != expected:
                raise RuntimeError(f'Archive content hash mismatch: {path}')
    result = {'archive': output.as_posix(), 'files': len(files),
              'bytes': output.stat().st_size, 'sha256': sha256(output)}
    Path('results/latest/research_20261106_archive.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
