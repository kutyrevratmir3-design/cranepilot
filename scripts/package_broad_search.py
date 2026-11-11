"""Package completed local broad research evidence without changing original runs."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'results/broad_search_20261107'
    seal = json.loads((out / 'broad_selection_seal.json').read_text())
    checks = {}
    for name in ('broad_search_20261107', 'reliability_blend_20261107'):
        run = root / 'artifacts/runs' / name
        provenance = json.loads((run / 'provenance.json').read_text())
        assert provenance['status'] == 'complete'
        actual = sha(run / 'selection.json')
        assert actual == seal[name]['selection_sha256'], name
        checks[name] = {'selection_sha256': actual, 'matches_pre_evaluation_seal': True}
        for relative, expected in provenance['result_hashes'].items():
            assert sha(run / relative) == expected, relative
    (out / 'selection_seal_verification.json').write_text(json.dumps(checks, indent=2) + '\n')
    sources = [
        'src/cranepilot/broad_strategies.py',
        'scripts/run_broad_search.py', 'scripts/reliability_blend.py',
        'scripts/run_reliability_blend.py', 'scripts/report_broad_search.py',
        'scripts/predict_research_candidate.py', 'scripts/verify_research_candidates.py',
        'scripts/benchmark_research_candidates.py', 'scripts/package_broad_search.py',
        'tests/test_broad_strategies.py', 'tests/test_reliability_blend.py',
        'configs/broad_search_20261107.json', 'configs/reliability_blend_20261107.json',
        'docs/BROAD_SEARCH_DESIGN_20261107.md', 'docs/RELIABILITY_BLEND_DESIGN_20261107.md',
        'docs/BROAD_RESEARCH_2026_11_07_RU.md', 'docs/NICHE_BENCHMARK_MAP_20261107.md',
    ]
    paths = [root / name for name in sources]
    for folder in ('artifacts/runs/broad_search_20261107', 'artifacts/runs/reliability_blend_20261107', 'results/broad_search_20261107'):
        paths.extend(p for p in (root / folder).rglob('*') if p.is_file()
                     and p.name not in {'evidence.zip', 'archive.json', 'archive_manifest.json'})
    paths = sorted(set(paths))
    manifest = {'scope': 'New stage evidence; requires restored main project, original data and upstream weights for training replay.',
                'files': {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in paths}}
    manifest_path = out / 'archive_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    paths.append(manifest_path)
    destination = out / 'evidence.zip'
    with ZipFile(destination, 'x', ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            archive.write(path, str(path.relative_to(root)))
    with ZipFile(destination) as archive:
        assert archive.testzip() is None
    metadata = {'file': destination.name, 'files': len(paths), 'bytes': destination.stat().st_size,
                'sha256': sha(destination), 'crc_verified': True}
    (out / 'archive.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
