"""Copy selected comparator weights; pin, but never duplicate/change, the primary."""
import json
import shutil
from pathlib import Path

from cranepilot.data import sha256
from scripts.dense_reference import CONFIG, OUT, RUN, require_frozen_development
from scripts.run_external_benchmark import write

DEST = Path('artifacts/dense_reference_bundle_20261108')


def main():
    if DEST.exists():raise FileExistsError('Preserve previous bundle')
    parent = require_frozen_development()
    verification = json.loads((OUT / 'development_verification.json').read_text())
    if not verification['passed'] or verification['development_sha256'] != sha256(RUN / 'development.json'):
        raise ValueError('Verified frozen development required')
    cfg = json.loads(CONFIG.read_text())
    selection = json.loads((RUN / 'selection.json').read_text())
    DEST.mkdir(parents=True)
    cases = {}
    for name, case in selection['cases'].items():
        (DEST / name).mkdir()
        selected = {choice['key'] for choice in case['reference'].values()}
        specs = {}
        for key in sorted(selected):
            spec = case['specs'][key].copy()
            if 'file' in spec:
                source = Path(spec.pop('file'))
                if sha256(source) != spec['source_sha256']:raise ValueError('Selected source model changed')
                spec['original_model_file'] = str(source)
                rel = name + '/model_' + spec['source_sha256'] + '.npz'
                if not (DEST / rel).exists():shutil.copyfile(source, DEST / rel)
                spec['weight_file'] = rel
            specs[key] = spec
        cases[name] = dict(reference={age: value['key'] for age, value in case['reference'].items()}, specs=specs)
    shutil.copyfile(RUN / 'selection.json', DEST / 'selection.json')
    sources = ['scripts/dense_reference.py', 'src/cranepilot/matched_dated_pilot.py',
        'scripts/dated_forecast.py', 'scripts/latency_observer.py', 'scripts/drive_conditioned.py',
        'scripts/modal_reference_challenge.py', 'scripts/modal_history.py',
        'scripts/modal_backtest_gate.py', 'scripts/regime_fusion.py',
        'src/cranepilot/physics.py', 'src/cranepilot/data.py']
    write(DEST / 'manifest.json', dict(cases=cases, train_ages_raw=cfg['train_ages_raw'],
        primary_selected=cfg['candidate_selected'], primary_manifest_sha256=parent['primary_manifest_sha256'],
        files={str(f.relative_to(DEST)): sha256(f) for f in DEST.rglob('*') if f.is_file()},
        source_hashes={p: sha256(Path(p)) for p in sources}, scope=cfg['scope'],
        development_sha256=sha256(RUN / 'development.json'), selection_sha256=sha256(RUN / 'selection.json'),
        development_verification_sha256=sha256(OUT / 'development_verification.json'),
        independent_confirmation=False, final_goal_achieved=False))
    print('DENSE_REFERENCE_BUNDLE', sum(f.is_file() for f in DEST.rglob('*')),
          sum(f.stat().st_size for f in DEST.rglob('*') if f.is_file()))


if __name__ == '__main__':main()
