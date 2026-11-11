"""End-to-end replay from history-only packets for both locked selection policies."""
import json
from pathlib import Path
import numpy as np
from cranepilot.data import load_dataset
from cranepilot.rollout import HORIZON_INDEX
from scripts.predict_research_candidate import forecast


def main():
    destination = Path('results/broad_search_20261107/history_only_verification.json')
    if destination.exists(): raise FileExistsError('Preserve earlier verification')
    data = load_dataset(Path('data/processed')); checks = []
    for kind in ['broad_search', 'reliability_blend']:
        run = Path('artifacts/runs') / f'{kind}_20261107'
        prov = json.loads((run / 'provenance.json').read_text())
        old = json.loads((Path(prov['config']['source_run']) / 'provenance.json').read_text())
        for fold in prov['config']['folds']:
            sid = old['config']['folds'][fold]['validation'][0]; s = data[sid]
            with np.load(run / fold / 'validation_predictions.npz') as retained:
                indices = np.array([0, len(retained['k']) // 2, len(retained['k']) - 1])
                k = retained['k'][indices]; past = k[:, None] - np.arange(41)
                packet = {'theta': s.signals['theta'][past], 'x': s.signals['x'][past],
                          'reference': s.signals['xr'][past], 'length_m': s.signals['L'][k],
                          'future_reference': s.signals['xr'][k[:, None] + np.arange(30) * 5]}
                for policy in ['selected', 'global']:
                    actual, spread, resolved = forecast(run, fold, policy, packet)
                    expected = retained[resolved].mean(0)[indices] if kind == 'broad_search' else retained[resolved][indices]
                    error = float(np.abs(np.rad2deg(actual - expected[:, HORIZON_INDEX])).max())
                    if error > 5e-5 or not np.isfinite(spread).all(): raise AssertionError((kind, fold, policy, error))
                    checks.append({'family': kind, 'fold': fold, 'policy': policy, 'resolved': resolved,
                                   'max_error_deg': error, 'packets': 3})
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {'passed': True, 'tolerance_deg': 5e-5, 'max_error_deg': max(x['max_error_deg'] for x in checks), 'checks': checks}
    destination.write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
