"""Check history-only inference against retained per-fold evaluation traces."""
from pathlib import Path
import json
import numpy as np
from cranepilot.data import load_dataset
from cranepilot.rollout import HORIZON_INDEX
from scripts.predict_temporal import predict


def main():
    run = Path('artifacts/runs/temporal_20261106')
    p = json.loads((run/'provenance.json').read_text())
    scenarios = load_dataset(Path('data/processed'))
    origins = np.array([40, 500, 1000])
    rows = []
    for fold, split in p['config']['folds'].items():
        sid = split['validation'][0]
        scenario = scenarios[sid]
        past = origins[:, None]-np.arange(41)
        inputs = {'theta': scenario.signals['theta'][past],
            'x': scenario.signals['x'][past], 'reference': scenario.signals['xr'][past],
            'length_m': scenario.signals['L'][origins],
            'future_reference': scenario.signals['xr'][origins[:, None]+np.arange(30)*5]}
        with np.load(run/fold/f'predictions_s{sid}.npz') as trace:
            index = np.searchsorted(trace['k'], origins)
            np.testing.assert_equal(trace['k'][index], origins)
            for variant in [*p['config']['models'], 'forced_driven_huber', 'selected_candidate']:
                actual, spread, resolved = predict(run, fold, variant, inputs)
                expected = trace[variant][index][:, HORIZON_INDEX]
                error = float(np.max(abs(np.rad2deg(actual-expected))))
                if error > 5e-5 or not np.isfinite(spread).all():
                    raise AssertionError(f'Inference mismatch {fold}/{variant}: {error} deg')
                rows.append({'fold': fold, 'variant': variant, 'resolved_variant': resolved,
                             'max_absolute_difference_deg': error, 'packets': len(origins)})
    report = {'history_only_inputs': list(inputs), 'comparisons': rows,
              'max_absolute_difference_deg': max(r['max_absolute_difference_deg'] for r in rows),
              'tolerance_deg': 5e-5, 'passed': True}
    Path('results/latest/inference_parity_20261106.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
