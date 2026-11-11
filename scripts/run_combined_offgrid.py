"""Frozen seven-method off-grid hypothesis test, planned after aligned results."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import torch
from cranepilot.benchmark import environment_record
from cranepilot.data import sha256
from cranepilot.rollout import HORIZON_INDEX, HORIZONS_S
from scripts.combined_candidate import CombinedCandidate
from scripts.combined_stress_simulator import simulate_stress
from scripts.combined_history_baseline import adaptive_free_forecast, FREQUENCY_RATIOS, DAMPING_PER_S


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_parent(parent):
    provenance = json.loads((parent / 'provenance.json').read_text())
    if provenance['status'] != 'complete':
        raise ValueError('Completed parent run required')
    for mapping, prefix in [('guarded_hashes', Path('.')), ('result_hashes', parent)]:
        for relative, expected in provenance[mapping].items():
            if sha256(prefix / relative.replace('\\', '/')) != expected:
                raise ValueError(f'Parent {mapping} changed: {relative}')
    return provenance


def evaluate(cfg, parent_cfg, out, provenance):
    torch.set_num_threads(4)
    rows, checks, gate_rows = [], [], []
    for fold in cfg['folds']:
        predictor = CombinedCandidate(parent_cfg, fold)
        calibration = predictor.calibration
        folder = out / fold
        folder.mkdir()
        meta, theta, target = simulate_stress(cfg['simulation'], calibration)
        finer_meta, finer_theta, finer_target = simulate_stress(cfg['simulation'], calibration, dt=.0005)
        if not meta.equals(finer_meta):
            raise AssertionError('Half-step metadata mismatch')
        delta = float(np.max(np.abs(np.rad2deg(target - finer_target))))
        if delta > cfg['integrator_tolerance_deg']:
            raise AssertionError(('integrator_half_step', fold, delta))
        checks.append({'fold': fold, 'check': 'integrator_half_step', 'max_error_deg': delta})
        packet = {'theta': theta, 'x': np.zeros_like(theta), 'reference': np.zeros_like(theta),
                  'length_m': meta.length_m.to_numpy(), 'future_reference': np.zeros((len(theta), 30))}
        predictions, seeds, gate = predictor.from_packet(packet, autonomous=True)
        adaptive, choices = adaptive_free_forecast(theta, calibration.omega(meta.length_m.to_numpy()))
        predictions['adaptive_physics'] = adaptive
        if len(meta) != 512 or len(predictions) != 7:
            raise AssertionError('Expected 512 observation cases and seven unchanged methods')
        for name, forecast in predictions.items():
            if forecast.shape != target.shape or not np.isfinite(forecast).all():
                raise AssertionError(('invalid_forecast', fold, name))
            error = np.abs(np.rad2deg(forecast[:, HORIZON_INDEX] - target[:, HORIZON_INDEX]))
            for j, horizon in enumerate(HORIZONS_S):
                local = meta.copy()
                local['case_id'] = np.arange(len(meta))
                local['fold'] = fold
                local['model'] = name
                local['horizon_s'] = horizon
                local['absolute_error_deg'] = error[:, j]
                rows.append(local)
        meta.to_csv(folder / 'cases.csv', index=False)
        pd.DataFrame(choices).to_csv(folder / 'adaptive_choices.csv', index=False)
        np.savez_compressed(folder / 'predictions.npz', theta=theta, target=target, gate=gate,
                            **predictions, **{name + '__seeds': value for name, value in seeds.items()})
        for (condition, measurement), positions in meta.groupby(['condition', 'measurement']).groups.items():
            values = gate[np.asarray(positions)]
            gate_rows.append({'fold': fold, 'condition': condition, 'measurement': measurement,
                              'mean_network_weight': float(values.mean()),
                              'p10': float(np.quantile(values, .1)), 'p90': float(np.quantile(values, .9))})
        print('OFFGRID', fold, len(meta), 'cases; max integration delta', delta, flush=True)
    metrics = pd.concat(rows, ignore_index=True)
    metrics.to_csv(out / 'metrics.csv', index=False)
    groups = ['fold', 'condition', 'measurement', 'model']
    case_errors = metrics.groupby(groups + ['case_id', 'trajectory_id'], as_index=False).absolute_error_deg.mean()
    case_errors.rename(columns={'absolute_error_deg': 'case_mae_deg'}, inplace=True)
    case_errors.to_csv(out / 'case_errors.csv', index=False)
    summaries = case_errors.groupby(groups).case_mae_deg.agg(
        mean_mae_deg='mean', median_case_mae_deg='median',
        p95_case_mae_deg=lambda values: values.quantile(.95), cases='count').reset_index()
    summaries.to_csv(out / 'group_summary.csv', index=False)
    pd.DataFrame(gate_rows).to_csv(out / 'gate_statistics.csv', index=False)
    write(out / 'verification.json', {'passed': True, 'tolerance_deg': cfg['integrator_tolerance_deg'], 'checks': checks})
    provenance.update(status='complete', finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                      cases_per_fold=512, clean_trajectories_per_fold=128,
                      result_hashes={str(path.relative_to(out)): sha256(path)
                                     for path in sorted(out.rglob('*')) if path.is_file() and path.name != 'provenance.json'})
    write(out / 'provenance.json', provenance)
    print('COMPLETE', out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/combined_offgrid_20261107.json'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/runs/combined_offgrid_20261107'))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    parent = Path(cfg['parent_run'])
    prior = verify_parent(parent)
    if cfg['folds'] != prior['config']['folds']:
        raise ValueError('Use all original folds without change')
    for condition in cfg['simulation']['conditions']:
        if (np.isclose(condition['omega_ratio'], FREQUENCY_RATIOS).any()
                or np.isclose(condition['damping_per_s'], DAMPING_PER_S).any()):
            raise ValueError('True ratios and damping must remain off the frozen grid')
    locked = [args.config, Path('docs/COMBINED_OFFGRID_DESIGN_20261107.md'),
              Path('scripts/run_combined_offgrid.py'), *sorted(Path('scripts').glob('combined_*.py')),
              *sorted(Path('tests').glob('test_combined_*.py'))]
    for path in locked:
        if subprocess.check_output(['git', 'show', f'HEAD:{path.as_posix()}']) != path.read_bytes():
            raise ValueError(f'Commit before evaluation: {path}')
    if args.output.exists():
        raise FileExistsError('Preserve previous outputs')
    hashes = dict(prior['guarded_hashes'])
    for path in [*locked, parent / 'provenance.json']:
        hashes[str(path)] = sha256(path)
    for name in ['source_run', 'broad_run']:
        folder = Path(prior['config'][name])
        upstream = json.loads((folder / 'provenance.json').read_text())
        for relative, expected in upstream['checkpoint_hashes'].items():
            path = folder / relative.replace('\\', '/')
            if sha256(path) != expected:
                raise ValueError(f'Upstream checkpoint changed: {path}')
            hashes[str(path)] = expected
    args.output.mkdir(parents=True)
    provenance = {'status': 'running', 'config': cfg, 'parent_config': prior['config'],
                  'parent_provenance_sha256': sha256(parent / 'provenance.json'),
                  'guarded_hashes': hashes, 'environment': environment_record(),
                  'started_utc': pd.Timestamp.now(tz='UTC').isoformat(),
                  'model_training_performed': False, 'campaign_selection_performed': False,
                  'adaptive_per_packet_estimation': True, 'posthoc_hypothesis_test': True,
                  'originally_reserved_confirmation': False, 'hardware_evidence': False}
    write(args.output / 'provenance.json', provenance)
    try:
        evaluate(cfg, prior['config'], args.output, provenance)
    except Exception as error:
        provenance.update(status='failed', failure=f'{type(error).__name__}: {error}',
                          failed_utc=pd.Timestamp.now(tz='UTC').isoformat())
        write(args.output / 'provenance.json', provenance)
        raise


if __name__ == '__main__':
    main()
