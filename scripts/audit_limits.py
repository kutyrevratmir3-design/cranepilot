"""Bounded frozen-weight audit on development records and exposed synthetic cases."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from cranepilot.benchmark import environment_record
from cranepilot.data import load_dataset, sha256
from cranepilot.forced_observer import analytic_rollout, refine_windows
from cranepilot.innovation import InnovationForecaster
from cranepilot.physics import LengthCalibration
from cranepilot.rollout import command_windows, tensors, HORIZON_INDEX, HORIZONS_S
from scripts.limit_ablation import predict_disabled
from scripts.predict_cranepilot import build_packet
from scripts.run_cleanroom import simulate_cases
from scripts.run_temporal import load_actor, prepare


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/limit_audit_20261107.json'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/runs/limit_audit_20261107'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous results; choose a fresh output')
    cfg = json.loads(args.config.read_text())
    angular_dtype = {'float32': torch.float32, 'float64': torch.float64}[cfg.get('angular_dtype', 'float32')]
    run = Path(cfg['source_run'])
    prior = json.loads((run / 'provenance.json').read_text())
    if prior['status'] != 'complete':
        raise ValueError('Completed source required')
    for group, base in [('guarded_hashes', Path('.')), ('checkpoint_hashes', run)]:
        for name, expected in prior[group].items():
            if sha256(base / name.replace('\\', '/')) != expected:
                raise ValueError(f'Changed source artifact: {name}')
    split = prior['config']['folds'][cfg['fold']]
    if cfg['train_records'] != split['train'] or cfg['validation_records'] != split['validation']:
        raise ValueError('Original development split required')
    info = prior['inputs'][cfg['fold']]
    info = dict(info, actor=info['actor'].replace('\\', '/'),
                weights=[p.replace('\\', '/') for p in info['weights']])
    if cfg['seeds'] != prior['config']['seeds']:
        raise ValueError('All retained seeds required')
    cal = LengthCalibration(**info['calibration'])
    actor = load_actor(info)
    models = []
    for path in info['weights']:
        model = InnovationForecaster(True).eval()
        model.load_state_dict(torch.load(path, weights_only=True))
        models.append(model.to(dtype=angular_dtype))
    torch.set_num_threads(4)
    sources = [args.config, Path(__file__).relative_to(Path.cwd()),
               Path('scripts/limit_ablation.py'), Path('scripts/run_cleanroom.py'),
               Path('scripts/predict_cranepilot.py'), Path('scripts/run_temporal.py'),
               Path('docs/LIMIT_AUDIT_DESIGN_20261107.md'),
               Path('tests/test_limit_ablation.py'), Path(cfg['synthetic_development']),
               run / 'provenance.json', run / 'primary/predictions_s4.npz',
               run / 'cleanroom/predictions.npz', Path(info['actor']),
               Path('data/processed/dataset.npz'), Path('data/processed/manifest.json'),
               *map(Path, info['weights']), *sorted(Path('src/cranepilot').glob('*.py'))]
    hashes = {str(p): sha256(p) for p in sources}
    provenance = {'status': 'running', 'config': cfg, 'input_sha256': hashes,
                  'environment': environment_record(), 'training_performed': False,
                  'test_records_scored': [], 'confirmation_evaluated': False,
                  'started_utc': pd.Timestamp.now(tz='UTC').isoformat()}
    args.output.mkdir(parents=True)
    write_json(args.output / 'provenance.json', provenance)
    start = time.perf_counter()
    try:
        metrics, cases, components, replay_checks = [], [], [], []
        physical_checks = []

        def evaluate(label, batch, target, subsets, metadata, expected=None, extra=None):
            # No future measured value or target key is delivered to a predictor.
            batch = {k: v.to(dtype=angular_dtype) for k, v in batch.items() if not k.startswith('target_')}
            arrays = {'target': target, **(extra or {})}
            for name, disabled in cfg['variants'].items():
                seed_predictions = []
                for seed, model in zip(cfg['seeds'], models):
                    blocks = []
                    for lo in range(0, len(target), cfg['batch_size']):
                        small = {k: v[lo:lo + cfg['batch_size']] for k, v in batch.items()}
                        blocks.append(predict_disabled(model, small, small['acceleration'], disabled).numpy())
                    seed_predictions.append(np.concatenate(blocks))
                seed_predictions = np.stack(seed_predictions)
                y = seed_predictions.mean(0)
                arrays[name] = seed_predictions
                if name == 'incumbent' and expected is not None:
                    difference = float(np.max(np.abs(np.rad2deg(y - expected))))
                    replay_checks.append({'group': label, 'max_difference_deg': difference})
                    if difference > cfg['replay_tolerance_deg']:
                        raise AssertionError(f'Replay {label}: {difference} deg')
                if name == 'physical_limit':
                    b = batch
                    physical = analytic_rollout(b['initial_q'].numpy(), b['initial_p'].numpy(),
                                                b['omega'].numpy(), b['acceleration'].numpy())
                    difference = float(np.max(np.abs(np.rad2deg(y - physical))))
                    physical_checks.append({'group': label, 'max_difference_deg': difference})
                    if difference > cfg['physical_limit_tolerance_deg']:
                        raise AssertionError(f'Physical limit {label}: {difference} deg')
                for seed_name, prediction in [('ensemble', y), *zip(map(str, cfg['seeds']), seed_predictions)]:
                    errors = np.rad2deg(prediction[:, HORIZON_INDEX] - target[:, HORIZON_INDEX])
                    for subset, mask in subsets.items():
                        for j, horizon in enumerate(HORIZONS_S):
                            err = errors[mask, j]
                            if len(err):
                                metrics.append({'group': label, 'variant': name, 'seed': seed_name,
                                                'subset': subset, 'horizon_s': horizon, 'n': len(err),
                                                'mae_deg': float(np.mean(np.abs(err))),
                                                'rmse_deg': float(np.sqrt(np.mean(err**2)))})
                if metadata is not None:
                    frame = metadata.copy()
                    frame['variant'] = name
                    frame['mae_deg'] = np.abs(np.rad2deg(y[:, HORIZON_INDEX] - target[:, HORIZON_INDEX])).mean(1)
                    cases.append(frame)
                print(f'{label}: {name}', flush=True)
            np.savez_compressed(args.output / f'{label}_predictions.npz', **arrays)
            with torch.no_grad():
                for seed, model in zip(cfg['seeds'], models):
                    context = model.encoder(batch['angular_context'])
                    adjustment = torch.tanh(model.observer(context))
                    correction = torch.tanh(model.initial(context))
                    damping, coupling, rates = model.cell.parameters_from_context(context)
                    components.append({'group': label, 'seed': seed,
                        'mean_abs_angle_shift_deg': float(torch.rad2deg(.02 * adjustment[:, 0]).abs().mean()),
                        'mean_abs_normalized_momentum_shift_deg': float(torch.rad2deg(.2 * correction[:, 0]).abs().mean()),
                        'mean_abs_frequency_shift_percent': float((15 * adjustment[:, 1]).abs().mean()),
                        'mean_dimensionless_damping': float(damping.mean()),
                        'mean_coupling_norm': float(torch.linalg.vector_norm(coupling, dim=1).mean())})

        # Use only the already exposed development simulation, never the reserved grid.
        synthetic_cfg = json.loads(Path(cfg['synthetic_development']).read_text())
        meta, theta, target = simulate_cases(synthetic_cfg, cal)
        _, _, refined_target = simulate_cases(synthetic_cfg, cal, dt=.0005)
        rk4_error = float(np.max(np.abs(np.rad2deg(refined_target - target))))
        if rk4_error > 1e-7:
            raise AssertionError('Synthetic integrator resolution')
        with np.load(run / 'cleanroom/predictions.npz') as original:
            # Cross-platform trig may differ at roundoff; this is 1e-12 rad,
            # far below the independent integrator and forecast tolerances.
            np.testing.assert_allclose(theta, original['theta'], rtol=0, atol=1e-12)
            np.testing.assert_allclose(target, original['target'], rtol=0, atol=1e-12)
            provenance['synthetic_target_reproduction_max_deg'] = float(
                np.max(np.abs(np.rad2deg(target - original['target']))))
            expected = original['forced_driven_huber'].copy()
        raw = build_packet(theta, np.zeros_like(theta), np.zeros_like(theta),
                           meta.length_m.to_numpy(), np.zeros((len(theta), 30)), cal)
        batch = tensors(refine_windows(raw))
        batch['acceleration'] = torch.zeros(len(theta), 30)
        subsets = {'all': np.ones(len(meta), dtype=bool)}
        for measurement in synthetic_cfg['measurement']:
            subsets[measurement] = meta.measurement.to_numpy() == measurement
            for damping in synthetic_cfg['damping_per_s']:
                subsets[f'{measurement}_damping_{damping:g}'] = subsets[measurement] & (meta.damping_per_s.to_numpy() == damping)
        evaluate('synthetic_development', batch, target, subsets, meta, expected, {'theta': theta})
        meta.to_csv(args.output / 'synthetic_cases.csv', index=False)

        data = load_dataset(Path('data/processed'))
        for role, ids in [('train', cfg['train_records']), ('validation', cfg['validation_records'])]:
            for sid in ids:
                scenario = data[sid]
                k = np.arange(40, len(scenario.time) - 150, cfg['stride'])
                raw = command_windows(scenario, cal, k)
                active = np.abs(raw['initial_velocity']) >= cfg['active_velocity_threshold_m_s']
                expected = None
                if role == 'validation':
                    with np.load(run / 'primary' / f'predictions_s{sid}.npz') as retained:
                        idx = np.searchsorted(retained['k'], k)
                        np.testing.assert_array_equal(retained['k'][idx], k)
                        expected = retained['forced_driven_huber'][idx].copy()
                evaluate(f'{role}_s{sid}', prepare(raw, actor), raw['target_q'],
                         {'all': np.ones(len(k), dtype=bool), 'active': active, 'quiet': ~active},
                         None, expected, {'k': k, 'active_at_origin': active})
        pd.DataFrame(metrics).to_csv(args.output / 'metrics.csv', index=False)
        pd.concat(cases, ignore_index=True).to_csv(args.output / 'synthetic_case_metrics.csv', index=False)
        pd.DataFrame(components).to_csv(args.output / 'component_statistics.csv', index=False)
        for path, expected in hashes.items():
            if sha256(Path(path)) != expected:
                raise AssertionError(f'Input changed during audit: {path}')
        provenance.update(status='complete', elapsed_s=time.perf_counter() - start,
                          rk4_half_step_max_error_deg=rk4_error, replay_checks=replay_checks,
                          physical_limit_checks=physical_checks,
                          output_sha256={p.name: sha256(p) for p in args.output.iterdir() if p.name != 'provenance.json'})
    except Exception as error:
        provenance.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(args.output / 'provenance.json', provenance)


if __name__ == '__main__':
    main()
