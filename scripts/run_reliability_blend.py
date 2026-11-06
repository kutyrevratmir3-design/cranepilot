"""Select fixed discrepancy gates on development data, then evaluate locked gates."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import torch
from cranepilot.benchmark import environment_record
from cranepilot.broad_strategies import make_model
from cranepilot.data import load_dataset, sha256
from cranepilot.forced_observer import refine_windows, analytic_rollout
from cranepilot.physics import LengthCalibration
from cranepilot.rollout import command_windows, HORIZON_INDEX, HORIZONS_S
from cranepilot.metrics import regression_metrics
from scripts.run_temporal import load_actor
from scripts.run_broad_search import inputs, predict, scores
from scripts.run_cleanroom import simulate_cases
from scripts.predict_cranepilot import build_packet
from scripts.reliability_blend import blend


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['select', 'evaluate'], required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/reliability_blend_20261107.json'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/runs/reliability_blend_20261107'))
    args = parser.parse_args(); run = args.output
    cfg = json.loads(args.config.read_text()); source = Path(cfg['source_run'])
    old = json.loads((source / 'provenance.json').read_text())
    if old['status'] != 'complete': raise ValueError('Completed original run required')
    for kind, base in [('guarded_hashes', Path('.')), ('checkpoint_hashes', source)]:
        for path, digest in old[kind].items():
            if sha256(base / path.replace('\\', '/')) != digest: raise ValueError(f'Original input changed: {path}')
    experiments = [args.config, Path('scripts/run_reliability_blend.py'), Path('scripts/reliability_blend.py'),
                   Path('tests/test_reliability_blend.py'), Path('docs/RELIABILITY_BLEND_DESIGN_20261107.md')]
    paths = [*experiments, Path('scripts/run_broad_search.py'), Path('scripts/run_cleanroom.py'),
             Path('scripts/predict_cranepilot.py'), Path('scripts/run_temporal.py'),
             Path(cfg['development_config']), Path(cfg['confirmation_config']), source / 'provenance.json',
             Path('data/processed/dataset.npz'), Path('data/processed/manifest.json'),
             *sorted(Path('src/cranepilot').glob('*.py'))]
    for fold in cfg['folds']:
        paths.extend(Path(p.replace('\\', '/')) for p in [old['inputs'][fold]['actor'], *old['inputs'][fold]['weights']])
    hashes = {str(p): sha256(p) for p in paths}
    if args.stage == 'select':
        if run.exists(): raise FileExistsError('Preserve previous run')
        for path in experiments:
            if subprocess.check_output(['git', 'show', f'HEAD:{path.as_posix()}']) != path.read_bytes():
                raise ValueError(f'Commit before selecting: {path}')
        run.mkdir(parents=True)
        provenance = {'status': 'selecting', 'config': cfg, 'guarded_hashes': hashes,
                      'environment': environment_record(), 'training_performed': False,
                      'started_utc': pd.Timestamp.now(tz='UTC').isoformat()}
        write_json(run / 'provenance.json', provenance)
    else:
        provenance = json.loads((run / 'provenance.json').read_text())
        if provenance['status'] != 'selection_complete' or provenance['guarded_hashes'] != hashes:
            raise ValueError('Unchanged completed selection required')
        if sha256(run / 'selection.json') != provenance['selection_sha256']:
            raise ValueError('Locked selection changed')
    torch.set_num_threads(4); data = load_dataset(Path('data/processed'))
    selections, metric_rows, gate_rows, checks = {}, [], [], []
    for fold in cfg['folds']:
        split = old['config']['folds'][fold]; info = old['inputs'][fold]
        cal = LengthCalibration(**info['calibration'])
        actor = load_actor(dict(info, actor=info['actor'].replace('\\', '/')))
        models = []
        for path in info['weights']:
            model = make_model({}); model.load_state_dict(torch.load(path.replace('\\', '/'), weights_only=True)); models.append(model)
        directory = run / fold
        if args.stage == 'select': directory.mkdir()

        def forecasts(raw, use_actor=True):
            b = inputs(raw, actor if use_actor else None)
            ys = np.stack([predict(m, b) for m in models])
            refined = refine_windows(raw)
            physical = analytic_rollout(refined['initial_q'], refined['initial_p'], raw['omega'], b['acceleration'].numpy())
            rms = np.sqrt(np.mean(refined['forced_residual']**2, axis=1))
            seeds = {'incumbent': ys, 'physical': np.broadcast_to(physical, ys.shape).copy()}
            weights = {'incumbent': np.ones(len(rms)), 'physical': np.zeros(len(rms))}
            for tau in cfg['tau_bins']:
                name = f'gate_{tau:g}'
                seeds[name] = np.stack([blend(y, physical, rms, tau)[0] for y in ys])
                weights[name] = blend(ys[0], physical, rms, tau)[1]
            return {n: arr.mean(0) for n, arr in seeds.items()}, seeds, weights

        def synthetic(c):
            meta, theta, target = simulate_cases(c, cal)
            raw = build_packet(theta, np.zeros_like(theta), np.zeros_like(theta), meta.length_m.to_numpy(), np.zeros((len(theta), 30)), cal)
            predictions, seeds, gates = forecasts(raw, False)
            return meta, theta, target, predictions, seeds, gates

        def packet(sid):
            return command_windows(data[sid], cal, np.arange(40, len(data[sid].time) - 150, cfg['evaluation_stride']))

        if args.stage == 'select':
            sid = split['validation'][0]; raw = packet(sid)
            predictions, seed_predictions, gates = forecasts(raw)
            active = abs(raw['initial_velocity']) >= cfg['active_threshold']
            validation = {n: scores(y, raw['target_q'], active) for n, y in predictions.items()}
            with np.load(source / fold / f'predictions_s{sid}.npz') as retained:
                np.testing.assert_array_equal(raw['k'], retained['k'])
                error = float(np.abs(np.rad2deg(predictions['incumbent'] - retained['forced_driven_huber'])).max())
                if error > 5e-5: raise AssertionError(f'Replay failed: {error}')
            checks.append({'fold': fold, 'incumbent_replay_max_deg': error})
            c = json.loads(Path(cfg['development_config']).read_text())
            meta, theta, target, dev, dev_seeds, dev_gates = synthetic(c)
            dev_scores = {n: float(np.abs(np.rad2deg(y[:, HORIZON_INDEX] - target[:, HORIZON_INDEX])).mean()) for n, y in dev.items()}
            pool = [n for n in validation if all(validation[n][g] <= cfg['validation_max_ratio'] * validation['incumbent'][g] + 1e-12 for g in ('all', 'active', 'quiet'))]
            selected = min(pool, key=dev_scores.get)
            selections[fold] = {'model': selected, 'eligible': pool, 'validation': validation, 'development_synthetic_mae': dev_scores}
            np.savez_compressed(directory / 'validation_predictions.npz', k=raw['k'], target=raw['target_q'], **predictions,
                                **{n + '__seeds': y for n, y in seed_predictions.items()})
            np.savez_compressed(directory / 'development_predictions.npz', target=target, theta=theta, **dev)
            meta.to_csv(directory / 'development_cases.csv', index=False)
            for role, values in [('validation', gates), ('synthetic_development', dev_gates)]:
                for name, g in values.items():
                    gate_rows.append({'fold': fold, 'group': role, 'model': name, 'mean_network_weight': float(g.mean()),
                                      'p10': float(np.quantile(g, .1)), 'p50': float(np.quantile(g, .5)), 'p90': float(np.quantile(g, .9))})
            print('LOCK', fold, selected, 'validation', validation[selected], 'development', dev_scores[selected], flush=True)
            continue

        selected_file = json.loads((run / 'selection.json').read_text())
        selected = selected_file['folds'][fold]['model']; global_name = selected_file['global_strategy']['model']
        train_lengths = [float(data[s].signals['L'][0]) for s in split['train']]
        for group, sids in [('validation', split['validation']), ('evaluation', split['evaluation']), ('load_stress', split['load_stress'])]:
            for sid in sids:
                raw = packet(sid); pred, seeds, gates = forecasts(raw)
                pred['selected_candidate'] = pred[selected]; pred['global_strategy'] = pred[global_name]
                active = np.abs(raw['initial_velocity']) >= cfg['active_threshold']
                length = float(data[sid].signals['L'][0]); extra = length < min(train_lengths) or length > max(train_lengths)
                for name, y in pred.items():
                    for regime, mask in [('all', np.ones(len(y), bool)), ('active', active), ('quiet', ~active)]:
                        if not mask.any(): continue
                        for j, h in enumerate(HORIZONS_S):
                            metric_rows.append({'fold': fold, 'group': group, 'scenario': sid, 'extrapolation': extra,
                                                'model': name, 'regime': regime, 'horizon_s': h,
                                                **regression_metrics(raw['target_q'][mask, HORIZON_INDEX[j]], y[mask, HORIZON_INDEX[j]])})
                np.savez_compressed(directory / f'predictions_s{sid}.npz', k=raw['k'], target=raw['target_q'], **pred,
                                    **{n + '__seeds': y for n, y in seeds.items()})
                print('EVALUATED', fold, sid, flush=True)
        if fold == 'primary':
            c = json.loads(Path(cfg['confirmation_config']).read_text())[cfg['confirmation_field']]
            meta, theta, target, pred, seeds, gates = synthetic(c)
            _, _, finer = simulate_cases(c, cal, dt=.0005)
            error = float(np.abs(np.rad2deg(target - finer)).max())
            if error > 1e-7: raise AssertionError('Integrator convergence')
            pred['selected_candidate'] = pred[selected]; pred['global_strategy'] = pred[global_name]
            rows = []
            for name, y in pred.items():
                for j, h in enumerate(HORIZONS_S):
                    local = meta.copy(); local['model'] = name; local['horizon_s'] = h
                    local['absolute_error_deg'] = np.abs(np.rad2deg(y[:, HORIZON_INDEX[j]] - target[:, HORIZON_INDEX[j]]))
                    rows.append(local)
            pd.concat(rows).to_csv(run / 'confirmation_metrics.csv', index=False)
            meta.to_csv(run / 'confirmation_cases.csv', index=False)
            np.savez_compressed(run / 'confirmation_predictions.npz', target=target, theta=theta, **pred,
                                **{n + '__seeds': y for n, y in seeds.items()})
            provenance['confirmation'] = {'cases': len(meta), 'trajectories': len(meta) // 2, 'rk4_half_step_max_deg': error}
    if args.stage == 'select':
        common = [n for n in selections[cfg['folds'][0]]['eligible'] if all(n in selections[f]['eligible'] for f in cfg['folds'])]
        def normalized(name):
            return float(np.mean([selections[f]['development_synthetic_mae'][name] / selections[f]['development_synthetic_mae']['incumbent'] for f in cfg['folds']]))
        global_name = min(common, key=normalized)
        write_json(run / 'selection.json', {'folds': selections, 'global_strategy': {'model': global_name, 'eligible': common,
                    'normalized_development_error': normalized(global_name)}, 'uses_real_test': False, 'uses_reserved_grid': False})
        pd.DataFrame(gate_rows).to_csv(run / 'gate_statistics.csv', index=False)
        provenance.update(status='selection_complete', selection_sha256=sha256(run / 'selection.json'), replay_checks=checks,
                          selection_locked_utc=pd.Timestamp.now(tz='UTC').isoformat())
    else:
        pd.DataFrame(metric_rows).to_csv(run / 'metrics.csv', index=False)
        provenance.update(status='complete', finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                          result_hashes={str(p.relative_to(run)): sha256(p) for p in run.rglob('*') if p.is_file() and p.name != 'provenance.json'})
    write_json(run / 'provenance.json', provenance)


if __name__ == '__main__':
    main()
