"""Locked matched continuation, validation selection, then held-out evaluation."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from cranepilot.benchmark import environment_record
from cranepilot.broad_strategies import make_model, fit_loss
from cranepilot.data import load_dataset, sha256
from cranepilot.forced_observer import refine_windows, analytic_rollout
from cranepilot.physics import LengthCalibration
from cranepilot.rollout import command_windows, stack_windows, tensors, HORIZON_INDEX, HORIZONS_S
from cranepilot.metrics import regression_metrics
from scripts.predict_cranepilot import build_packet
from scripts.run_cleanroom import simulate_cases
from scripts.run_temporal import load_actor


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False))


def inputs(raw, actor=None):
    refined = refine_windows(raw)
    b = tensors(refined)
    with torch.no_grad():
        acceleration = actor(tensors(raw))[1] if actor is not None else torch.zeros(len(raw['initial_q']), 30)
    result = {k: b[k] for k in ['angular_context', 'initial_q', 'initial_p', 'omega', 'future_u']}
    result['observer_residual_rms'] = torch.as_tensor(np.sqrt(np.mean(refined['forced_residual']**2, axis=1)), dtype=torch.float32)
    result['acceleration'] = acceleration
    return result


@torch.no_grad()
def predict(model, batch, chunk=512):
    model.eval()
    ys = []
    for start in range(0, len(batch['omega']), chunk):
        b = {k: v[start:start + chunk] for k, v in batch.items()}
        ys.append(model(b, b['acceleration'])[0].numpy())
    return np.concatenate(ys)


def scores(prediction, target, active):
    error = np.abs(np.rad2deg(prediction[:, HORIZON_INDEX] - target[:, HORIZON_INDEX]))
    if not active.any() or active.all():
        raise ValueError('Both validation motion regimes are required')
    return {'all': float(error.mean()), 'active': float(error[active].mean()),
            'quiet': float(error[~active].mean())}


def criterion(value, reference):
    return .5 * (value['all'] / reference['all'] + value['active'] / reference['active'])


def eligible(value, reference, cfg):
    return all(value[g] <= reference[g] * cfg[f'{g}_max_ratio'] + 1e-12 for g in ('all', 'active', 'quiet'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['train', 'evaluate', 'verify'], required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/broad_search_20261107.json'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/runs/broad_search_20261107'))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text()); run = args.output
    source = Path(cfg['source_run']); old = json.loads((source / 'provenance.json').read_text())
    if old['status'] != 'complete':
        raise ValueError('Completed source required')
    for mapping, base in [('guarded_hashes', Path('.')), ('checkpoint_hashes', source)]:
        for path, expected in old[mapping].items():
            if sha256(base / path.replace('\\', '/')) != expected:
                raise ValueError(f'Changed retained input: {path}')
    locked = [args.config, Path('docs/BROAD_SEARCH_DESIGN_20261107.md'),
              Path('scripts/run_broad_search.py'), Path('scripts/run_cleanroom.py'),
              Path('scripts/predict_cranepilot.py'), Path('scripts/run_temporal.py'),
              Path('tests/test_broad_strategies.py'), Path(cfg['confirmation_config']),
              Path('data/processed/dataset.npz'), Path('data/processed/manifest.json'),
              source / 'provenance.json', *sorted(Path('src/cranepilot').glob('*.py')),
              *sorted(source.glob('*/predictions_s*.npz'))]
    sources = {}
    for fold in cfg['folds']:
        info = old['inputs'][fold]
        sources[fold] = dict(info, actor=info['actor'].replace('\\', '/'),
                             weights=[p.replace('\\', '/') for p in info['weights']])
        locked.extend([Path(sources[fold]['actor']), *map(Path, sources[fold]['weights'])])
        if sorted(r['sid'] for r in info['calibration']['evidence']) != sorted(old['config']['folds'][fold]['train']):
            raise ValueError('Calibration split mismatch')
    hashes = {str(p): sha256(p) for p in locked}
    if args.stage == 'train':
        if run.exists():
            raise FileExistsError('Preserve retained runs; use a new configuration and output')
        # New experimental sources/config/design must be committed before fitting.
        for path in [args.config, Path('docs/BROAD_SEARCH_DESIGN_20261107.md'),
                     Path('scripts/run_broad_search.py'), Path('src/cranepilot/broad_strategies.py'),
                     Path('tests/test_broad_strategies.py')]:
            committed = subprocess.check_output(['git', 'show', f'HEAD:{path.as_posix()}'])
            if committed != path.read_bytes():
                raise ValueError(f'Commit experimental source first: {path}')
        run.mkdir(parents=True)
        provenance = {'status': 'training', 'config': cfg, 'inputs': sources,
                      'guarded_hashes': hashes, 'environment': environment_record(),
                      'started_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'selected_before_evaluation': False}
    else:
        provenance = json.loads((run / 'provenance.json').read_text())
        if provenance['guarded_hashes'] != hashes:
            raise ValueError('Source/config/data changed since training')
        required = 'validation_complete' if args.stage == 'evaluate' else 'complete'
        if provenance['status'] != required:
            raise ValueError(f'{required} run required')
        for path, expected in provenance['checkpoint_hashes'].items():
            if sha256(run / path) != expected:
                raise ValueError(f'Changed selected checkpoint: {path}')
    def save():
        write_json(run / 'provenance.json', provenance)
    if args.stage == 'train': save()
    torch.set_num_threads(4)
    data = load_dataset(Path('data/processed'))
    selection, validation_scores, logs, fit_rows = {}, {}, [], []
    metrics, replay = [], []

    def record_metrics(fold, group, sid, extra, raw, predictions):
        active = np.abs(raw['initial_velocity']) >= cfg['active_threshold']
        for name, y in predictions.items():
            for regime, mask in [('all', np.ones(len(y), bool)), ('active', active), ('quiet', ~active)]:
                if not mask.any(): continue
                for j, h in enumerate(HORIZONS_S):
                    metrics.append({'fold': fold, 'group': group, 'scenario': sid, 'extrapolation': extra,
                                    'model': name, 'regime': regime, 'horizon_s': h,
                                    **regression_metrics(raw['target_q'][mask, HORIZON_INDEX[j]], y[mask, HORIZON_INDEX[j]])})

    for fold in cfg['folds']:
        split = old['config']['folds'][fold]; info = sources[fold]
        cal = LengthCalibration(**info['calibration']); actor = load_actor(info)
        directory = run / fold
        def packet(sids, stride):
            raw = stack_windows([command_windows(data[s], cal, np.arange(40, len(data[s].time) - 150, stride)) for s in sids])
            return raw, inputs(raw, actor)
        raw_va, va = packet(split['validation'], cfg['evaluation_stride'])
        va_active = np.abs(raw_va['initial_velocity']) >= cfg['active_threshold']
        incumbent, normalizers = [], []
        for seed, weight in zip(cfg['seeds'], info['weights']):
            base = make_model({}); base.load_state_dict(torch.load(weight, weights_only=True))
            y = predict(base, va); incumbent.append(y)
            normalizers.append(scores(y, raw_va['target_q'], va_active))
        incumbent = np.stack(incumbent)
        if args.stage == 'train':
            directory.mkdir()
            raw_tr, tr = packet(split['train'], cfg['train_stride'])
            target_tr = torch.as_tensor(raw_tr['target_q'], dtype=torch.float32)
            active_tr = torch.as_tensor(np.abs(raw_tr['initial_velocity']) >= cfg['active_threshold'])
            lengths = [float(data[s].signals['L'][0]) for s in split['train']]
            synth_cfg = {k: v for k, v in cfg['synthetic_training'].items() if k != 'length_rule'}
            synth_cfg['length_m'] = np.linspace(min(lengths), max(lengths), 5).tolist()
            meta, theta, synth_target = simulate_cases(synth_cfg, cal)
            syn_raw = build_packet(theta, np.zeros_like(theta), np.zeros_like(theta), meta.length_m.to_numpy(), np.zeros((len(theta), 30)), cal)
            synthetic = inputs(syn_raw)
            synth_y = torch.as_tensor(synth_target, dtype=torch.float32)
            meta.to_csv(directory / 'synthetic_training_cases.csv', index=False)
            write_json(directory / 'synthetic_training_config.json', synth_cfg)
            validation = {'incumbent': incumbent}
            for name, choice in cfg['variants'].items():
                predictions = []
                for i, seed in enumerate(cfg['seeds']):
                    started = time.perf_counter(); torch.manual_seed(seed)
                    model = make_model(choice); model.load_state_dict(torch.load(info['weights'][i], weights_only=True))
                    rng = np.random.default_rng(seed); syn_rng = np.random.default_rng(1907 + seed)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])
                    best_values = scores(predict(model, va), raw_va['target_q'], va_active)
                    best_score = criterion(best_values, normalizers[i]); best_epoch = -1
                    best_state = copy.deepcopy(model.state_dict()); stale = 0
                    logs.append({'fold': fold, 'model': name, 'seed': seed, 'epoch': -1,
                                 'selection_score': best_score, **{f'validation_{k}': v for k, v in best_values.items()}})
                    print(f'FIT {fold} {name} seed={seed}', flush=True)
                    for epoch in range(cfg['training']['epochs']):
                        model.train(); losses = []
                        for ids in np.array_split(rng.permutation(len(target_tr)), int(np.ceil(len(target_tr) / cfg['training']['batch_size']))):
                            b = {k: v[ids] for k, v in tr.items()}
                            y, diag = model(b, b['acceleration'])
                            loss = fit_loss(y, target_tr[ids], active_tr[ids], choice.get('balanced', False), cfg['smooth_l1_beta'])
                            loss = loss + .01 * diag['initial_memory'].square().mean() + .01 * diag['innovation_force'].square().mean()
                            if choice.get('synthetic_weight', 0):
                                si = syn_rng.integers(0, len(synth_y), cfg['synthetic_batch_size'])
                                sb = {k: v[si] for k, v in synthetic.items()}
                                sy, _ = model(sb, sb['acceleration'])
                                loss = loss + choice['synthetic_weight'] * fit_loss(sy, synth_y[si], torch.zeros(len(si), dtype=torch.bool), beta=cfg['smooth_l1_beta'])
                            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                            optimizer.zero_grad(set_to_none=True); loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
                            optimizer.step(); losses.append(float(loss.detach()))
                        values = scores(predict(model, va), raw_va['target_q'], va_active)
                        score = criterion(values, normalizers[i])
                        logs.append({'fold': fold, 'model': name, 'seed': seed, 'epoch': epoch,
                                     'training_loss': float(np.mean(losses)), 'selection_score': score,
                                     **{f'validation_{k}': v for k, v in values.items()}})
                        if score < best_score - 1e-8:
                            best_score, best_epoch, stale = score, epoch, 0
                            best_state = copy.deepcopy(model.state_dict()); best_values = values
                        else: stale += 1
                        if epoch % 10 == 0:
                            print(f'  epoch={epoch} selection={score:.6f} best={best_score:.6f}', flush=True)
                        if stale >= cfg['training']['patience']: break
                    model.load_state_dict(best_state); model.eval()
                    torch.save(model.state_dict(), directory / f'{name}_seed{seed}.pt')
                    predictions.append(predict(model, va))
                    fit_rows.append({'fold': fold, 'model': name, 'seed': seed, 'selected_epoch': best_epoch,
                                     'epochs_executed': epoch + 1, 'selection_score': best_score,
                                     'fit_seconds': time.perf_counter() - started,
                                     **{f'validation_{k}': v for k, v in best_values.items()}})
                    pd.DataFrame(logs).to_csv(run / 'training.csv', index=False)
                    pd.DataFrame(fit_rows).to_csv(run / 'fits.csv', index=False)
                validation[name] = np.stack(predictions)
            values = {name: scores(ys.mean(0), raw_va['target_q'], va_active) for name, ys in validation.items()}
            pool = [n for n in values if eligible(values[n], values['incumbent'], cfg['ensemble_selection'])]
            selected = min(pool, key=lambda n: values[n]['all'])
            selection[fold] = {'model': selected, 'eligible': pool, 'validation': values}
            validation_scores[fold] = values
            np.savez_compressed(directory / 'validation_predictions.npz', k=raw_va['k'], target=raw_va['target_q'], active=va_active, **validation)
            with np.load(source / fold / f'predictions_s{split["validation"][0]}.npz') as retained:
                np.testing.assert_array_equal(raw_va['k'], retained['k'])
                error = float(np.abs(np.rad2deg(incumbent.mean(0) - retained['forced_driven_huber'])).max())
            if error > cfg['replay_tolerance_deg']: raise AssertionError(f'Incumbent replay {fold}: {error}')
            replay.append({'fold': fold, 'incumbent_validation_max_difference_deg': error})
            print('SELECTED', fold, selected, values[selected], flush=True)
            continue

        ensembles = {}
        for name, choice in cfg['variants'].items():
            ensembles[name] = []
            for seed in cfg['seeds']:
                m = make_model(choice); m.load_state_dict(torch.load(directory / f'{name}_seed{seed}.pt', weights_only=True))
                ensembles[name].append(m.eval())
        if args.stage == 'verify':
            indices = [0, len(raw_va['k']) // 2, len(raw_va['k']) - 1]
            small = {k: v[indices] for k, v in va.items()}
            with np.load(directory / 'validation_predictions.npz') as retained:
                for name, models in ensembles.items():
                    for i, model in enumerate(models):
                        y = predict(model, small)
                        error = float(np.abs(np.rad2deg(y - retained[name][i, indices])).max())
                        if error > cfg['replay_tolerance_deg']: raise AssertionError((fold, name, i, error))
                        replay.append({'fold': fold, 'model': name, 'seed': cfg['seeds'][i], 'max_difference_deg': error})
            continue
        locked_selection = json.loads((run / 'selection.json').read_text())
        selected = locked_selection['folds'][fold]['model']
        global_strategy = locked_selection['global_strategy']['model']
        train_lengths = [float(data[s].signals['L'][0]) for s in split['train']]
        for group, sids in [('validation', split['validation']), ('evaluation', split['evaluation']), ('load_stress', split['load_stress'])]:
            for sid in sids:
                raw, batch = packet([sid], cfg['evaluation_stride'])
                seeds = {n: np.stack([predict(m, batch) for m in ms]) for n, ms in ensembles.items()}
                predictions = {n: ys.mean(0) for n, ys in seeds.items()}
                with np.load(source / fold / f'predictions_s{sid}.npz') as retained:
                    np.testing.assert_array_equal(retained['k'], raw['k'])
                    np.testing.assert_array_equal(retained['target'], raw['target_q'])
                    predictions['incumbent'] = retained['forced_driven_huber'].copy()
                    for name in ['forced_physics', 'dense_horizon', 'temporal_uniform', 'temporal_horizon', 'tcn_reference']:
                        predictions['retained_' + name] = retained[name].copy()
                predictions['selected_candidate'] = predictions[selected]
                predictions['global_strategy'] = predictions[global_strategy]
                length = float(data[sid].signals['L'][0])
                extra = length < min(train_lengths) or length > max(train_lengths)
                record_metrics(fold, group, sid, extra, raw, predictions)
                np.savez_compressed(directory / f'predictions_s{sid}.npz', k=raw['k'], target=raw['target_q'],
                                    **predictions, **{n + '__seeds': ys for n, ys in seeds.items()})
                pd.DataFrame(metrics).to_csv(run / 'metrics.csv', index=False)
                print('EVALUATED', fold, group, sid, flush=True)
        if fold == cfg['confirmation_fold']:
            # Selection for every fold was locked before this reserved grid is opened.
            c = json.loads(Path(cfg['confirmation_config']).read_text())[cfg['confirmation_field']]
            meta, theta, target = simulate_cases(c, cal)
            _, _, finer = simulate_cases(c, cal, dt=.0005)
            error = float(np.abs(np.rad2deg(finer - target)).max())
            if error > 1e-7: raise AssertionError('Confirmation integration resolution')
            raw = build_packet(theta, np.zeros_like(theta), np.zeros_like(theta), meta.length_m.to_numpy(), np.zeros((len(theta), 30)), cal)
            batch = inputs(raw)
            ys = {n: np.stack([predict(m, batch) for m in ms]) for n, ms in ensembles.items()}
            prior_models = []
            for weight in info['weights']:
                m = make_model({}); m.load_state_dict(torch.load(weight, weights_only=True)); prior_models.append(m)
            ys['incumbent'] = np.stack([predict(m, batch) for m in prior_models])
            physical = analytic_rollout(batch['initial_q'].double().numpy(), batch['initial_p'].double().numpy(),
                                        batch['omega'].double().numpy(), batch['acceleration'].double().numpy())
            pred = {n: arr.mean(0) for n, arr in ys.items()}; pred['physical'] = physical
            pred['selected_candidate'] = pred[selected]; pred['global_strategy'] = pred[global_strategy]
            rows = []
            for name, y in pred.items():
                for j, horizon in enumerate(HORIZONS_S):
                    local = meta.copy(); local['model'] = name; local['horizon_s'] = horizon
                    local['absolute_error_deg'] = np.abs(np.rad2deg(y[:, HORIZON_INDEX[j]] - target[:, HORIZON_INDEX[j]]))
                    rows.append(local)
            pd.concat(rows).to_csv(run / 'confirmation_metrics.csv', index=False)
            meta.to_csv(run / 'confirmation_cases.csv', index=False)
            np.savez_compressed(run / 'confirmation_predictions.npz', theta=theta, target=target, **pred,
                                **{n + '__seeds': arr for n, arr in ys.items()})
            provenance['confirmation'] = {'cases': len(meta), 'trajectories': len(meta) // 2,
                                          'rk4_half_step_max_deg': error, 'selection_was_locked': True}

    if args.stage == 'train':
        universal = [n for n in ['incumbent', *cfg['variants']] if all(n in selection[f]['eligible'] for f in cfg['folds'])]
        def mean_ratio(name):
            return float(np.mean([validation_scores[f][name]['all'] / validation_scores[f]['incumbent']['all'] for f in cfg['folds']]))
        global_name = min(universal, key=mean_ratio)
        write_json(run / 'selection.json', {'folds': selection, 'global_strategy': {'model': global_name, 'eligible': universal,
                    'mean_validation_all_ratio': mean_ratio(global_name)}, 'uses_test': False, 'uses_confirmation': False})
        provenance['checkpoint_hashes'] = {str(p.relative_to(run)): sha256(p) for p in [*run.rglob('*.pt'), run / 'selection.json']}
        provenance.update(status='validation_complete', selected_before_evaluation=True, incumbent_replay=replay,
                          completed_fits=len(fit_rows), finished_training_utc=pd.Timestamp.now(tz='UTC').isoformat())
        save()
    elif args.stage == 'evaluate':
        provenance.update(status='complete', finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                          result_hashes={str(p.relative_to(run)): sha256(p) for p in run.rglob('*') if p.is_file() and p.name != 'provenance.json'})
        save()
    else:
        write_json(run / 'verification.json', {'checked_checkpoints': len(replay), 'packets_each': 3, 'tolerance_deg': cfg['replay_tolerance_deg'],
                    'maximum_difference_deg': max(r['max_difference_deg'] for r in replay), 'checks': replay})
        print('VERIFIED', len(replay), 'checkpoints', flush=True)


if __name__ == '__main__':
    main()
