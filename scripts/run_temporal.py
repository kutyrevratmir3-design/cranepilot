"""Locked temporal/objective campaign. Train all folds before evaluating any."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from cranepilot.data import load_dataset, sha256
from cranepilot.physics import LengthCalibration
from cranepilot.rollout import (command_windows, stack_windows, tensors,
    NeuralActuator, train_module, HORIZON_INDEX, HORIZONS_S)
from cranepilot.forced_observer import refine_windows, analytic_rollout
from cranepilot.innovation import InnovationForecaster
from cranepilot.temporal_observer import temporal_model
from cranepilot.benchmark import environment_record
from cranepilot.metrics import regression_metrics


def retained_inputs(fold, split, seeds):
    root = Path('artifacts/runs')/split['retained_run']
    p = json.loads((root/'provenance.json').read_text())
    if p['status'] != 'complete':
        raise ValueError('Retained source run is incomplete')
    original = p['config'] if fold == 'primary' else p['config']['folds'][fold]
    for key in ('train', 'validation', 'evaluation'):
        if original[key] != split[key]:
            raise ValueError('Retained source does not match this split')
    directory = root if fold == 'primary' else root/fold
    calibration = p['calibration'] if fold == 'primary' else p['folds'][fold]['calibration']
    if sorted(row['sid'] for row in calibration['evidence']) != sorted(split['train']):
        raise ValueError('Calibration was not fit on precisely this training split')
    actor = (Path('artifacts/runs/actuator_development/actuator.pt')
             if fold == 'primary' else directory/'actor.pt')
    if fold == 'primary':
        expected_actor = next(h for key, h in p['guarded_hashes'].items()
                              if key.replace('\\', '/').endswith('/actuator.pt'))
    else:
        expected_actor = p['checkpoint_hashes'][str(actor.relative_to(root))]
    if sha256(actor) != expected_actor:
        raise ValueError('Retained actuator changed')
    weights = [directory/f'forced_driven_huber_seed{s}.pt' for s in seeds]
    for path in weights:
        if sha256(path) != p['checkpoint_hashes'][str(path.relative_to(root))]:
            raise ValueError('Retained angular checkpoint changed')
    for path in ('data/processed/dataset.npz', 'data/processed/manifest.json'):
        old = {k.replace('\\', '/'): v for k, v in p['guarded_hashes'].items()}
        if sha256(Path(path)) != old[path]:
            raise ValueError('Retained data changed')
    return {'actor': str(actor), 'weights': [str(x) for x in weights],
            'calibration': calibration, 'source_provenance': str(root/'provenance.json')}


def load_actor(info):
    actor = NeuralActuator()
    actor.load_state_dict(torch.load(info['actor'], weights_only=True))
    return actor.eval()


def prepare(raw, actor):
    refined = refine_windows(raw)
    with torch.no_grad():
        acceleration = actor(tensors(raw))[1]
    refined['physical_base'] = analytic_rollout(refined['initial_q'],
        refined['initial_p'], raw['omega'], acceleration.numpy())
    batch = tensors(refined)
    batch['acceleration'] = acceleration
    return batch


def load_models(run, fold, cfg, info):
    models = {}
    for name in [*cfg['models'], 'forced_driven_huber']:
        models[name] = []
        for i, seed in enumerate(cfg['seeds']):
            old = name == 'forced_driven_huber'
            model = InnovationForecaster(True) if old else temporal_model(name)
            path = Path(info['weights'][i]) if old else run/fold/f'{name}_seed{seed}.pt'
            model.load_state_dict(torch.load(path, weights_only=True))
            models[name].append(model.eval())
    return models


def predictions(models, batch):
    result = {'forced_physics': batch['physical_base'].numpy()}
    with torch.no_grad():
        for name, ensemble in models.items():
            result[name] = np.mean([m(batch, batch['acceleration'])[0].numpy()
                                    for m in ensemble], axis=0)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('artifacts/runs/temporal_20261106'))
    args = parser.parse_args()
    run = args.output
    cfg_path = Path('configs/temporal_campaign.json')
    cfg = json.loads(cfg_path.read_text())
    sources = {fold: retained_inputs(fold, split, cfg['seeds'])
               for fold, split in cfg['folds'].items()}
    guarded = [cfg_path, Path('configs/cleanroom_20261106.json'),
        Path('docs/TEMPORAL_DESIGN.md'), Path(__file__).relative_to(Path.cwd()),
        Path('scripts/run_cleanroom.py'), Path('scripts/predict_cranepilot.py'), Path('data/processed/dataset.npz'),
        Path('data/processed/manifest.json'), *sorted(Path('src/cranepilot').glob('*.py'))]
    for info in sources.values():
        guarded.extend(Path(s) for s in [info['actor'], info['source_provenance'], *info['weights']])
    hashes = {str(path): sha256(path) for path in guarded}
    if args.evaluate:
        p = json.loads((run/'provenance.json').read_text())
        if p['status'] != 'validation_complete' or p['guarded_hashes'] != hashes:
            raise ValueError('Unchanged, completed validation required')
        for path, expected in p['checkpoint_hashes'].items():
            if sha256(run/path) != expected:
                raise ValueError('Trained checkpoint or locked selection changed')
    else:
        if run.exists():
            raise FileExistsError('Preserve earlier runs; choose fresh output')
        run.mkdir(parents=True)
        p = {'status': 'running', 'config': cfg, 'inputs': sources,
             'guarded_hashes': hashes, 'environment': environment_record(),
             'started_utc': pd.Timestamp.now(tz='UTC').isoformat()}
    def save():
        (run/'provenance.json').write_text(json.dumps(p, indent=2))
    if not args.evaluate:
        save()
    torch.set_num_threads(4)
    scenarios = load_dataset(Path('data/processed'))
    logs, scores, selections, metrics = [], [], {}, []
    for fold, split in cfg['folds'].items():
        directory = run/fold
        info = sources[fold]
        cal = LengthCalibration(**info['calibration'])
        actor = load_actor(info)
        def packet(sids, stride):
            raw = stack_windows([command_windows(scenarios[s], cal,
                np.arange(40, len(scenarios[s].time)-150, stride)) for s in sids])
            return raw, prepare(raw, actor)
        if not args.evaluate:
            directory.mkdir()
            _, tr = packet(split['train'], cfg['train_stride'])
            _, va = packet(split['validation'], cfg['evaluation_stride'])
            for name, choice in cfg['models'].items():
                for seed in cfg['seeds']:
                    torch.manual_seed(seed)
                    model = temporal_model(name)
                    def loss(module, batch):
                        y, d = module(batch, batch['acceleration'])
                        error = (y-batch['target_q'])/.1
                        if not module.training:
                            return error[:, HORIZON_INDEX].abs().mean()
                        if choice['loss_steps'] == 'seven':
                            error = error[:, HORIZON_INDEX]
                        fit = F.smooth_l1_loss(error, torch.zeros_like(error),
                                              beta=cfg['smooth_l1_beta'])
                        if d:
                            fit = fit+.01*d['initial_memory'].square().mean()+.01*d['innovation_force'].square().mean()
                        return fit
                    print('FIT', fold, name, seed, flush=True)
                    log = train_module(model, tr, va, loss, cfg['training'], seed)
                    logs.extend({'fold': fold, 'model': name, 'seed': seed, **r} for r in log)
                    torch.save(model.state_dict(), directory/f'{name}_seed{seed}.pt')
                    with torch.no_grad():
                        score = float(torch.rad2deg(model(va, va['acceleration'])[0][:, HORIZON_INDEX]
                            -va['target_q'][:, HORIZON_INDEX]).abs().mean())
                    scores.append({'fold': fold, 'model': name, 'seed': seed,
                        'validation_mae_deg': score, 'parameters': sum(x.numel() for x in model.parameters())})
                    pd.DataFrame(logs).to_csv(run/'training.csv', index=False)
                    pd.DataFrame(scores).to_csv(run/'validation_seeds.csv', index=False)
            pred = predictions(load_models(run, fold, cfg, info), va)
            validation = {name: float(abs(np.rad2deg(y[:, HORIZON_INDEX]
                -va['target_q'].numpy()[:, HORIZON_INDEX])).mean()) for name, y in pred.items()}
            selected = min(cfg['candidate_pool'], key=validation.get)
            selections[fold] = {'model': selected, 'ensemble_validation_mae_deg': validation}
            np.savez_compressed(directory/'validation_full.npz',
                                target=va['target_q'].numpy(), **pred)
            print('VALIDATION', fold, selections[fold], flush=True)
            continue
        models = load_models(run, fold, cfg, info)
        selected = json.loads((run/'selection.json').read_text())[fold]['model']
        train_lengths = [float(scenarios[s].signals['L'][0]) for s in split['train']]
        for group, sids in [('validation', split['validation']), ('evaluation', split['evaluation']),
                            ('load_stress', split['load_stress'])]:
            for sid in sids:
                raw, batch = packet([sid], cfg['evaluation_stride'])
                pred = predictions(models, batch)
                pred['selected_candidate'] = pred[selected]
                length = float(scenarios[sid].signals['L'][0])
                extrapolation = length < min(train_lengths) or length > max(train_lengths)
                for name, y in pred.items():
                    for j, h in enumerate(HORIZONS_S):
                        for regime, mask in [('all', np.ones(len(y), bool)),
                            ('active_at_origin', abs(raw['initial_velocity']) > .005),
                            ('quiet_at_origin', abs(raw['initial_velocity']) <= .005)]:
                            if mask.any():
                                metrics.append({'fold': fold, 'group': group, 'scenario': sid,
                                    'model': name, 'horizon_s': h, 'regime': regime,
                                    'extrapolation': extrapolation, 'length_m': length,
                                    **regression_metrics(raw['target_q'][mask, HORIZON_INDEX[j]],
                                                         y[mask, HORIZON_INDEX[j]])})
                np.savez_compressed(directory/f'predictions_s{sid}.npz', k=raw['k'],
                    target=raw['target_q'], initial_velocity=raw['initial_velocity'], **pred)
                pd.DataFrame(metrics).to_csv(run/'metrics.csv', index=False)
        print('EVALUATED', fold, flush=True)
    if not args.evaluate:
        (run/'selection.json').write_text(json.dumps(selections, indent=2))
        files = [*run.rglob('*.pt'), run/'selection.json']
        p['checkpoint_hashes'] = {str(x.relative_to(run)): sha256(x) for x in files}
        p['status'] = 'validation_complete'
    else:
        p['status'] = 'complete'
        p['finished_utc'] = pd.Timestamp.now(tz='UTC').isoformat()
        print(pd.DataFrame(metrics).query("group == 'evaluation' and regime == 'all' and extrapolation")
              .groupby(['fold', 'model']).mae_deg.mean().to_string())
    save()


if __name__ == '__main__':
    main()
