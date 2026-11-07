"""History-only inference for locked broad-search or reliability-blend candidates."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from cranepilot.broad_strategies import make_model
from cranepilot.data import sha256
from cranepilot.forced_observer import refine_windows, analytic_rollout
from cranepilot.physics import LengthCalibration
from cranepilot.rollout import HORIZON_INDEX, HORIZONS_S
from scripts.run_temporal import load_actor
from scripts.run_broad_search import inputs as prepare_inputs, predict
from scripts.predict_cranepilot import build_packet
from scripts.reliability_blend import blend


def forecast(run, fold, variant, packet):
    run = Path(run)
    provenance = json.loads((run / 'provenance.json').read_text())
    if provenance['status'] != 'complete': raise ValueError('Completed and evaluated run required')
    cfg = provenance['config']
    if fold not in cfg['folds']: raise ValueError('Unknown fold')
    selection_path = run / 'selection.json'
    expected = (provenance['checkpoint_hashes']['selection.json'] if 'variants' in cfg
                else provenance['selection_sha256'])
    if sha256(selection_path) != expected: raise ValueError('Selection changed')
    selection = json.loads(selection_path.read_text())
    if variant == 'selected': variant = selection['folds'][fold]['model']
    if variant == 'global': variant = selection['global_strategy']['model']
    names = (['incumbent', *cfg['variants']] if 'variants' in cfg
             else ['incumbent', 'physical', *[f'gate_{t:g}' for t in cfg['tau_bins']]])
    if variant not in names: raise ValueError('Unknown variant')
    source = Path(cfg['source_run'])
    if sha256(source / 'provenance.json') != provenance['guarded_hashes'][str(source / 'provenance.json')]:
        raise ValueError('Original provenance changed')
    original = json.loads((source / 'provenance.json').read_text())
    info = original['inputs'][fold]
    info = dict(info, actor=info['actor'].replace('\\', '/'), weights=[p.replace('\\', '/') for p in info['weights']])
    # Inference needs source, actor and weights; it does not require a dataset file.
    for path, expected in provenance['guarded_hashes'].items():
        if path.endswith('.py') or path == info['actor']:
            if sha256(Path(path)) != expected: raise ValueError(f'Inference dependency changed: {path}')
    raw = build_packet(packet['theta'], packet['x'], packet['reference'], packet['length_m'], packet['future_reference'],
                       LengthCalibration(**info['calibration']))
    torch.set_num_threads(4)
    batch = prepare_inputs(raw, load_actor(info))
    predictions = []
    for i, seed in enumerate(cfg['seeds']):
        is_new = 'variants' in cfg and variant != 'incumbent'
        if is_new:
            relative = f'{fold}/{variant}_seed{seed}.pt'; path = run / relative
            expected = provenance['checkpoint_hashes'][relative]; choice = cfg['variants'][variant]
        else:
            path = Path(info['weights'][i]); expected = provenance['guarded_hashes'][str(path)]; choice = {}
        if sha256(path) != expected: raise ValueError(f'Weight changed: {path}')
        model = make_model(choice); model.load_state_dict(torch.load(path, weights_only=True))
        predictions.append(predict(model, batch))
    if 'variants' not in cfg and variant != 'incumbent':
        refined = refine_windows(raw)
        physical = analytic_rollout(refined['initial_q'], refined['initial_p'], raw['omega'], batch['acceleration'].numpy())
        if variant == 'physical':
            predictions = [physical for _ in predictions]
        else:
            rms = np.sqrt(np.mean(refined['forced_residual']**2, axis=1)); tau = float(variant.split('_', 1)[1])
            predictions = [blend(y, physical, rms, tau)[0] for y in predictions]
    ys = np.stack(predictions)[:, :, HORIZON_INDEX]
    return ys.mean(0), ys.std(0), variant


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--fold', default='primary')
    p.add_argument('--variant', default='selected')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError('Choose a new output file')
    with np.load(a.input, allow_pickle=False) as packet:
        mean, spread, resolved = forecast(a.run, a.fold, a.variant, packet)
    frame = pd.DataFrame({f'angle_deg_{h:g}s': np.rad2deg(mean[:, j]) for j, h in enumerate(HORIZONS_S)})
    for j, h in enumerate(HORIZONS_S): frame[f'seed_std_deg_{h:g}s'] = np.rad2deg(spread[:, j])
    a.output.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(a.output, index=False)
    print(f'Saved {resolved}; seed spread is descriptive, not a confidence interval.')


if __name__ == '__main__': main()
