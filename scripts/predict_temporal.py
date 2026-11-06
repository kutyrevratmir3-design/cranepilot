"""History-only inference for the retained November 6 campaign; no dataset access."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
from cranepilot.data import sha256
from cranepilot.physics import LengthCalibration
from cranepilot.rollout import HORIZON_INDEX, HORIZONS_S
from cranepilot.innovation import InnovationForecaster
from cranepilot.temporal_observer import temporal_model
from cranepilot.fast_inference import fast_prediction
from scripts.predict_cranepilot import build_packet
from scripts.run_temporal import load_actor, prepare


def predict(run, fold, variant, inputs, fast=True):
    run = Path(run)
    p = json.loads((run/'provenance.json').read_text())
    if p['status'] != 'complete':
        raise ValueError('A completed research run is required')
    if fold not in p['inputs']:
        raise ValueError('Unknown training fold')
    # Normalize stored relative paths so Windows-created checkpoints can be used elsewhere.
    checks = {key.replace('\\', '/'): value for key, value in p['checkpoint_hashes'].items()}
    guarded = {key.replace('\\', '/'): value for key, value in p['guarded_hashes'].items()}
    info = dict(p['inputs'][fold])
    info['actor'] = info['actor'].replace('\\', '/')
    info['weights'] = [x.replace('\\', '/') for x in info['weights']]
    if sha256(Path(info['actor'])) != guarded[info['actor']]:
        raise ValueError('Actuator checkpoint changed')
    if sha256(run/'selection.json') != checks['selection.json']:
        raise ValueError('Locked selection changed')
    if variant == 'selected_candidate':
        variant = json.loads((run/'selection.json').read_text())[fold]['model']
    if variant not in [*p['config']['models'], 'forced_driven_huber']:
        raise ValueError('Unknown variant')
    raw = build_packet(inputs['theta'], inputs['x'], inputs['reference'],
        inputs['length_m'], inputs['future_reference'], LengthCalibration(**info['calibration']))
    torch.set_num_threads(4)
    batch = prepare(raw, load_actor(info))
    result = []
    for i, seed in enumerate(p['config']['seeds']):
        old = variant == 'forced_driven_huber'
        path = Path(info['weights'][i]) if old else run/fold/f'{variant}_seed{seed}.pt'
        expected = guarded[info['weights'][i]] if old else checks[f'{fold}/{variant}_seed{seed}.pt']
        if sha256(path) != expected:
            raise ValueError('Angular checkpoint changed')
        model = InnovationForecaster(True) if old else temporal_model(variant)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        with torch.no_grad():
            if fast and isinstance(model, InnovationForecaster):
                y = fast_prediction(model, batch, batch['acceleration'])
            else:
                y = model(batch, batch['acceleration'])[0]
        result.append(y.numpy()[:, HORIZON_INDEX])
    return np.mean(result, axis=0), np.std(result, axis=0), variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('artifacts/runs/temporal_20261106'))
    parser.add_argument('--fold', default='primary')
    parser.add_argument('--variant', default='selected_candidate')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-flow', action='store_true')
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve earlier forecast files')
    with np.load(a.input, allow_pickle=False) as inputs:
        mean, spread, variant = predict(a.run, a.fold, a.variant, inputs, fast=not a.reference_flow)
    frame = pd.DataFrame({f'angle_deg_{h:g}s': np.rad2deg(mean[:, j]) for j, h in enumerate(HORIZONS_S)})
    for j, h in enumerate(HORIZONS_S):
        frame[f'seed_std_deg_{h:g}s'] = np.rad2deg(spread[:, j])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(a.output, index=False)
    print(f'Saved {variant}; seed spread is not a calibrated confidence interval.')


if __name__ == '__main__':
    main()
