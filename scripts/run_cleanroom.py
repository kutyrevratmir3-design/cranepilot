"""Frozen angular-only free-sway falsification test. No model fitting."""
from pathlib import Path
from itertools import product
import argparse
import json
import numpy as np
import pandas as pd
import torch
from cranepilot.data import QUANTIZATION_RAD, sha256
from cranepilot.physics import LengthCalibration
from cranepilot.rollout import tensors, HORIZON_INDEX, HORIZONS_S
from cranepilot.forced_observer import refine_windows, analytic_rollout
from scripts.predict_cranepilot import build_packet
from scripts.run_temporal import load_models, predictions


def simulate_cases(cfg, calibration, dt=.001):
    """Start at -0.4 s; integrate an independent nonlinear ODE through +1.5 s."""
    rows = [dict(zip(('length_m', 'amplitude_deg', 'phase_rad', 'damping_per_s'), row))
            for row in product(cfg['length_m'], cfg['amplitude_deg'],
                               cfg['phase_rad'], cfg['damping_per_s'])]
    meta = pd.DataFrame(rows)
    omega = calibration.omega(meta.length_m.to_numpy())
    amplitude = np.deg2rad(meta.amplitude_deg.to_numpy())
    phase = meta.phase_rad.to_numpy()
    q = amplitude*np.cos(phase)
    v = -omega*amplitude*np.sin(phase)
    damping = meta.damping_per_s.to_numpy()
    state = np.column_stack([q, v])
    history = [q.copy()]
    def rhs(z):
        return np.column_stack([z[:, 1], -omega**2*np.sin(z[:, 0])-damping*z[:, 1]])
    steps = int(round(1.9/dt))
    sample_every = int(round(.01/dt))
    if abs(sample_every*dt-.01) > 1e-12:
        raise ValueError('Integration step must divide 10 ms')
    for step in range(steps):
        a = rhs(state)
        b = rhs(state+.5*dt*a)
        c = rhs(state+.5*dt*b)
        d = rhs(state+dt*c)
        state += dt*(a+2*b+2*c+d)/6
        if (step+1) % sample_every == 0:
            history.append(state[:, 0].copy())
    trajectory = np.stack(history, axis=1)
    parts, past, targets = [], [], []
    for measurement in cfg['measurement']:
        local = meta.copy()
        local['measurement'] = measurement
        local['trajectory_id'] = np.arange(len(meta))
        parts.append(local)
        y = trajectory[:, :41][:, ::-1].copy()
        if measurement == 'quantized':
            y = np.round(y/QUANTIZATION_RAD)*QUANTIZATION_RAD
        elif measurement != 'exact':
            raise ValueError('Unknown measurement mode')
        past.append(y)
        targets.append(trajectory[:, 40+np.arange(1, 31)*5])
    return pd.concat(parts, ignore_index=True), np.concatenate(past), np.concatenate(targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path('artifacts/runs/temporal_20261106'))
    args = parser.parse_args()
    run = args.run
    p = json.loads((run/'provenance.json').read_text())
    if p['status'] != 'complete':
        raise ValueError('Finish the locked real-data campaign first')
    for path, expected in p['guarded_hashes'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('Code/config/input changed since training')
    for path, expected in p['checkpoint_hashes'].items():
        if sha256(run/path) != expected:
            raise ValueError('Checkpoint changed')
    out = run/'cleanroom'
    if out.exists():
        raise FileExistsError('Preserve existing diagnostic results')
    out.mkdir()
    cfg = json.loads(Path('configs/cleanroom_20261106.json').read_text())
    cal = LengthCalibration(**p['inputs']['primary']['calibration'])
    metadata, theta, target = simulate_cases(cfg, cal)
    raw = build_packet(theta, np.zeros_like(theta), np.zeros_like(theta),
        metadata.length_m.to_numpy(), np.zeros((len(theta), 30)), cal)
    refined = refine_windows(raw)
    acceleration = np.zeros((len(theta), 30))
    refined['physical_base'] = analytic_rollout(refined['initial_q'], refined['initial_p'],
                                               raw['omega'], acceleration)
    batch = tensors(refined)
    batch['acceleration'] = torch.as_tensor(acceleration, dtype=torch.float32)
    torch.set_num_threads(4)
    models = load_models(run, 'primary', p['config'], p['inputs']['primary'])
    pred = predictions(models, batch)
    rows = []
    for name, y in pred.items():
        for j, h in enumerate(HORIZONS_S):
            error = np.rad2deg(y[:, HORIZON_INDEX[j]]-target[:, HORIZON_INDEX[j]])
            for i, e in enumerate(error):
                rows.append({'case': i, 'model': name, 'horizon_s': h,
                             'absolute_error_deg': abs(e), 'signed_error_deg': e,
                             **metadata.iloc[i].to_dict()})
    metrics = pd.DataFrame(rows)
    metadata.to_csv(out/'cases.csv', index=False)
    metrics.to_csv(out/'metrics.csv', index=False)
    np.savez_compressed(out/'predictions.npz', theta=theta, target=target, **pred)
    # Independent integrator resolution audit; does not select any model.
    _, _, finer = simulate_cases(cfg, cal, dt=.0005)
    audit = {'config': cfg, 'cases': len(theta), 'physical_trajectories': int(len(theta)/2),
             'rk4_resolution_max_deg': float(np.max(abs(np.rad2deg(finer-target)))),
             'angular_only': True, 'selected_or_fitted_here': False}
    (out/'provenance.json').write_text(json.dumps(audit, indent=2))
    print(metrics.groupby(['measurement', 'model']).absolute_error_deg.mean().to_string())
    print('RK4 RESOLUTION', audit['rk4_resolution_max_deg'])


if __name__ == '__main__':
    main()
