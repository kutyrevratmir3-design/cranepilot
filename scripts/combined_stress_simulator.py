"""Fixed nonlinear mismatch/noise stress for frozen angular forecasters.

Observed angles stop at t=0 and are returned newest first. Targets are clean
angles at 0.05,...,1.50 s. A future horizontal disturbance, when configured,
is unobserved by every predictor and begins at +0.25 s, not in the history.
This is controlled model mismatch, not independently collected hardware data.
"""
from itertools import product
import numpy as np
import pandas as pd
from cranepilot.data import QUANTIZATION_RAD


MEASUREMENT_MODES = {'exact', 'quantized', 'noisy_quantized', 'biased_quantized'}


def simulate_stress(cfg, calibration, dt=.001):
    """Return (metadata, observed_history[N,41], clean_future[N,30]).

    A trajectory identifier is shared by its different observation variants.
    omega_true is the train-calibrated omega multiplied by a fixed mismatch
    ratio, while the runner supplies the unchanged nominal length/calibration.
    Optional noise_std_bins and bias_bins are in angle encoder bins, not radians.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('Positive integration step must divide 10 ms')
    sample_every = int(round(.01 / dt))
    if sample_every < 1 or abs(sample_every * dt - .01) > 1e-12:
        raise ValueError('Positive integration step must divide 10 ms')
    modes = cfg['measurement']
    if not modes or len(set(modes)) != len(modes) or not set(modes) <= MEASUREMENT_MODES:
        raise ValueError('Distinct supported measurement modes required')
    conditions = cfg['conditions']
    if not conditions or len({c['name'] for c in conditions}) != len(conditions):
        raise ValueError('Distinct named conditions required')
    rows = []
    for length, amplitude, phase, condition in product(
            cfg['length_m'], cfg['amplitude_deg'], cfg['phase_rad'], conditions):
        rows.append({'length_m': length, 'actual_length_m': length,
                     'input_length_m': length, 'amplitude_deg': amplitude,
                     'phase_rad': phase, 'condition': condition['name'],
                     'omega_ratio': condition['omega_ratio'],
                     'damping_per_s': condition['damping_per_s'],
                     'future_accel_m_s2': condition.get('future_accel_m_s2', 0.0)})
    meta = pd.DataFrame(rows)
    if meta.empty:
        raise ValueError('Nonempty simulation grid required')
    numeric = meta.drop(columns='condition').to_numpy(dtype=float)
    if (not np.isfinite(numeric).all() or (meta.length_m <= 0).any()
            or (meta.omega_ratio <= 0).any() or (meta.damping_per_s < 0).any()):
        raise ValueError('Finite grid with positive lengths/frequencies and nonnegative damping required')
    length = meta.length_m.to_numpy(dtype=float)
    nominal = calibration.omega(length)
    omega = nominal * meta.omega_ratio.to_numpy()
    amplitude = np.deg2rad(meta.amplitude_deg.to_numpy())
    phase = meta.phase_rad.to_numpy()
    damping = meta.damping_per_s.to_numpy()
    disturbance = meta.future_accel_m_s2.to_numpy()
    meta['nominal_omega_rad_s'] = nominal
    meta['true_omega_rad_s'] = omega
    meta['trajectory_id'] = np.arange(len(meta))
    meta['disturbance_start_s'] = .25
    meta['disturbance_end_s'] = .50
    state = np.column_stack([amplitude * np.cos(phase), -omega * amplitude * np.sin(phase)])
    trajectory = [state[:, 0].copy()]

    def rhs(z, acceleration):
        return np.column_stack([z[:, 1], -omega**2 * np.sin(z[:, 0])
                                - damping * z[:, 1]
                                - acceleration * np.cos(z[:, 0]) / length])

    for step in range(190 * sample_every):
        # Both discontinuities lie on sample boundaries. Hold the appropriate
        # one-sided forcing over each RK4 interval, avoiding endpoint leakage.
        midpoint = -.4 + (step + .5) * dt
        acceleration = disturbance if .25 <= midpoint < .50 else np.zeros_like(disturbance)
        a = rhs(state, acceleration)
        b = rhs(state + .5 * dt * a, acceleration)
        c = rhs(state + .5 * dt * b, acceleration)
        d = rhs(state + dt * c, acceleration)
        state += dt * (a + 2*b + 2*c + d) / 6
        if (step + 1) % sample_every == 0:
            trajectory.append(state[:, 0].copy())
    trajectory = np.stack(trajectory, axis=1)
    clean_past = trajectory[:, :41][:, ::-1].copy()
    clean_target = trajectory[:, 40 + np.arange(1, 31) * 5]
    noise_std = float(cfg.get('noise_std_bins', .25))
    bias = float(cfg.get('bias_bins', .5))
    if not np.isfinite([noise_std, bias]).all() or noise_std < 0:
        raise ValueError('Finite bias and nonnegative finite noise required')
    rng = np.random.default_rng(cfg.get('noise_seed', 7112026))
    # Match measurement noise across dynamics conditions for the same initial
    # length/amplitude/phase. In particular, a hidden future disturbance must
    # not change even the noisy observed input before its onset.
    noise_keys = list(zip(meta.length_m, meta.amplitude_deg, meta.phase_rad))
    unique_keys = {key: i for i, key in enumerate(dict.fromkeys(noise_keys))}
    noise_ids = np.array([unique_keys[key] for key in noise_keys])
    meta['noise_pair_id'] = noise_ids
    noise_bank = rng.normal(0, noise_std * QUANTIZATION_RAD, size=(len(unique_keys), 41))
    noise = noise_bank[noise_ids]
    frames, histories, targets = [], [], []
    for mode in modes:
        local = meta.copy()
        local['measurement'] = mode
        observed = clean_past.copy()
        if mode == 'noisy_quantized':
            observed += noise
        elif mode == 'biased_quantized':
            observed += bias * QUANTIZATION_RAD
        if mode != 'exact':
            observed = np.round(observed / QUANTIZATION_RAD) * QUANTIZATION_RAD
        frames.append(local)
        histories.append(observed)
        targets.append(clean_target)
    return pd.concat(frames, ignore_index=True), np.concatenate(histories), np.concatenate(targets)
