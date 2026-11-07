"""Nonlinear angular diagnostic with explicitly prescribed future acceleration.

The past trolley position is an analytic sinusoid sampled at 100 Hz. Future
acceleration is prescribed, held for 50 ms, and supplied identically to every
angular predictor. Zero command placeholders are not an actuator simulation;
this diagnostic must bypass the learned actuator. Targets are clean angles.
"""
from itertools import product
import numpy as np
import pandas as pd
from cranepilot.data import QUANTIZATION_RAD


DRIVE_FREQUENCY_HZ = .7
DRIVE_PHASE_RAD = .3


def simulate_forced(cfg, calibration, dt=.001):
    """Return metadata, history-only packet, acceleration[N,30], targets[N,30].

    Packet histories are newest first at 10 ms. Targets are at .05,...,1.50 s;
    the first acceleration acts over [0,.05). True frequency, force law and
    damping appear only in metadata, never in the operational input packet.
    An optional future_acceleration_gain acts only on simulated future truth;
    every method receives the same unscaled prescribed acceleration. This is
    an unobserved actuator mismatch, not measured future information.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('Positive integration step must divide 10 ms')
    sample_every = int(round(.01 / dt))
    if sample_every < 1 or abs(sample_every * dt - .01) > 1e-12:
        raise ValueError('Positive integration step must divide 10 ms')
    modes = cfg['measurement']
    if not modes or len(set(modes)) != len(modes) or not set(modes) <= {'exact', 'noisy_quantized'}:
        raise ValueError('Distinct exact/noisy_quantized measurement modes required')
    keys = ['length_m', 'amplitude_deg', 'phase_rad', 'omega_ratio',
            'damping_per_s', 'drive_amplitude_m', 'force_law', 'future_acceleration_gain']
    axes = [cfg.get(key, [1.]) if key == 'future_acceleration_gain' else cfg[key] for key in keys]
    rows = [dict(zip(keys, values)) for values in product(*axes)]
    meta = pd.DataFrame(rows)
    if meta.empty:
        raise ValueError('Nonempty simulation grid required')
    numbers = meta.drop(columns='force_law').to_numpy(dtype=float)
    if (not np.isfinite(numbers).all() or (meta.length_m <= 0).any()
            or (meta.omega_ratio <= 0).any() or (meta.damping_per_s < 0).any()
            or (meta.future_acceleration_gain <= 0).any()
            or (meta.amplitude_deg < 0).any() or (meta.drive_amplitude_m < 0).any()
            or not set(meta.force_law) <= {'effective', 'geometric'}):
        raise ValueError('Finite physical grid, positive lengths/frequencies and supported force laws required')
    length = meta.length_m.to_numpy(dtype=float)
    nominal = np.asarray(calibration.omega(length), dtype=float)
    if nominal.shape != length.shape or not np.isfinite(nominal).all() or (nominal <= 0).any():
        raise ValueError('Calibration must return finite positive frequencies per case')
    omega = nominal * meta.omega_ratio.to_numpy()
    damping = meta.damping_per_s.to_numpy()
    gain = np.where(meta.force_law.to_numpy() == 'effective', omega**2 / 9.81, 1 / length)
    drive = meta.drive_amplitude_m.to_numpy()
    future_gain = meta.future_acceleration_gain.to_numpy()
    drive_omega = 2 * np.pi * DRIVE_FREQUENCY_HZ
    amplitude = np.deg2rad(meta.amplitude_deg.to_numpy())
    phase = meta.phase_rad.to_numpy()
    state = np.column_stack([amplitude*np.cos(phase), -omega*amplitude*np.sin(phase)])
    meta['nominal_omega_rad_s'] = nominal
    meta['true_omega_rad_s'] = omega
    meta['force_gain_per_m'] = gain
    meta['trajectory_id'] = np.arange(len(meta))
    meta['drive_frequency_hz'] = DRIVE_FREQUENCY_HZ
    meta['drive_phase_rad'] = DRIVE_PHASE_RAD
    meta['future_acceleration_contract'] = 'prescribed_zoh_50ms_actor_bypassed'
    future_time = np.arange(30) * .05
    acceleration = -drive[:, None] * drive_omega**2 * np.sin(drive_omega*future_time + DRIVE_PHASE_RAD)

    def past_acceleration(time):
        return -drive * drive_omega**2 * np.sin(drive_omega*time + DRIVE_PHASE_RAD)

    def rhs(z, a):
        return np.column_stack([z[:, 1], -omega**2*np.sin(z[:, 0])
                                - damping*z[:, 1] - gain*a*np.cos(z[:, 0])])

    trajectory = [state[:, 0].copy()]
    past_steps = 40 * sample_every
    future_interval_steps = 5 * sample_every
    for step in range(190 * sample_every):
        if step < past_steps:
            time = -.4 + step*dt
            a0 = past_acceleration(time)
            ah = past_acceleration(time + .5*dt)
            a1 = past_acceleration(time + dt)
        else:
            # Each RK4 interval stays inside a single commanded-acceleration
            # segment. The final stage uses its left-limit acceleration, so
            # the next segment cannot leak across a discontinuity.
            index = (step - past_steps) // future_interval_steps
            a0 = ah = a1 = future_gain * acceleration[:, index]
        k1 = rhs(state, a0)
        k2 = rhs(state + .5*dt*k1, ah)
        k3 = rhs(state + .5*dt*k2, ah)
        k4 = rhs(state + dt*k3, a1)
        state += dt*(k1 + 2*k2 + 2*k3 + k4)/6
        if (step + 1) % sample_every == 0:
            trajectory.append(state[:, 0].copy())
    trajectory = np.stack(trajectory, axis=1)
    clean_past = trajectory[:, :41][:, ::-1].copy()
    clean_target = trajectory[:, 40 + 5*np.arange(1, 31)]
    past_time = -np.arange(41)*.01
    past_x = drive[:, None]*np.sin(drive_omega*past_time + DRIVE_PHASE_RAD)
    noise_std = float(cfg.get('noise_std_bins', .25))
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError('Nonnegative finite measurement noise required')
    # Pair noise across forcing amplitude/law for the same free initial state.
    # In particular, zero-drive effective/geometric cases have identical inputs.
    noise_keys = list(zip(meta.length_m, meta.amplitude_deg, meta.phase_rad,
                          meta.omega_ratio, meta.damping_per_s))
    unique = {key: i for i, key in enumerate(dict.fromkeys(noise_keys))}
    noise_ids = np.array([unique[key] for key in noise_keys])
    meta['noise_pair_id'] = noise_ids
    rng = np.random.default_rng(cfg.get('noise_seed', 8112026))
    noise = rng.normal(0, noise_std*QUANTIZATION_RAD, (len(unique), 41))[noise_ids]
    frames, histories, xs, lengths, accelerations, targets = [], [], [], [], [], []
    for mode in modes:
        local = meta.copy(); local['measurement'] = mode; frames.append(local)
        observed = clean_past.copy()
        if mode == 'noisy_quantized':
            observed = np.round((observed + noise) / QUANTIZATION_RAD) * QUANTIZATION_RAD
        histories.append(observed); xs.append(past_x); lengths.append(length)
        accelerations.append(acceleration); targets.append(clean_target)
    theta = np.concatenate(histories)
    packet = {'theta': theta, 'x': np.concatenate(xs), 'reference': np.zeros_like(theta),
              'length_m': np.concatenate(lengths), 'future_reference': np.zeros((len(theta), 30))}
    return pd.concat(frames, ignore_index=True), packet, np.concatenate(accelerations), np.concatenate(targets)
