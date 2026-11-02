"""Train-only frequency calibration and causal, finite-history oscillator forecasts."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize_scalar, least_squares
from .data import DT

G = 9.81


def estimate_frequency(theta, dt=DT, start_s=35.0, end_s=90.0):
    """Development identification on an entire TRAINING record, never a test record.

    The tail is a calibration interval, not an automatic physical free-decay label.
    Frequency estimates are descriptive; actuator activity is reported separately.
    """
    y = np.asarray(theta[int(start_s / dt):int(end_s / dt):5], dtype=float)
    t = np.arange(len(y)) * dt * 5
    y = y - y.mean()
    freq = np.fft.rfftfreq(len(y), dt * 5)
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    valid = (freq >= 0.25) & (freq <= 4.5)
    f0 = freq[np.flatnonzero(valid)[np.argmax(spectrum[valid])]]

    def loss(w):
        basis = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones(len(t))])
        residual = y - basis @ np.linalg.lstsq(basis, y, rcond=None)[0]
        return float(residual @ residual)

    w0 = 2 * np.pi * f0
    result = minimize_scalar(loss, bounds=(max(0.5, w0 - 0.13), w0 + 0.13), method="bounded")
    w = float(result.x)
    return {"omega_rad_s": w, "effective_length_m": G / w**2,
            "tail_rms_rad": float(np.sqrt(np.mean(y**2))),
            "relative_fit_error": float(np.sqrt(result.fun / max(y @ y, 1e-20))),
            "calibration_start_s": start_s, "calibration_end_s": end_s}


@dataclass
class LengthCalibration:
    a: float = 1.0
    b: float = 0.0
    damping: float = 0.0
    evidence: list | None = None

    @classmethod
    def fit(cls, scenarios, train_ids, axis):
        rows = []
        for sid in train_ids:
            scenario = scenarios[sid]
            if not scenario.fixed_length:
                raise ValueError("Frequency/length calibration requires fixed TRAIN lengths")
            fit = estimate_frequency(scenario.angle(axis))
            rows.append({"sid": sid, "L_m": float(scenario.signals["L"][0]), **fit})
        length = np.array([r["L_m"] for r in rows])
        effective = np.array([r["effective_length_m"] for r in rows])
        # Both coefficients positive: explicitly avoids negative extrapolated lengths.
        initial = np.linalg.lstsq(np.column_stack([length, np.ones(len(length))]), effective, rcond=None)[0]
        result = least_squares(lambda p: p[0] * length + p[1] - effective,
                               np.maximum(initial, [0.05, 1e-5]), bounds=([0.05, 0], [4, 2]))
        return cls(float(result.x[0]), float(result.x[1]), 0.0, rows)

    def omega(self, length):
        return np.sqrt(G / np.maximum(self.a * np.asarray(length) + self.b, 0.01))


def oscillator_forecast(past, omega, horizons_s, dt=DT, damping=0.0):
    """Fit oscillator state to past angles only. No test-future smoothing.

    A*exp(-d*t/2) cos(w_d*t) + B*exp(-d*t/2) sin(w_d*t)
    is fitted at the actual past timestamps and evaluated at future horizons.
    """
    past = np.asarray(past, dtype=float)
    omega = np.broadcast_to(np.asarray(omega, dtype=float), (len(past),))
    wd = np.sqrt(np.maximum(omega**2 - damping**2 / 4, 1e-8))
    time = -np.arange(past.shape[1]) * dt
    phase = wd[:, None] * time
    decay = np.exp(-damping * time / 2)
    c, s = np.cos(phase) * decay, np.sin(phase) * decay
    gram = np.stack([np.stack([(c*c).sum(1), (c*s).sum(1)], axis=1),
                     np.stack([(c*s).sum(1), (s*s).sum(1)], axis=1)], axis=1)
    rhs = np.column_stack([(past*c).sum(1), (past*s).sum(1)])
    coef = np.linalg.solve(gram + 1e-12 * np.eye(2)[None], rhs[..., None])[..., 0]
    future = wd[:, None] * np.asarray(horizons_s)
    prediction = np.exp(-damping * np.asarray(horizons_s) / 2) * (
        coef[:, 0, None] * np.cos(future) + coef[:, 1, None] * np.sin(future))
    return prediction, coef


def rollout_sensitivities(past, omega, horizons_s, dt=DT):
    eps_w, eps_d = 1e-3, 1e-3
    center, _ = oscillator_forecast(past, omega, horizons_s, dt)
    plus, _ = oscillator_forecast(past, np.asarray(omega) + eps_w, horizons_s, dt)
    minus, _ = oscillator_forecast(past, np.asarray(omega) - eps_w, horizons_s, dt)
    damped, _ = oscillator_forecast(past, omega, horizons_s, dt, damping=eps_d)
    return np.stack([(plus-minus)/(2*eps_w), (damped-center)/eps_d], axis=2)


def phase_features(past, omega, dt=DT):
    """Dimensionless oscillator coordinates plus its causal history discrepancy.

    Every model sees at most the same recorded history; no length label inferred
    from test futures. Omega is supplied by TRAIN-only length calibration.
    """
    _, coef = oscillator_forecast(past, omega, [0.0], dt)
    omega = np.asarray(omega)
    tau = -omega[:, None] * np.arange(past.shape[1]) * dt
    reconstructed = coef[:, :1]*np.cos(tau) + coef[:, 1:]*np.sin(tau)
    residual = past - reconstructed
    # Fixed array size, causal discrepancy samples; time scale carried explicitly.
    pick = np.unique(np.linspace(0, past.shape[1]-1, min(12, past.shape[1])).astype(int))
    return np.column_stack([coef, residual[:, pick], omega * dt])


def pendulum_rhs(state, acceleration, length, damping=0.02, length_rate=0.0):
    """Prescribed trolley acceleration, taut cable, point load, angle in radians."""
    state = np.asarray(state)
    theta, rate = state[..., 0], state[..., 1]
    angular_acceleration = -(G*np.sin(theta) + np.asarray(acceleration)*np.cos(theta))/length
    angular_acceleration -= (2*length_rate/length + damping) * rate
    return np.stack([rate, angular_acceleration], axis=-1)

