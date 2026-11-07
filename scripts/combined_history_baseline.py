"""Fixed causal adaptive baseline for autonomous angular-motion diagnostics.

This is a new baseline, not a reproduction of a published method. It assumes
unforced motion and must not be used as the forced real-data comparator. Only
the 41 measured past angles and a nominal train-calibrated frequency are input.
The frequency penalty is a fixed shrinkage heuristic in radian-squared units,
not a calibrated probabilistic prior. No intercept absorbs sensor bias.
"""
import numpy as np

from cranepilot.data import QUANTIZATION_RAD


FREQUENCY_RATIOS = (.75, .85, 1., 1.15, 1.25)
DAMPING_PER_S = (0., .1, .2, .4)


def adaptive_free_forecast(theta, omega):
    """Return (N,30) forecasts and a dictionary of per-packet selection arrays.

    ``theta`` is radians, shape (N,41), newest first at 10 ms spacing. ``omega``
    is a positive nominal angular frequency in rad/s, scalar or shape (N,).
    Forecasts are at .05, .10, ..., 1.50 seconds after the latest measurement.
    Damping denotes the coefficient of q' in q'' + damping*q' + omega**2*q = 0.
    Grid candidates that are not underdamped are omitted for that packet;
    zero damping remains available for every positive nominal frequency.
    """
    theta = np.asarray(theta, dtype=float)
    nominal = np.asarray(omega, dtype=float)
    if theta.ndim != 2 or theta.shape[1] != 41 or not len(theta):
        raise ValueError('Nonempty (N,41) newest-first angle history required')
    if nominal.ndim > 1:
        raise ValueError('Frequency must be scalar or shape (N,)')
    try:
        nominal = np.broadcast_to(nominal, (len(theta),))
    except ValueError as error:
        raise ValueError('Frequency must broadcast to (N,)') from error
    if not np.isfinite(theta).all() or not np.isfinite(nominal).all() or (nominal <= 0).any():
        raise ValueError('Finite angle history and finite positive frequency required')

    past_time = -np.arange(41) * .01
    future_time = np.arange(1, 31) * .05
    best = np.full(len(theta), np.inf)
    forecast = np.empty((len(theta), 30))
    metadata = {name: np.empty(len(theta)) for name in (
        'frequency_ratio', 'omega_rad_s', 'damping_per_s',
        'history_mse_rad2', 'penalized_score_rad2')}

    for ratio in FREQUENCY_RATIOS:
        frequency = ratio * nominal
        penalty = QUANTIZATION_RAD**2 / 41 * ((ratio - 1) / .15)**2
        for damping in DAMPING_PER_S:
            valid = frequency**2 > damping**2 / 4
            if not valid.any():
                continue
            indices = np.flatnonzero(valid)
            wd = np.sqrt(frequency[valid]**2 - damping**2 / 4)
            phase = wd[:, None] * past_time
            basis = np.stack((np.cos(phase), np.sin(phase)), axis=-1)
            basis *= np.exp(-damping * past_time / 2)[None, :, None]
            # Batched pseudoinverse keeps the fit defined for short, nearly
            # collinear histories without a target-dependent regularizer.
            coef = (np.linalg.pinv(basis) @ theta[valid, :, None])[..., 0]
            fitted = np.einsum('nti,ni->nt', basis, coef)
            mse = np.mean((fitted - theta[valid])**2, axis=1)
            score = mse + penalty
            improved = score < best[valid]
            if not improved.any():
                continue
            take = indices[improved]
            phase = wd[improved, None] * future_time
            future_basis = np.stack((np.cos(phase), np.sin(phase)), axis=-1)
            future_basis *= np.exp(-damping * future_time / 2)[None, :, None]
            forecast[take] = np.einsum('nti,ni->nt', future_basis, coef[improved])
            best[take] = score[improved]
            metadata['frequency_ratio'][take] = ratio
            metadata['omega_rad_s'][take] = frequency[take]
            metadata['damping_per_s'][take] = damping
            metadata['history_mse_rad2'][take] = mse[improved]
            metadata['penalized_score_rad2'][take] = score[improved]
    if not np.isfinite(forecast).all() or not np.isfinite(best).all():
        raise ValueError('Numerically invalid harmonic extrapolation')
    return forecast, metadata
