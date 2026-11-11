"""Select forced-oscillator parameters by withheld observed-angle predictions.

Every packet still contains only the original 41 past samples. A chronological
prefix fits two homogeneous coefficients; its remaining angles are withheld.
The later trolley positions are already observed at the actual forecast origin
and drive the history check. This is conditional historical validation, not a
claim to have forecast those trolley positions. The final selected oscillator
is refitted to all 41 samples before predicting future prescribed acceleration.
"""
import numpy as np
from cranepilot.data import QUANTIZATION_RAD
from scripts.forced_adaptive_physics import (
    FREQUENCY_RATIOS, DAMPING_PER_S, GRAVITY, _history, _parameters, _transition,
    _vector, forced_fit, damped_rollout,
)


def _prefixes(values):
    values = tuple(values)
    if (not values or len(set(values)) != len(values)
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                   or not 3 <= v < 41 for v in values)):
        raise ValueError('Distinct integer prefix counts from 3 through 40 required')
    return values


def _history_terms(x, omega, damping):
    """Particular response and homogeneous basis, chronological and angle-free."""
    x = x[:, ::-1]
    x = x - x[:, :1]
    n = len(x)
    kappa = omega**2 / GRAVITY
    m00, m01, m10, m11 = _transition(omega, damping, .01)
    z, zdot = np.zeros(n), np.zeros(n)
    part = np.zeros_like(x)
    for j in range(40):
        slope = (x[:, j+1] - x[:, j]) / .01
        q = z - kappa*x[:, j]
        v = zdot - kappa*slope
        z, zdot = (kappa*x[:, j+1] + m00*q + m01*v,
                   kappa*slope + m10*q + m11*v)
        part[:, j+1] = z - kappa*x[:, j+1]
    t = np.arange(41)*.01
    wd = np.sqrt(omega**2 - damping**2/4)
    phase = wd[:, None]*t
    basis = np.exp(-damping[:, None]*t/2)[:, :, None] * np.stack(
        (np.cos(phase), np.sin(phase)), axis=-1)
    return part, basis


def _predict_prefix(chronological_theta, part, basis, prefix_count):
    b = basis[:, :prefix_count]
    observed = chronological_theta[:, :prefix_count] - part[:, :prefix_count]
    gram = np.einsum('nti,ntj->nij', b, b)
    rhs = np.einsum('nti,nt->ni', b, observed)
    coef = np.linalg.solve(gram + 1e-12*np.eye(2)[None], rhs[..., None])[..., 0]
    prediction = np.einsum('nti,ni->nt', basis, coef) + part
    return prediction, coef


def history_prediction(theta, x, omega, damping, prefix_count):
    """Return chronological predicted angles and prefix-fit metadata.

    Only the first chronological ``prefix_count`` angle samples enter the fit.
    A changed withheld angle cannot change any returned prediction/coefficient.
    Later observed x may change later predictions, but cannot alter earlier
    particular response, earlier predictions or fitted prefix coefficients.
    """
    theta, x = _history(theta, x)
    prefix_count = _prefixes((prefix_count,))[0]
    omega, damping = _parameters(omega, damping, len(theta))
    part, basis = _history_terms(x, omega, damping)
    prediction, coef = _predict_prefix(theta[:, ::-1], part, basis, prefix_count)
    return prediction, {'coefficients': coef, 'prefix_count': np.full(len(theta), prefix_count)}


def _backtest_mse(theta, x, omega, damping, prefixes):
    part, basis = _history_terms(x, omega, damping)
    chronological = theta[:, ::-1]
    errors = []
    for prefix in prefixes:
        prediction, _ = _predict_prefix(chronological, part, basis, prefix)
        errors.append(np.mean((prediction[:, prefix:] - chronological[:, prefix:])**2, axis=1))
    # Each split has equal weight, not each reused held-back observation.
    return np.mean(errors, axis=0)


def backtest_forced_forecast(theta, x, omega, acceleration, velocity=None,
                            ratios=FREQUENCY_RATIOS, dampings=DAMPING_PER_S,
                            penalty=1., prefixes=(21,), guard_fraction=0.,
                            guard_absolute_bins2=0.):
    """Choose parameters from historical holdouts, refit, forecast, and diagnose.

    Score = mean split holdout MSE + penalty*Q²/41*((ratio-1)/.15)².
    If either guard is nonzero, an adaptive candidate must strictly improve the
    nominal score and improve by at least both guard_fraction*nominal_score and
    guard_absolute_bins2*Q²/41. Otherwise use ratio=1,damping=0. The nominal
    comparator exists even if a custom candidate grid omits it. With no guard,
    the best grid candidate is retained, preserving an unguarded comparison.
    Future acceleration and supplied causal endpoint velocity do not enter
    parameter selection. All-state refitting uses frozen ``forced_fit``.
    """
    theta, x = _history(theta, x)
    n = len(theta)
    nominal = _vector(omega, n, 'omega')
    if (nominal <= 0).any(): raise ValueError('Positive nominal omega required')
    acceleration = np.asarray(acceleration, dtype=float)
    if acceleration.shape != (n, 30) or not np.isfinite(acceleration).all():
        raise ValueError('Finite acceleration with shape (N,30) required')
    prefixes = _prefixes(prefixes)
    ratios, dampings = np.asarray(ratios, dtype=float), np.asarray(dampings, dtype=float)
    if (ratios.ndim != 1 or dampings.ndim != 1 or not len(ratios) or not len(dampings)
            or not np.isfinite(ratios).all() or not np.isfinite(dampings).all()
            or (ratios <= 0).any() or (dampings < 0).any()
            or len(np.unique(ratios)) != len(ratios) or len(np.unique(dampings)) != len(dampings)
            or not np.isfinite([penalty, guard_fraction, guard_absolute_bins2]).all()
            or min(penalty, guard_fraction, guard_absolute_bins2) < 0):
        raise ValueError('Distinct finite positive frequency grid, nonnegative damping/penalty/guards required')
    velocity = None if velocity is None else _vector(velocity, n, 'velocity')
    nominal_mse = _backtest_mse(theta, x, nominal, np.zeros(n), prefixes)
    best, second = np.full(n, np.inf), np.full(n, np.inf)
    count = np.zeros(n, dtype=int)
    chosen_ratio, chosen_damping, chosen_mse = np.empty(n), np.empty(n), np.empty(n)
    for ratio in ratios:
        frequency = nominal * ratio
        prior = penalty*QUANTIZATION_RAD**2/41*((ratio-1)/.15)**2
        for damping in dampings:
            valid = frequency**2 > damping**2/4
            if not valid.any(): continue
            positions = np.flatnonzero(valid)
            mse = _backtest_mse(theta[valid], x[valid], frequency[valid],
                                np.full(valid.sum(), damping), prefixes)
            score = mse + prior
            improved = score < best[valid]
            second[valid] = np.minimum(second[valid], np.where(improved, best[valid], score))
            count[valid] += 1
            take = positions[improved]
            best[take] = score[improved]
            chosen_ratio[take] = ratio
            chosen_damping[take] = damping
            chosen_mse[take] = mse[improved]
    if not np.isfinite(best).all(): raise ValueError('Every packet needs an underdamped grid candidate')
    ratio_before = chosen_ratio.copy()
    damping_before = chosen_damping.copy()
    nonnominal = (chosen_ratio != 1.) | (chosen_damping != 0.)
    improvement = nominal_mse - best
    guard_enabled = guard_fraction > 0 or guard_absolute_bins2 > 0
    passes_guard = np.ones(n, dtype=bool)
    if guard_enabled:
        passes_guard = ((improvement > 0)
                        & (improvement >= guard_fraction*nominal_mse)
                        & (improvement >= guard_absolute_bins2*QUANTIZATION_RAD**2/41))
        chosen_ratio = np.where(passes_guard, chosen_ratio, 1.)
        chosen_damping = np.where(passes_guard, chosen_damping, 0.)
        chosen_mse = np.where(passes_guard, chosen_mse, nominal_mse)
    selected_score = np.where(passes_guard, best, nominal_mse)
    selected_omega = nominal * chosen_ratio
    q, qdot, residual = forced_fit(theta, x, selected_omega, chosen_damping, velocity)
    forecast = damped_rollout(q, qdot, selected_omega, chosen_damping, acceleration)
    metadata = {
        'frequency_ratio': chosen_ratio, 'omega_rad_s': selected_omega,
        'damping_per_s': chosen_damping, 'backtest_mse_rad2': chosen_mse,
        'nominal_backtest_mse_rad2': nominal_mse, 'penalized_score_rad2': selected_score,
        'best_score_rad2': best, 'second_best_score_rad2': np.where(count > 1, second, best),
        'score_gap_rad2': np.where(count > 1, second, best) - best,
        'chosen_before_guard_ratio': ratio_before, 'chosen_before_guard_damping': damping_before,
        'improvement_before_guard_rad2': improvement,
        'accepted_adaptation': nonnominal & passes_guard,
        'guard_rejected_candidate': nonnominal & ~passes_guard,
        'candidate_count': count, 'history_mse_rad2': np.mean(residual**2, axis=1),
        'history_rms_rad': np.sqrt(np.mean(residual**2, axis=1)),
        'full_history_rms': np.sqrt(np.mean(residual**2, axis=1)),
        'initial_q': q, 'initial_qdot': qdot,
        'ratio_boundary_hit': (chosen_ratio == ratios.min()) | (chosen_ratio == ratios.max()),
    }
    if not all(np.isfinite(v).all() for v in metadata.values()):
        raise ValueError('Nonfinite historical-selection diagnostic')
    return forecast, metadata
