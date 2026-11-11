"""Causal damped forced-oscillator estimation on 0.4 s of measured history.

The convention matches the existing effective-length physics:
    theta'' + damping*theta' + omega**2*theta = -(omega**2/g)*x''.
No geometric 1/length substitution or second numerical derivative is used.
Past trolley positions are piecewise linear. A transformed state z=theta+k*x,
k=omega**2/g, allows exact interval propagation with first slopes only. Future
acceleration is the common prescribed/model-predicted input, held for 50 ms.
The fixed shrinkage score is a heuristic, not a calibrated Bayesian posterior.
"""
import numpy as np
from cranepilot.data import QUANTIZATION_RAD

FREQUENCY_RATIOS = (.75, .85, 1., 1.15, 1.25)
DAMPING_PER_S = (0., .1, .2, .4)
GRAVITY = 9.81


def _vector(value, n, name):
    value = np.asarray(value, dtype=float)
    if value.ndim > 1:
        raise ValueError(f'{name} must be scalar or shape (N,)')
    try:
        result = np.broadcast_to(value, (n,))
    except ValueError as error:
        raise ValueError(f'{name} must broadcast to (N,)') from error
    if not np.isfinite(result).all():
        raise ValueError(f'Finite {name} required')
    return result


def _parameters(omega, damping, n):
    omega = _vector(omega, n, 'omega')
    damping = _vector(damping, n, 'damping')
    if (omega <= 0).any() or (damping < 0).any() or (omega**2 <= damping**2/4).any():
        raise ValueError('Positive underdamped frequency and nonnegative damping required')
    return omega, damping


def _history(theta, x):
    theta, x = np.asarray(theta, dtype=float), np.asarray(x, dtype=float)
    if theta.ndim != 2 or theta.shape[1] != 41 or not len(theta) or x.shape != theta.shape:
        raise ValueError('Matching nonempty (N,41) newest-first theta/x required')
    if not np.isfinite(theta).all() or not np.isfinite(x).all():
        raise ValueError('Finite observed histories required')
    return theta, x


def _transition(omega, damping, dt):
    half = damping/2
    wd = np.sqrt(omega**2-half**2)
    decay = np.exp(-half*dt)
    sine = np.sin(wd*dt)/wd
    cosine = np.cos(wd*dt)
    m01 = decay*sine
    return decay*(cosine+half*sine), m01, -omega**2*m01, decay*(cosine-half*sine)


def forced_fit(theta, x, omega, damping, velocity=None):
    """Return fitted origin angle, angular velocity, newest-first residual.

    theta is radians, x metres, omega rad/s, damping 1/s. Endpoint trolley
    velocity is the same causal 21-sample OLS estimate as the original observer,
    unless the caller provides that estimate explicitly. Angular velocity is
    rad/s; unlike the legacy observer's p it is not divided by omega.
    """
    theta, x = _history(theta, x)
    n = len(theta); omega, damping = _parameters(omega, damping, n)
    theta, x = theta[:, ::-1], x[:, ::-1]
    x = x-x[:, :1]
    dt = .01; kappa = omega**2/GRAVITY
    m00, m01, m10, m11 = _transition(omega, damping, dt)
    z, zdot = np.zeros(n), np.zeros(n)
    part = np.zeros_like(theta)
    for j in range(40):
        slope = (x[:, j+1]-x[:, j])/dt
        local_q = z-kappa*x[:, j]
        local_v = zdot-kappa*slope
        z, zdot = (kappa*x[:, j+1]+m00*local_q+m01*local_v,
                   kappa*slope+m10*local_q+m11*local_v)
        part[:, j+1] = z-kappa*x[:, j+1]
    time = np.arange(41)*dt
    half = damping/2; wd = np.sqrt(omega**2-half**2)
    phase = wd[:, None]*time
    cosine, sine = np.cos(phase), np.sin(phase)
    decay = np.exp(-half[:, None]*time)
    basis = decay[:, :, None]*np.stack((cosine, sine), axis=-1)
    gram = np.einsum('nti,ntj->nij', basis, basis)
    rhs = np.einsum('nti,nt->ni', basis, theta-part)
    coefficients = np.linalg.solve(gram+1e-12*np.eye(2)[None], rhs[..., None])[..., 0]
    fitted = np.einsum('nti,ni->nt', basis, coefficients)+part
    if velocity is None:
        t = np.arange(21)*dt
        velocity = x[:, -21:]@((t-t.mean())/np.sum((t-t.mean())**2))
    velocity = _vector(velocity, n, 'velocity')
    derivative = decay[:, -1, None]*np.column_stack([
        -half*cosine[:, -1]-wd*sine[:, -1],
        -half*sine[:, -1]+wd*cosine[:, -1]])
    qdot = np.sum(derivative*coefficients, axis=1)+zdot-kappa*velocity
    return fitted[:, -1], qdot, (theta-fitted)[:, ::-1].copy()


def damped_rollout(q, qdot, omega, damping, acceleration):
    """Exact damped response at 0.05,...,1.50 s for 30 held accelerations."""
    acceleration = np.asarray(acceleration, dtype=float)
    if acceleration.ndim != 2 or acceleration.shape[1] != 30 or not len(acceleration):
        raise ValueError('Nonempty acceleration shape (N,30) required')
    if not np.isfinite(acceleration).all():
        raise ValueError('Finite acceleration required')
    n = len(acceleration)
    omega, damping = _parameters(omega, damping, n)
    q = _vector(q, n, 'q').copy(); qdot = _vector(qdot, n, 'qdot').copy()
    m00, m01, m10, m11 = _transition(omega, damping, .05)
    result = []
    for a in acceleration.T:
        equilibrium = -a/GRAVITY
        shifted = q-equilibrium
        q, qdot = (equilibrium+m00*shifted+m01*qdot,
                   m10*shifted+m11*qdot)
        result.append(q.copy())
    forecast = np.stack(result, axis=1)
    if not np.isfinite(forecast).all():
        raise ValueError('Numerically invalid oscillator extrapolation')
    return forecast


def adaptive_forced_forecast(theta, x, omega, acceleration, velocity=None,
                             ratios=FREQUENCY_RATIOS, dampings=DAMPING_PER_S, penalty=1.):
    """Select parameters from measured history only; then forecast common input.

    The score is past MSE + penalty*Q**2/41*((ratio-1)/.15)**2.
    Metadata has one array per packet, including history_rms_rad, chosen omega,
    damping, initial angle/angular velocity, penalized score, second-best score
    gap, candidate count and whether the selected ratio is a grid boundary.
    Invalid (non-underdamped) grid candidates are skipped per packet; every
    packet must have at least one valid candidate. No intercept absorbs bias.
    """
    theta, x = _history(theta, x); n = len(theta)
    nominal = _vector(omega, n, 'omega')
    if (nominal <= 0).any(): raise ValueError('Positive nominal omega required')
    ratios, dampings = np.asarray(ratios, dtype=float), np.asarray(dampings, dtype=float)
    if (ratios.ndim != 1 or dampings.ndim != 1 or not len(ratios) or not len(dampings)
            or not np.isfinite(ratios).all() or not np.isfinite(dampings).all()
            or (ratios <= 0).any() or (dampings < 0).any()
            or not np.isfinite(penalty) or penalty < 0):
        raise ValueError('Finite positive frequency grid, nonnegative damping and penalty required')
    velocity = None if velocity is None else _vector(velocity, n, 'velocity')
    best = np.full(n, np.inf)
    second = np.full(n, np.inf)
    count = np.zeros(n, dtype=int)
    names = ('frequency_ratio', 'omega_rad_s', 'damping_per_s', 'history_mse_rad2',
             'history_rms_rad', 'penalized_score_rad2', 'initial_q', 'initial_qdot')
    metadata = {name: np.empty(n) for name in names}
    for ratio in ratios:
        frequency = ratio*nominal
        shrinkage = penalty*QUANTIZATION_RAD**2/41*((ratio-1)/.15)**2
        for damping in dampings:
            valid = frequency**2 > damping**2/4
            if not valid.any(): continue
            indices = np.flatnonzero(valid)
            q, qdot, residual = forced_fit(theta[valid], x[valid], frequency[valid], damping,
                                          None if velocity is None else velocity[valid])
            mse = np.mean(residual**2, axis=1); score = mse+shrinkage
            improved = score < best[valid]
            second[valid] = np.minimum(second[valid], np.where(improved, best[valid], score))
            count[valid] += 1
            if not improved.any(): continue
            take = indices[improved]; best[take] = score[improved]
            values = (np.full(len(take), ratio), frequency[take], np.full(len(take), damping),
                      mse[improved], np.sqrt(mse[improved]), score[improved], q[improved], qdot[improved])
            for name,value in zip(names,values): metadata[name][take] = value
    if not np.isfinite(best).all():
        raise ValueError('Each packet needs at least one underdamped grid candidate')
    # A single-candidate grid has no alternative: record zero gap and count=1.
    metadata['second_best_score_rad2'] = np.where(count > 1, second, best)
    metadata['score_gap_rad2'] = metadata['second_best_score_rad2']-best
    metadata['candidate_count'] = count
    metadata['ratio_boundary_hit'] = ((metadata['frequency_ratio'] == ratios.min())
                                      | (metadata['frequency_ratio'] == ratios.max()))
    metadata['history_rms'] = metadata['history_rms_rad'].copy()
    prediction = damped_rollout(metadata['initial_q'], metadata['initial_qdot'],
                                metadata['omega_rad_s'], metadata['damping_per_s'], acceleration)
    return prediction, metadata
