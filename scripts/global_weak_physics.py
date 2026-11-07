"""Train-record weak-form identification with fixed parameters at inference.

Equation: theta'' + c*theta' + kscale*omega0(L)^2*theta
          = -bscale*omega0(L)^2/g*x''.
Restoring stiffness and acceleration coupling are separate fitted quantities.
Only supplied training records enter fitting. Inference refits the two current
state coefficients from 41 causal samples; dynamics parameters remain frozen.
"""
from collections.abc import Mapping
import numpy as np
from scipy.optimize import lsq_linear
from scripts.forced_adaptive_physics import forced_fit, damped_rollout, _history, _vector

GRAVITY = 9.81
DURATIONS_S = (1., 2., 4.)
LOWER = np.array([0., .5, .2])
UPPER = np.array([2., 1.5, 2.])
PRIOR = np.array([0., 1., 1.])


def _weak_kernel(duration, dt):
    """Trapezoidal weights for phi, phi' and phi'' on compact support."""
    intervals = int(round(duration/dt))
    if intervals < 4 or abs(intervals*dt-duration) > 1e-10:
        raise ValueError('Support durations must be resolved by the sample interval')
    u = np.linspace(-1., 1., intervals+1)
    phi = (1-u*u)**4
    first = -16*u*(1-u*u)**3/duration
    second = 32*(1-u*u)**2*(7*u*u-1)/duration**2
    quadrature = np.full(intervals+1, dt)
    quadrature[[0,-1]] *= .5
    noise_norm = np.sqrt(np.sum(quadrature*second**2))
    return quadrature*phi/noise_norm, quadrature*first/noise_norm, quadrature*second/noise_norm


def fit_global(scenarios, calibration, ridge=0.):
    """Fit one c/kscale/bscale on complete supplied TRAIN scenarios.

    Fixed supports are 1/2/4 s, centres spaced half a support apart. Rows are
    normalized by the L2 norm of phi'', then weighted equally by record and
    support duration. The data loss is divided by its global target RMS before
    adding ridge*||parameters-[0,1,1]||². No hyperparameter search occurs here.
    ``scenarios`` may be an iterable of Scenario objects or a mapping of them.
    Returned diagnostics and parameters are JSON-serializable Python values.
    """
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError('Finite nonnegative ridge required')
    items = list(scenarios.values()) if isinstance(scenarios, Mapping) else list(scenarios)
    if not items or len({s.sid for s in items}) != len(items):
        raise ValueError('Distinct nonempty training scenarios required')
    evidence = getattr(calibration, 'evidence', None)
    if evidence is not None and sorted(r['sid'] for r in evidence) != sorted(s.sid for s in items):
        raise ValueError('Calibration evidence must use exactly these training scenarios')
    matrices, labels, weights, record_rows = [], [], [], []
    for scenario in sorted(items, key=lambda s:s.sid):
        t = np.asarray(scenario.time, dtype=float)
        theta = np.asarray(scenario.signals['theta'], dtype=float)
        x = np.asarray(scenario.signals['x'], dtype=float)
        length = np.asarray(scenario.signals['L'], dtype=float)
        if (t.ndim != 1 or len(t)<5 or theta.shape!=t.shape or x.shape!=t.shape or length.shape!=t.shape
                or not all(np.isfinite(v).all() for v in [t,theta,x,length])
                or (length<=0).any() or np.ptp(length)>=1e-6):
            raise ValueError('Finite fixed-length training theta/x/time/L records required')
        dt = float(np.median(np.diff(t)))
        if dt<=0 or not np.allclose(np.diff(t),dt,rtol=1e-9,atol=1e-12) or t[-1]-t[0] < max(DURATIONS_S):
            raise ValueError('Uniform training grid covering all 1/2/4-second supports required')
        omega = float(np.asarray(calibration.omega(float(length[0]))))
        if not np.isfinite(omega) or omega<=0: raise ValueError('Positive calibrated frequency required')
        for duration in DURATIONS_S:
            w0,w1,w2 = _weak_kernel(duration,dt)
            size = len(w0); stride = max(1,int(round(duration/(2*dt))))
            starts = np.arange(0,len(t)-size+1,stride)
            if not len(starts): raise ValueError('Insufficient training support')
            indices = starts[:,None]+np.arange(size)
            q = theta[indices]
            # Translation removes quadrature residue from an arbitrary x origin.
            position = x[indices]-x[indices][:,:1]
            matrix = np.column_stack([-q@w1, omega**2*(q@w0), omega**2/GRAVITY*(position@w2)])
            target = -(q@w2)
            matrices.append(matrix); labels.append(target)
            weights.append(np.full(len(starts),1/np.sqrt(len(items)*len(DURATIONS_S)*len(starts))))
            record_rows.append({'sid':int(scenario.sid),'duration_s':duration,'rows':int(len(starts)),
                                'length_m':float(length[0]),'omega0_rad_s':omega,'dt_s':dt})
    matrix = np.concatenate(matrices); target = np.concatenate(labels); weight = np.concatenate(weights)
    weighted = matrix*weight[:,None]; rhs = target*weight
    informative = bool(np.linalg.norm(weighted)>1e-14)
    scale = max(float(np.linalg.norm(rhs)),float(np.linalg.norm(weighted@PRIOR))*1e-12,1e-12)
    design, response = weighted/scale, rhs/scale
    singular = np.linalg.svd(design,compute_uv=False)
    rank = int(np.linalg.matrix_rank(design))
    if not informative:
        coefficients = PRIOR.copy(); status = 0; message = 'No excitation; retained nominal prior'; active_mask = np.zeros(3,dtype=int)
    else:
        objective, objective_rhs = design, response
        if ridge>0:
            objective = np.vstack([design,np.sqrt(ridge)*np.eye(3)])
            objective_rhs = np.concatenate([response,np.sqrt(ridge)*PRIOR])
        result = lsq_linear(objective,objective_rhs,bounds=(LOWER,UPPER),tol=1e-12,max_iter=500,lsmr_tol='auto')
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError(f'Global weak fit did not converge: {result.message}')
        coefficients = result.x; status = int(result.status); message = str(result.message); active_mask = result.active_mask.astype(int)
    residual = design@coefficients-response
    return {'c':float(coefficients[0]),'kscale':float(coefficients[1]),'bscale':float(coefficients[2]),
            'ridge':float(ridge),'training_sids':[int(s.sid) for s in sorted(items,key=lambda s:s.sid)],
            'durations_s':list(DURATIONS_S),'support_stride_fraction':.5,'rows':int(len(target)),
            'row_details':record_rows,'data_target_scale':scale,'data_rank':rank,'identified_full_rank':bool(informative and rank==3),
            'singular_values':singular.tolist(),'condition_number':float(singular[0]/singular[-1]) if rank==3 else None,
            'scaled_data_mse':float(residual@residual),'ridge_penalty':float(ridge*np.sum((coefficients-PRIOR)**2)),
            'active_bounds':active_mask.tolist(),'solver_status':status,'solver_message':message,
            'equation':'theta_ddot+c*theta_dot+kscale*omega0^2*theta=-bscale*omega0^2/g*x_ddot',
            'weighting':'phi_ddot L2 normalization; equal record-duration loss; global target RMS; ridge toward[0,1,1]',
            'parameters_adapted_at_inference':False}


def _general_transition(kappa, damping, time):
    """Stable 2x2 oscillator exponential for under/critical/overdamped states."""
    half = damping/2
    delta = kappa-half**2
    root = np.sqrt(np.abs(delta))
    decay = np.exp(-half*time)
    if np.ndim(time):
        root = root[:,None]; delta = delta[:,None]
    argument = root*time
    denominator = np.where(root>1e-14,root,1.)
    s = np.where(delta>1e-14,np.sin(argument)/denominator,
                 np.where(delta < -1e-14,np.sinh(argument)/denominator,time))
    c = np.where(delta>1e-14,np.cos(argument),np.where(delta < -1e-14,np.cosh(argument),1.))
    s = decay*s; c = decay*c
    k = kappa[:,None] if np.ndim(time) else kappa
    return c+half*s,s,-k*s,c-half*s


def _general_forecast(theta,x,kappa,b,damping,acceleration,velocity):
    """General damping fallback; state fitting uses stable transition columns."""
    theta,x = theta[:,::-1],x[:,::-1]
    x = x-x[:,:1]
    m00,m01,m10,m11 = _general_transition(kappa,damping,.01)
    z,zdot = np.zeros(len(theta)),np.zeros(len(theta)); part=np.zeros_like(theta)
    for j in range(40):
        slope=(x[:,j+1]-x[:,j])/.01
        q=z-b*x[:,j];v=zdot-b*slope
        z,zdot=b*x[:,j+1]+m00*q+m01*v,b*slope+m10*q+m11*v
        part[:,j+1]=z-b*x[:,j+1]
    time=np.arange(41)*.01
    a,c,d,e=_general_transition(kappa,damping,time)
    basis=np.stack([a,c],axis=-1)
    gram=np.einsum('nti,ntj->nij',basis,basis)
    rhs=np.einsum('nti,nt->ni',basis,theta-part)
    coefficients=np.linalg.solve(gram+1e-12*np.eye(2)[None],rhs[...,None])[...,0]
    q=np.sum(basis[:,-1]*coefficients,axis=1)+part[:,-1]
    if velocity is None:
        t=np.arange(21)*.01;velocity=x[:,-21:]@((t-t.mean())/np.sum((t-t.mean())**2))
    qdot=d[:,-1]*coefficients[:,0]+e[:,-1]*coefficients[:,1]+zdot-b*velocity
    m00,m01,m10,m11=_general_transition(kappa,damping,.05);out=[]
    for acceleration_now in acceleration.T:
        equilibrium=-b/kappa*acceleration_now;shifted=q-equilibrium
        q,qdot=equilibrium+m00*shifted+m01*qdot,m10*shifted+m11*qdot
        out.append(q.copy())
    return np.stack(out,axis=1)


def forecast_global(model,theta,x,omega,acceleration,velocity=None):
    """Forecast30 steps from41 samples; only current state is refitted."""
    theta,x=_history(theta,x);n=len(theta)
    omega=_vector(omega,n,'omega');velocity=None if velocity is None else _vector(velocity,n,'velocity')
    c,kscale,bscale=map(float,[model['c'],model['kscale'],model['bscale']])
    parameters=np.array([c,kscale,bscale])
    if not np.isfinite(parameters).all() or np.any(parameters<LOWER-1e-10) or np.any(parameters>UPPER+1e-10) or (omega<=0).any():
        raise ValueError('Model parameters outside declared physical bounds')
    acceleration=np.asarray(acceleration,dtype=float)
    if acceleration.shape!=(n,30) or not np.isfinite(acceleration).all():raise ValueError('Finite(N,30) acceleration required')
    frequency=omega*np.sqrt(kscale);coupling_ratio=bscale/kscale
    if np.all(frequency**2>c*c/4):
        q,qdot,_=forced_fit(theta,coupling_ratio*x,frequency,c,
                            None if velocity is None else coupling_ratio*velocity)
        result=damped_rollout(q,qdot,frequency,c,coupling_ratio*acceleration)
    else:
        result=_general_forecast(theta,x,frequency**2,bscale*omega**2/GRAVITY,c,acceleration,velocity)
    if not np.isfinite(result).all():raise ValueError('Nonfinite global physical forecast')
    return result
