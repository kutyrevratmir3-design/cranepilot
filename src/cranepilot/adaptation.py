"""A strong adaptive-physics reference, using a causal prefix only.

Uses measured trolley motion inside the prefix, not future trolley motion in
the scored forecasts. This is a separate calibration information set from the
zero-shot angle-only benchmark, not an architecture-only ablation.
"""
import numpy as np
from scipy.optimize import least_squares
from .data import DT
from .physics import G


def fit_prefix_oscillator(theta, trolley, prior_omega, budget_s, dt=DT):
    if budget_s <= 0:
        return {"omega":float(prior_omega), "damping":0.0, "status":"zero_shot", "budget_s":0.0}
    n=min(len(theta), int(np.floor(budget_s/dt))+1)
    stride=max(1,int(round(.1/dt)))
    y=np.asarray(theta[:n:stride]); x=np.asarray(trolley[:n:stride])
    step=dt*stride
    if len(y)<12 or np.std(y)<.005:
        return {"omega":float(prior_omega), "damping":0.0, "status":"insufficient_excitation", "budget_s":budget_s}
    delta2=x[2:]-2*x[1:-1]+x[:-2]
    scale=max(float(np.std(y)),.01)
    def residual(p):
        omega,damping,gain=p
        rho=np.exp(-damping*step/2)
        wd=np.sqrt(max(omega**2-damping**2/4,1e-8))
        pred=2*rho*np.cos(wd*step)*y[1:-1]-rho**2*y[:-2]+gain*delta2
        # Weak prior and zero-damping shrinkage are fixed development settings.
        return np.r_[(pred-y[2:])/scale, .02*np.log(omega/prior_omega), .01*damping]
    fit=least_squares(residual,[prior_omega,.01,-prior_omega**2/G],
                      bounds=([.75*prior_omega,0,-30],[1.25*prior_omega,.5,0]),max_nfev=200)
    return {"omega":float(fit.x[0]),"damping":float(fit.x[1]),"input_gain":float(fit.x[2]),
            "status":"fitted" if fit.success else "optimizer_failed", "budget_s":float(budget_s),
            "n_calibration_samples":len(y), "relative_residual_rms":float(np.sqrt(np.mean(fit.fun[:-2]**2)))}

