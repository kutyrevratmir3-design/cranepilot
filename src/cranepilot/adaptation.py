"""Adaptive-physics references using a causal prefix only.

Uses measured trolley motion inside the prefix, not future trolley motion in
the scored forecasts. This is a separate calibration information set from the
zero-shot angle-only benchmark, not an architecture-only ablation.
"""
import numpy as np
from scipy.optimize import least_squares
from .data import DT
from .physics import G, oscillator_forecast


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


def fit_validated_prefix(theta, prior_omega, budget_s, dt=DT, history=41,
                         train_fraction=.6, minimum_improvement=.05,
                         minimum_blocks=2):
    """Forecast-oriented physical calibration with chronological acceptance.

    Fits frequency-only and damped candidates on the first 60% of the prefix.
    Validates all horizons on the last 40%, using disjoint one-second target
    blocks. The candidate is retained without refitting after validation.
    These fixed development choices are not a guarantee of future safety.
    Only angle measurements inside the declared prefix are consulted.
    """
    prior_omega = float(prior_omega)
    if prior_omega <= 0 or dt <= 0 or history < 2:
        raise ValueError("Positive frequency/time step and at least two history samples required")
    if not 0 < train_fraction < 1 or not 0 <= minimum_improvement < 1 or minimum_blocks < 1:
        raise ValueError("Invalid chronological acceptance settings")
    result = {"omega": prior_omega, "damping": 0.0, "budget_s": float(budget_s),
              "status": "zero_shot" if budget_s <= 0 else "insufficient_prefix",
              "accepted": False, "information": "angle-only calibration prefix"}
    if budget_s <= 0:
        return result
    n = min(len(theta), int(np.floor(budget_s/dt))+1)
    y = np.asarray(theta[:n], dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("Non-finite prefix observations")
    horizons = np.array([.1, .5, 1.])
    offsets = np.rint(horizons/dt).astype(int)
    max_h = int(offsets.max())
    stride = max(1, int(round(.1/dt)))
    split = int(np.floor(train_fraction*n))
    train_origins = np.arange(history-1, split-max_h, stride)
    # Validation histories and targets are wholly after the training prefix.
    valid_origins = np.arange(split+history-1, n-max_h, stride)
    # Blocks span the latest future target, so their target intervals do not overlap.
    block_origins = np.arange(split+history-1, n-max_h, max_h)
    result.update(n_prefix_samples=n, split_sample=split,
                  n_train_origins=len(train_origins), n_validation_origins=len(valid_origins),
                  n_validation_blocks=len(block_origins))
    if len(train_origins) < 12 or len(block_origins) < minimum_blocks:
        return result
    if np.std(y[:split]) < .005:
        result["status"] = "insufficient_excitation"
        return result

    def windows(origins):
        return (y[origins[:, None]-np.arange(history)],
                y[origins[:, None]+offsets])

    train_past, train_target = windows(train_origins)
    valid_past, valid_target = windows(valid_origins)
    block_past, block_target = windows(block_origins)
    scale = max(float(np.std(train_target)), .005)
    prior_pred, _ = oscillator_forecast(valid_past, prior_omega, horizons, dt)
    prior_mae = np.mean(np.abs(prior_pred-valid_target), axis=0)
    prior_block_pred, _ = oscillator_forecast(block_past, prior_omega, horizons, dt)
    prior_block_error = np.mean(np.abs(prior_block_pred-block_target), axis=1)
    candidates = []
    for damped in (False, True):
        def residual(p):
            prediction, _ = oscillator_forecast(train_past, p[0], horizons, dt,
                                                damping=p[1] if damped else 0.)
            return ((prediction-train_target)/scale).ravel()
        initial = [prior_omega, .01] if damped else [prior_omega]
        bounds = ([.75*prior_omega, 0.], [1.25*prior_omega, .5]) if damped else (
            [.75*prior_omega], [1.25*prior_omega])
        fitted = least_squares(residual, initial, bounds=bounds, max_nfev=100)
        omega, damping = float(fitted.x[0]), float(fitted.x[1]) if damped else 0.
        prediction, _ = oscillator_forecast(valid_past, omega, horizons, dt, damping=damping)
        mae = np.mean(np.abs(prediction-valid_target), axis=0)
        block_pred, _ = oscillator_forecast(block_past, omega, horizons, dt, damping=damping)
        block_error = np.mean(np.abs(block_pred-block_target), axis=1)
        improvement = 1.-float(mae.mean()/max(prior_mae.mean(), 1e-12))
        block_win_fraction = float(np.mean(block_error < prior_block_error))
        # No scored evaluation-tail observations enter any of these conditions.
        accepted = bool(fitted.success and improvement >= minimum_improvement
                        and np.all(mae <= prior_mae+1e-12) and block_win_fraction >= .75)
        candidates.append({"model": "damped" if damped else "frequency_only",
                           "omega": omega, "damping": damping, "accepted": accepted,
                           "validation_mae_rad": mae.tolist(), "relative_improvement": improvement,
                           "block_win_fraction": block_win_fraction,
                           "optimizer_success": bool(fitted.success)})
    result.update(prior_validation_mae_rad=prior_mae.tolist(), candidates=candidates)
    eligible = [c for c in candidates if c["accepted"]]
    if eligible:
        chosen = min(eligible, key=lambda c: np.mean(c["validation_mae_rad"]))
        result.update(omega=chosen["omega"], damping=chosen["damping"], accepted=True,
                      status="accepted", chosen_model=chosen["model"])
    else:
        result["status"] = "rejected_by_validation"
    return result

