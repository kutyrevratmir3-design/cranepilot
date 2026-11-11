"""Global finite-history regression for 30-step physical residual forecasts.

Own LPV/NVAR-inspired bases, not reproductions of SMILE, G3P-SR or Koopman
control. Only training rows enter fit_path; validation selects an alpha outside
this module. Deployment adds predict_residual(...) to the supplied unchanged
physical baseline. Inputs use the existing newest-first 41-sample packet.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy import linalg

from cranepilot.forced_observer import forced_observer


LAGS = np.arange(0, 41, 5)
N_STEPS = 30
BASE_DIM = 60
FEATURE_DIMS = {"lpv_linear": 180, "nvar_quadratic": 1950}
INPUT_MODES = ("actor", "command")
SCHEMA_VERSION = 1
RMS_FLOOR = 1e-12


@dataclass(frozen=True)
class DelayLiftModel:
    basis: str
    input_mode: str
    alpha: float
    scale: np.ndarray
    coef: np.ndarray
    metadata: dict


def _array(value, shape, name):
    out = np.asarray(value, dtype=np.float64)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return out


def _options(basis, input_mode):
    if basis not in FEATURE_DIMS:
        raise ValueError(f"Unknown basis: {basis!r}")
    if input_mode not in INPUT_MODES:
        raise ValueError(f"Unknown input mode: {input_mode!r}")


def features(raw: Mapping, acc=None, input_mode="actor", basis="lpv_linear"):
    """Return unnormalized N x D features without reading any measured future.

    Only the first 41 angular-context columns, the 82 actuator-context columns,
    initial_velocity, omega, and the selected prescribed/actor future are read.
    The two trailing harmonic coefficients in angular_context are ignored: the
    finite-history forced observer recomputes q,p from past theta and x only.
    Command mode ignores acc; actor mode ignores raw['future_u'].
    """
    _options(basis, input_mode)
    angular = np.asarray(raw["angular_context"], dtype=np.float64)
    if angular.ndim != 2 or angular.shape[1] != 125:
        raise ValueError("angular_context must have shape (N,125)")
    n = len(angular)
    theta = _array(angular[:, :41], (n, 41), "past theta") * .1
    actuator = _array(raw["actuator_context"], (n, 82), "actuator_context")
    x = actuator[:, :41] * .1
    # Re-anchor even if a caller supplies an equivalent shifted x context.
    x = x - x[:, :1]
    omega = _array(raw["omega"], (n,), "omega")
    if np.any(omega <= 0):
        raise ValueError("omega must be positive")
    velocity = _array(raw["initial_velocity"], (n,), "initial_velocity")
    future = _array(acc if input_mode == "actor" else raw["future_u"],
                    (n, N_STEPS), input_mode + " future") / 2.
    if n:
        q, p, _ = forced_observer(theta, x, omega, velocity=velocity)
    else:
        q = p = np.empty(0)
    v = np.column_stack((theta[:, LAGS] / .1, x[:, LAGS] / .1,
                         actuator[:, 41 + LAGS], velocity / .2,
                         q / .1, p / .1, future))
    ell = omega[:, None] / 8. - 1.
    if basis == "lpv_linear":
        result = np.column_stack((v, v * ell, v * ell**2))
    else:
        i, j = np.triu_indices(BASE_DIM)
        quadratic = v[:, i] * v[:, j] / np.sqrt(BASE_DIM)
        result = np.column_stack((v, v * ell, quadratic))
    # No intercept or standalone ell: zero motion/command has zero residual.
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite lifted features")
    return result


def fit_path(raw, acc, target_residual, alphas=(1e-4, 1e-2, 1.),
             basis="lpv_linear", input_mode="actor", weights=None):
    """Fit one compact SVD and return {alpha: DelayLiftModel} for training rows.

    Objective: sum_i w_i * mean_j((X_scaled W - Y)[i,j]**2)
               + alpha * ||W||_F**2, where sum_i w_i = 1 and j has 30 values.
    Feature RMS uses these same TRAIN-only row weights, without centering.
    Thus, for SVD(sqrt(w) * X_scaled), the ridge denominator is s**2+30*alpha.
    Penalty applies to coefficients in this saved RMS-scaled basis.
    """
    x = features(raw, acc, input_mode=input_mode, basis=basis)
    n, d = x.shape
    if not n:
        raise ValueError("At least one training row is required")
    y = _array(target_residual, (n, N_STEPS), "target_residual")
    alpha_values = tuple(float(a) for a in alphas)
    if not alpha_values or any(not np.isfinite(a) or a <= 0 for a in alpha_values):
        raise ValueError("alphas must be finite and strictly positive")
    if weights is None:
        w = np.full(n, 1. / n)
    else:
        w = _array(weights, (n,), "weights")
        if np.any(w < 0) or not np.any(w > 0):
            raise ValueError("weights must be nonnegative with positive total")
        # Dividing first avoids overflow for otherwise valid large weights.
        w = w / w.max()
        w = w / w.sum()
    rms = np.sqrt(np.einsum("n,nd,nd->d", w, x, x))
    if not np.isfinite(rms).all():
        raise ValueError("Nonfinite training feature RMS")
    scale = np.where(rms > RMS_FLOOR, rms, 1.)
    positive = w > 0
    root_weight = np.sqrt(w[positive])[:, None]
    design = (x[positive] / scale) * root_weight
    response = y[positive] * root_weight
    try:
        u, s, vh = linalg.svd(design, full_matrices=False, lapack_driver="gesdd")
        driver = "gesdd"
    except linalg.LinAlgError:
        u, s, vh = linalg.svd(design, full_matrices=False, lapack_driver="gesvd")
        driver = "gesvd"
    projection = u.T @ response
    rank_threshold = max(design.shape) * np.finfo(np.float64).eps * s[0]
    common = {
        "schema_version": SCHEMA_VERSION,
        "training_rows": n,
        "positive_weight_rows": int(positive.sum()),
        "feature_dim": d,
        "base_dim": BASE_DIM,
        "outputs": N_STEPS,
        "lags_newest_first": LAGS.tolist(),
        "scale": "training_weighted_RMS_no_centering",
        "rms_floor": RMS_FLOOR,
        "intercept": False,
        "standalone_length_feature": False,
        "objective": "weighted_mean_windows_mean_30_outputs_plus_alpha_frobenius_squared",
        "numerical_rank": int(np.count_nonzero(s > rank_threshold)),
        "svd_driver": driver,
        "svd_largest_singular_value": float(s[0]),
    }
    models = {}
    for alpha in alpha_values:
        shrink = s / (s * s + N_STEPS * alpha)
        coef = vh.T @ (shrink[:, None] * projection)
        models[alpha] = DelayLiftModel(basis, input_mode, alpha, scale.copy(), coef,
                                       {**common, "basis": basis,
                                        "input_mode": input_mode, "alpha": alpha})
    return models


def _validate_model(model):
    _options(model.basis, model.input_mode)
    d = FEATURE_DIMS[model.basis]
    scale = _array(model.scale, (d,), "model scale")
    _array(model.coef, (d, N_STEPS), "model coefficients")
    if np.any(scale <= 0) or not np.isfinite(model.alpha) or model.alpha <= 0:
        raise ValueError("Invalid model scale or alpha")


def predict_residual(model, raw, acc=None):
    """Predict radians; caller adds the unchanged physical baseline."""
    _validate_model(model)
    x = features(raw, acc, input_mode=model.input_mode, basis=model.basis)
    return (x / model.scale) @ model.coef


def save_model(model, path):
    """Save coefficients, training scales and JSON metadata without pickle.

    Refuses overwriting retained results; parent directory must already exist.
    """
    _validate_model(model)
    metadata = {**model.metadata, "schema_version": SCHEMA_VERSION,
                "basis": model.basis, "input_mode": model.input_mode,
                "alpha": float(model.alpha)}
    with Path(path).open("xb") as out:
        np.savez_compressed(out, scale=model.scale, coef=model.coef,
                            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))


def load_model(path):
    """Restore a predictor without targets, original training rows or pickle."""
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"].item()))
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported global delay lift model version")
        model = DelayLiftModel(metadata["basis"], metadata["input_mode"],
                               float(metadata["alpha"]), saved["scale"].copy(),
                               saved["coef"].copy(), metadata)
    _validate_model(model)
    return model
