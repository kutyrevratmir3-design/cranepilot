"""Fixed causal discrepancy gate; neither probability nor calibrated uncertainty."""
import numpy as np
from cranepilot.data import QUANTIZATION_RAD


def blend(neural, physical, observer_residual_rms, tau_bins):
    neural, physical = np.asarray(neural), np.asarray(physical)
    rms = np.asarray(observer_residual_rms)
    if tau_bins <= 0 or neural.shape != physical.shape or rms.shape != neural.shape[:1]:
        raise ValueError('Positive scale and matching (N,T) forecasts / (N,) RMS required')
    if not np.isfinite(rms).all() or (rms < 0).any():
        raise ValueError('Finite nonnegative past residual RMS required')
    weight = -np.expm1(-(rms / (tau_bins * QUANTIZATION_RAD))**2)
    return physical + weight[:, None] * (neural - physical), weight
