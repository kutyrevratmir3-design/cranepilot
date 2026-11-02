import numpy as np


def regression_metrics(y, prediction):
    error = np.rad2deg(np.asarray(prediction) - np.asarray(y))
    if not np.isfinite(error).all():
        raise ValueError("Nonfinite predictions must be reported as a model failure")
    return {"mae_deg": float(np.mean(np.abs(error))),
            "rmse_deg": float(np.sqrt(np.mean(error**2))),
            "p95_ae_deg": float(np.quantile(np.abs(error), 0.95)),
            "max_ae_deg": float(np.max(np.abs(error))), "n": len(error)}


def paired_bootstrap(difference, seed=0, n_bootstrap=5000):
    """Resample INDEPENDENT scenario/episode differences, never time windows."""
    difference = np.asarray(difference, dtype=float)
    if len(difference) < 2:
        return {"mean": float(difference.mean()), "low": None, "high": None, "n": len(difference)}
    rng = np.random.default_rng(seed)
    means = difference[rng.integers(0, len(difference), (n_bootstrap, len(difference)))].mean(1)
    return {"mean": float(difference.mean()), "low": float(np.quantile(means, .025)),
            "high": float(np.quantile(means, .975)), "n": len(difference)}

