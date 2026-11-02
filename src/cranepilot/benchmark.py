"""Scenario-separated retrospective forecast benchmark, including honest failures."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import time
import numpy as np
import pandas as pd
from .data import load_dataset, make_windows, combine_windows, DT
from .physics import LengthCalibration, oscillator_forecast, phase_features, rollout_sensitivities
from .models import (raw_features, lpv_features, LinearForecaster, BoostedTrees,
                     GPResidual, NeuralForecaster, NeuralConfig)
from .metrics import regression_metrics


def environment_record():
    from importlib.metadata import version
    source = Path(__file__).resolve().parent
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")},
              "packages": {p: version(p) for p in ["numpy", "scipy", "scikit-learn", "torch", "pandas", "matplotlib"]}}
    try:
        result["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
    return result


def causal_activity(scenario, index, axis):
    """Activity from backward differences; neither time>15 nor a future label."""
    signals = scenario.signals
    active = np.zeros(len(index), dtype=bool)
    for name, threshold in (("x", .005), ("J", .02), ("L", .005)):
        if name not in signals:
            continue
        k0 = np.maximum(index-25, 0)
        rate = np.abs(signals[name][index]-signals[name][k0])/np.maximum((index-k0)*DT, DT)
        active |= rate > threshold
    return active


def run_benchmark(data_dir: Path, output: Path, config: dict):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scenarios = load_dataset(data_dir)
    config_bytes = json.dumps(config, sort_keys=True).encode()
    provenance = {"config": config, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                  "environment": environment_record(), "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                  "dataset_manifest_sha256": hashlib.sha256((Path(data_dir)/"manifest.json").read_bytes()).hexdigest(),
                  "status": "running", "claims": "retrospective exploratory; no new hardware validation"}
    (output/"provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    rows, training = [], []
    horizons = config["horizons_steps"]
    horizon_s = np.array(horizons)*DT
    for history in config["histories"]:
        for fold in config["folds"]:
            ids, val, test = set(fold["train"]), set(fold["validation"]), set(fold["test"])
            if ids & val or ids & test or val & test:
                raise ValueError("Scenario leakage in split manifest")
            axis = fold["axis"]
            tr = combine_windows(scenarios, fold["train"], axis, horizons, history, config["train_stride"])
            va = combine_windows(scenarios, fold["validation"], axis, horizons, history, config["train_stride"])
            cal = LengthCalibration.fit(scenarios, fold["train"], axis)
            physics_tr, _ = oscillator_forecast(tr["past"], cal.omega(tr["L"]), horizon_s)
            physics_va, _ = oscillator_forecast(va["past"], cal.omega(va["L"]), horizon_s)
            phase_tr = phase_features(tr["past"], cal.omega(tr["L"]))
            phase_va = phase_features(va["past"], cal.omega(va["L"]))
            raw_tr, raw_va = raw_features(tr), raw_features(va)
            predictors = {}
            model_metadata = {}
            print(f"Fitting {fold['name']} history={history} n={len(tr['y'])}", flush=True)

            def fit_linear(name, x, y):
                started = time.perf_counter()
                predictors[name] = LinearForecaster(0.0 if name == "ols" else 1.0).fit(x, y)
                model_metadata[name] = {"fit_s": time.perf_counter()-started}

            fit_linear("ols", raw_tr, tr["y"])
            fit_linear("lpv", lpv_features(tr), tr["y"])
            fit_linear("phase_ridge_residual", phase_tr, tr["y"]-physics_tr)
            if config.get("trees", True):
                started = time.perf_counter()
                predictors["boosted_trees"] = BoostedTrees().fit(raw_tr, tr["y"])
                model_metadata["boosted_trees"] = {"fit_s": time.perf_counter()-started}
            if config.get("gp", True):
                started = time.perf_counter()
                predictors["gp_residual"] = GPResidual().fit(phase_tr, tr["y"]-physics_tr)
                model_metadata["gp_residual"] = {"fit_s": time.perf_counter()-started}
            sensitivity = rollout_sensitivities(tr["past"], cal.omega(tr["L"]), horizon_s)
            neural_specs = [("mlp", raw_tr, raw_va, tr["y"], va["y"], 0.0),
                            ("phase_nn_residual", phase_tr, phase_va, tr["y"]-physics_tr, va["y"]-physics_va, 0.0),
                            ("projected_phase_residual", phase_tr, phase_va, tr["y"]-physics_tr, va["y"]-physics_va,
                             config.get("projection_weight", .03))]
            if config.get("neural", True):
                for name, xt, xv, yt, yv, weight in neural_specs:
                    ensemble = []
                    started = time.perf_counter()
                    for seed in config["seeds"]:
                        nc = NeuralConfig(epochs=config["epochs"], projection_weight=weight)
                        model = NeuralForecaster(nc, seed).fit(xt, yt, xv, yv, sensitivity if weight else None)
                        model.save(output/"models"/f"{fold['name']}_h{history}_{name}_seed{seed}.pt")
                        ensemble.append(model)
                        training.append({"fold": fold["name"], "history": history, "model": name,
                                         "seed": seed, "n_parameters": model.n_parameters,
                                         "epochs": len(model.log), "val_mse_rad2": model.best_validation_mse_rad2})
                    predictors[name] = ensemble
                    model_metadata[name] = {"fit_s": time.perf_counter()-started, "ensemble_size": len(ensemble)}
                    print(f"  {name} done in {model_metadata[name]['fit_s']:.1f}s", flush=True)
            for sid in fold["test"]:
                te = make_windows(scenarios[sid], axis, horizons, history, config["eval_stride"])
                omega = cal.omega(te["L"])
                physical, _ = oscillator_forecast(te["past"], omega, horizon_s)
                ideal, _ = oscillator_forecast(te["past"], np.sqrt(9.81/te["L"]), horizon_s)
                xt, xp = raw_features(te), phase_features(te["past"], omega)
                predictions = {"persistence": np.repeat(te["past"][:, :1], len(horizons), axis=1),
                               "ideal_oscillator": ideal, "calibrated_oscillator": physical}
                for name, model in predictors.items():
                    x = lpv_features(te) if name == "lpv" else xp if "residual" in name else xt
                    started = time.perf_counter()
                    p = np.mean([m.predict(x) for m in model], axis=0) if isinstance(model, list) else model.predict(x)
                    predictions[name] = p + physical if "residual" in name else p
                    model_metadata[name]["predict_batch_s"] = time.perf_counter()-started
                activity = causal_activity(scenarios[sid], te["k"], axis)
                regimes = {"all": np.ones(len(activity), dtype=bool), "active": activity, "quiet": ~activity}
                archive = {"target": te["y"], "k": te["k"], **predictions}
                np.savez_compressed(output/f"predictions_{fold['name']}_s{sid}_h{history}.npz", **archive)
                for name, p in predictions.items():
                    for regime, mask in regimes.items():
                        if not mask.any():
                            continue
                        for j, horizon in enumerate(horizon_s):
                            rows.append({"fold": fold["name"], "evaluation": fold["evaluation"],
                                "family": fold["family"], "axis": axis, "scenario": sid,
                                "history_samples": history, "horizon_s": float(horizon), "regime": regime,
                                "model": name, "L_m": float(te["L"][0]),
                                **regression_metrics(te["y"][mask, j], p[mask, j])})
                print(f"  tested scenario {sid}", flush=True)
            (output/f"calibration_{fold['name']}_h{history}.json").write_text(
                json.dumps({"calibration": asdict(cal), "models": model_metadata}, indent=2), encoding="utf-8")
            pd.DataFrame(rows).to_csv(output/"metrics.csv", index=False)
            pd.DataFrame(training).to_csv(output/"training.csv", index=False)
    provenance.update(status="complete", completed_utc=pd.Timestamp.now(tz="UTC").isoformat())
    (output/"provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return pd.DataFrame(rows)

