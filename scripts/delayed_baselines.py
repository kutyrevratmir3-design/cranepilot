"""Chronological, timestamp-aware reference models for the archived Inteco replay.

Run the develop and replay subcommands separately. Development writes a frozen
selection and its source/protocol hashes before the replay subcommand can run.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

DT = 0.01
ENCODER_STEP = 2 * np.pi / 4096
G = 9.81
OBSERVE = np.array([1., 0., 1.])
IDENTITY = np.eye(3)
DERIVATIVE_SAMPLES = 21
SUPPORT_HISTORY = 401
LOCAL_HISTORY = 251
LOCAL_DECIMATION = 5
DESIGN = Path(__file__).with_name("delayed_baselines_design.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def check_record(archive, scenario, family):
    names = {"trolley": ("L", "x", "theta"), "jib": ("L", "J", "theta", "beta"),
             "simult": ("L", "x", "J", "theta", "beta")}[family]
    out = {key: archive[f"s{scenario}_{key}"] for key in names}
    if not all(v.shape == (9901,) and np.isfinite(v).all() for v in out.values()):
        raise ValueError(f"Invalid sample grid or values for record {scenario}")
    if not (out["L"] > 0).all():
        raise ValueError(f"Nonpositive length for record {scenario}")
    return out


def polynomial_coefficients(samples=DERIVATIVE_SAMPLES, centered=False):
    t = DT * (np.arange(samples) - (samples // 2 if centered else samples - 1))
    return np.linalg.pinv(np.column_stack([np.ones(samples), t, t*t/2]))


def derivatives(x, centered=False):
    taps = polynomial_coefficients(centered=centered)
    count = taps.shape[1]
    left = count // 2 if centered else count - 1
    padded = np.pad(x, [(0, 0)]*(x.ndim-1) + [(left, count - 1 - left)], mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, count, axis=-1)
    return np.einsum("...tw,dw->...td", windows, taps)


def support_features(signals):
    """Candidate acceleration terms using measured history through this time."""
    x = derivatives(signals["x"]) if "x" in signals else None
    jib = derivatives(signals["J"]) if "J" in signals else None
    if x is None:
        return np.stack([jib[..., 2], jib[..., 1]**2], axis=-1)
    if jib is None:
        return x[..., 2:3]
    radius = signals["x"]
    return np.stack([x[..., 2], radius*jib[..., 1]**2,
                     radius*jib[..., 2], 2*x[..., 1]*jib[..., 1]], axis=-1)


def estimate_length(records, axis):
    """Dominant frequency on fixed-length TRAIN tails, in the 0.25–4.5 Hz band."""
    design, target = [], []
    for record in records:
        if np.ptp(record["L"]) >= 1e-6:
            continue
        y = record[axis][3500:9000:5]
        n_fft = 16384
        freq = np.fft.rfftfreq(n_fft, DT*5)
        power = np.abs(np.fft.rfft((y-y.mean())*np.hanning(len(y)), n=n_fft))
        band = (freq >= 0.25) & (freq <= 4.5)
        mode = freq[band][np.argmax(power[band])]
        design.append([float(record["L"][0]), 1.])
        target.append(G/(2*np.pi*mode)**2)
    fit = lsq_linear(np.array(design), np.array(target), bounds=([.05, 0], [4, 2]))
    if not fit.success:
        raise ValueError("TRAIN effective-length fit did not converge")
    return fit.x


def effective(length, calibration):
    scale, offset = calibration
    return np.maximum(scale*length + offset, 0.01)


def length_state(length, calibration):
    ell = effective(length, calibration)
    length_rate = derivatives(length)[..., 1] * calibration[0]
    return ell, np.clip(2*length_rate/ell, -2., 2.)


def fit_drive(records, axis, calibration, damping):
    """Ridge regression of the TRAIN angular-acceleration residual on support."""
    inputs, targets = [], []
    for record in records:
        ell, extension = length_state(record["L"], calibration)
        y, velocity, accel = derivatives(record[axis], centered=True).T
        center = float(np.median(record[axis]))
        # Retain t=0.25..98.75 only, outside both derivative edge transients.
        sample = slice(25, -25, 5)
        inputs.append((support_features(record)/ell[:, None])[sample])
        residual = accel + (damping + extension)*velocity + G/ell*(y-center)
        targets.append(residual[sample])
    x, y = np.concatenate(inputs), np.concatenate(targets)
    scale = np.maximum(np.sqrt(np.mean(x*x, axis=0)), 1e-6)
    features = x/scale
    normalized = np.linalg.solve(features.T@features/len(y) + 0.03*np.eye(x.shape[1]),
                                 features.T@y/len(y))
    return normalized/scale


def build_packet(signals, scenario, axis, case_index, age, noise, protocol, horizons, thresholds):
    first = int(round(protocol["origin_start_s"]/DT))
    k = np.arange(first, len(signals[axis])-max(horizons), protocol["origin_stride_raw"])
    history = k[:, None] + np.arange(1 - SUPPORT_HISTORY, 1)
    measured = {key: signals[key][history] for key in ("L", "x", "J") if key in signals}
    angles = signals[axis][history-age].copy()
    if noise:
        seed = np.random.SeedSequence([protocol["noise_seed"], scenario, case_index, age, int(noise*1000)])
        # Archived random draws are applied newest-first before reversing for the filter.
        draw = np.random.default_rng(seed).uniform(-noise*ENCODER_STEP, noise*ENCODER_STEP, angles.shape)
        angles += draw[:, ::-1]
    drive = np.zeros(len(k), dtype=bool)
    for channel in ("x", "J"):
        if channel in measured:
            drive |= np.ptp(measured[channel], axis=1) > thresholds["past_position_range"]
    moving_length = np.ptp(measured["L"], axis=1) > thresholds["past_length_range_m"]
    return dict(angle=angles, measured=measured, age=age, noise=noise, k=k, drive=drive,
                regime=drive.astype(int)+2*moving_length.astype(int))


def prepare(packet, calibration):
    age = packet["age"]
    ell, extra_damping = length_state(packet["measured"]["L"], calibration)
    force_features = support_features(packet["measured"])/ell[..., None]
    # For the first 'age' angle timestamps support is older than its packet.
    # Keep the oldest observed effective length and use zero drive for those steps.
    return (
        G/np.pad(ell, ((0, 0), (age, 0)), mode="edge"),
        np.pad(extra_damping, ((0, 0), (age, 0))),
        np.pad(force_features, ((0, 0), (age, 0), (0, 0))),
    )


def transition(stiffness, damping):
    """Second-order Taylor discretization of xdot=F x + [0,u,0]."""
    shape = stiffness.shape + (3, 3)
    generator = np.zeros(shape)
    generator[..., 0, 1] = 1.
    generator[..., 1, 0] = -stiffness
    generator[..., 1, 1] = -damping
    a = IDENTITY + DT*generator + DT**2/2*(generator@generator)
    b = np.stack([np.full_like(damping, DT**2/2),
                  DT - DT**2/2*damping, np.zeros_like(damping)], axis=-1)
    return a, b


def estimate_at_origin(packet, matrices, force, acceleration_std):
    """Kalman updates end at t-age; no angle update is made inside the gap."""
    a, b = matrices
    inputs = force[..., None]*b
    angle = packet["angle"]
    n = len(angle)
    state = np.zeros((n, 3))
    state[:, 2] = angle[:, 0]
    covariance = np.broadcast_to(np.diag(np.deg2rad([5., 30., 5.])**2), (n, 3, 3)).copy()
    q_acc = acceleration_std**2
    q = np.array([[q_acc*DT**3/3, q_acc*DT**2/2, 0],
                  [q_acc*DT**2/2, q_acc*DT, 0],
                  [0, 0, np.deg2rad(0.05)**2*DT]])
    r = ENCODER_STEP**2/12
    for step in range(a.shape[1]):
        if step < angle.shape[1]:
            projection = covariance@OBSERVE
            innovation_variance = projection@OBSERVE + r
            gain = projection/innovation_variance[:, None]
            innovation = angle[:, step] - state@OBSERVE
            state += gain*innovation[:, None]
            # Symmetric rank-one covariance update for the scalar observation.
            covariance -= gain[..., None]*projection[:, None, :]
        if step + 1 < a.shape[1]:
            step_a = a[:, step]
            state = (step_a@state[..., None])[..., 0] + inputs[:, step]
            if step + 1 < angle.shape[1]:
                covariance = step_a@covariance@step_a.swapaxes(-1, -2) + q
    if not np.isfinite(state).all():
        raise ValueError("Nonfinite filter state")
    return state


def predict_horizons(state, stiffness, damping, last_force, continuation, horizons):
    a, b = transition(stiffness, np.full_like(stiffness, damping))
    forecasts = []
    for step in range(max(horizons)):
        if continuation == "coast":
            forcing = 0.
        elif continuation == "decay":
            forcing = last_force*np.exp(-step*DT/.1)
        else:
            raise ValueError(continuation)
        state = (a@state[..., None])[..., 0] + b*np.asarray(forcing)[..., None]
        if step+1 in horizons:
            forecasts.append(state@OBSERVE)
    return np.column_stack(forecasts)


def harmonic(packet, stiffness, horizons):
    t = DT * np.arange(-LOCAL_HISTORY+1, 1, LOCAL_DECIMATION)
    w = np.sqrt(stiffness[:, SUPPORT_HISTORY-1])
    phase = w[:, None]*t
    basis = np.stack([np.ones_like(phase), np.cos(phase), np.sin(phase)], axis=-1)
    past = packet["angle"][:, -LOCAL_HISTORY::LOCAL_DECIMATION]
    transpose = basis.swapaxes(-1, -2)
    fit = np.linalg.solve(transpose@basis, transpose@past[..., None])[..., 0]
    future = w[:, None]*(np.asarray(horizons)+packet["age"])*DT
    return fit[:, 0, None] + fit[:, 1, None]*np.cos(future) + fit[:, 2, None]*np.sin(future)


def autoregression(packet, order, ridge, horizons):
    raw = packet["angle"][:, -LOCAL_HISTORY::LOCAL_DECIMATION]
    center = raw.mean(axis=1)
    z = raw - center[:, None]
    xy = np.lib.stride_tricks.sliding_window_view(z, order+1, axis=1)
    x, y = xy[..., :-1], xy[..., -1]
    xt = x.swapaxes(-1, -2)
    normal = xt@x
    scale = np.maximum(np.trace(normal, axis1=-2, axis2=-1)/order, ENCODER_STEP**2)
    coef = np.linalg.solve(normal + ridge*scale[:, None, None]*np.eye(order),
                           xt@y[..., None])[..., 0]
    companion = np.broadcast_to(np.eye(order, k=1), (len(z), order, order)).copy()
    companion[:, -1] = coef
    # Keep the free recursion bounded, even on a near-constant local window.
    radius = np.max(np.abs(np.linalg.eigvals(companion)), axis=1)
    shrink = np.minimum(1, .999/np.maximum(radius, 1e-12))
    coef *= shrink[:, None]**np.arange(order, 0, -1)
    state = z[:, -order:].copy()
    lagged = (np.asarray(horizons)+packet["age"])/LOCAL_DECIMATION
    values = [state[:, -1]+center]
    for _ in range(int(np.ceil(lagged.max()))):
        value = np.sum(coef*state, axis=-1)
        state = np.column_stack([state[:, 1:], value])
        values.append(value+center)
    lo, hi = np.floor(lagged).astype(int), np.ceil(lagged).astype(int)
    y = np.column_stack(values)
    return y[:, lo] + (lagged-lo)*(y[:, hi]-y[:, lo])


def balanced_score(predictions, target, drive):
    error = np.abs(np.rad2deg(predictions-target))
    return float(np.mean([error[drive==label].mean() for label in np.unique(drive)]))


def candidates(design):
    return list(itertools.product(design["damping_per_s"],
                                  design["acceleration_process_std"],
                                  design["continuations"]))


def ar_candidates(design):
    return list(itertools.product(design["local_ar_orders"], design["local_ar_ridge"]))


def validate_archived_packet(path, packet, targets):
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(packet["k"], z["k"])
        np.testing.assert_array_equal(packet["regime"], z["regime"])
        if "target" in z:
            np.testing.assert_array_equal(targets, z["target"])


def forecast_kalman(packet, prepared, drive_coefficient, damping, acceleration_std, horizons,
                    continuations):
    stiffness, extra_damping, features = prepared
    force = features@drive_coefficient
    matrices = transition(stiffness, damping + extra_damping)
    state = estimate_at_origin(packet, matrices, force, acceleration_std)
    return {continuation: predict_horizons(state, stiffness[:, -1], damping, force[:, -1],
                                           continuation, horizons) for continuation in continuations}


def metric_rows(pred, targets, packet, case, scenario, horizons):
    groups = ["low_past_motion", "drive_active", "length_active", "drive_and_length_active"]
    for method, values in pred.items():
        error = np.rad2deg(values-targets)
        if error.shape != targets.shape or not np.isfinite(error).all():
            raise ValueError("Invalid output; do not discard any forecast")
        for group, mask in [("all", np.ones(len(error), dtype=bool))] + [
            (name, packet["regime"] == index) for index, name in enumerate(groups)
        ]:
            if not mask.any():
                continue
            mae = np.abs(error[mask]).mean(axis=0)
            rmse = np.sqrt(np.square(error[mask]).mean(axis=0))
            for horizon, m, r in zip(horizons, mae, rmse):
                yield dict(case=case, scenario=scenario, group=group, method=method,
                           horizon_s=horizon*DT, windows=int(mask.sum()), mae_deg=m, rmse_deg=r,
                           age_ms=packet["age"]*10, noise_quantum=packet["noise"])


def develop(archive, protocol, task, archived_development, design, destination, hashes):
    scores, selection = [], {}
    horizons = task["horizons_raw"]
    idx = 0
    for family, fold in task["families"].items():
        records = [check_record(archive, scenario, family) for scenario in fold["train"]]
        for axis in fold["axes"]:
            name = f"{family}_{axis}"
            calibration = estimate_length(records, axis)
            fit = {damping: fit_drive(records, axis, calibration, damping)
                   for damping in design["damping_per_s"]}
            kalman_scores = {key: [] for key in candidates(design)}
            ar_scores = {key: [] for key in ar_candidates(design)}
            for scenario in fold["validation"]:
                record = check_record(archive, scenario, family)
                for age, noise in itertools.product(protocol["train_ages_raw"],
                                                     protocol["noise_quantum_multipliers"]):
                    packet = build_packet(record, scenario, axis, idx, age, noise, protocol, horizons, task)
                    target = record[axis][packet["k"][:, None]+horizons]
                    validate_archived_packet(archived_development/name/f"validation_s{scenario}_a{age}_n{noise}.npz",
                                             packet, target)
                    prepared = prepare(packet, calibration)
                    for damping, process in itertools.product(design["damping_per_s"], design["acceleration_process_std"]):
                        pred = forecast_kalman(packet, prepared, fit[damping], damping, process, horizons,
                                               design["continuations"])
                        for continuation, values in pred.items():
                            key = (damping, process, continuation)
                            value = balanced_score(values, target, packet["drive"])
                            kalman_scores[key].append(value)
                            scores.append(dict(case=name, scenario=scenario, age_ms=age*10,
                                               noise_quantum=noise, method="kalman", key=str(key), mae_deg=value))
                    for order, ridge in ar_scores:
                        key = (order, ridge)
                        y = autoregression(packet, order, ridge, horizons)
                        value = balanced_score(y, target, packet["drive"])
                        ar_scores[key].append(value)
                        scores.append(dict(case=name, scenario=scenario, age_ms=age*10,
                                           noise_quantum=noise, method="local_ar", key=str(key), mae_deg=value))
            best = min(kalman_scores, key=lambda key: np.mean(kalman_scores[key]))
            ar_best = min(ar_scores, key=lambda key: np.mean(ar_scores[key]))
            selection[name] = dict(family=family, axis=axis, train_ids=fold["train"], validation_ids=fold["validation"],
                                   length_calibration=calibration.tolist(),
                                   kalman=dict(damping=best[0], acceleration_std=best[1], continuation=best[2],
                                               drive_coefficient=fit[best[0]].tolist(),
                                               validation_mae_deg=float(np.mean(kalman_scores[best]))),
                                   local_ar=dict(order=ar_best[0], ridge=ar_best[1],
                                                 validation_mae_deg=float(np.mean(ar_scores[ar_best]))))
            print(name, selection[name]["kalman"], selection[name]["local_ar"], flush=True)
            idx += 1
    pd.DataFrame(scores).to_csv(destination/"validation_scores.csv", index=False)
    write_json(destination/"selection.json", dict(cases=selection, hashes=hashes,
               score_file_sha256=digest(destination/"validation_scores.csv"),
               scope=design["scope"], independent_test=False))


def replay(archive, protocol, task, archived_replay, design, destination, hashes):
    selected = json.loads((destination/"selection.json").read_text())
    if selected["hashes"] != hashes or selected["score_file_sha256"] != digest(destination/"validation_scores.csv"):
        raise ValueError("Selection inputs no longer match")
    selection_hash = digest(destination/"selection.json")
    rows, comparisons = [], 0
    horizons = task["horizons_raw"]
    idx = 0
    for family, fold in task["families"].items():
        for axis in fold["axes"]:
            name = f"{family}_{axis}"
            case = selected["cases"][name]
            fit = case["kalman"]
            ar = case["local_ar"]
            calibration = np.array(case["length_calibration"])
            for scenario in fold["confirmation"]:
                if scenario in (case["train_ids"]+case["validation_ids"]):
                    raise ValueError("Replay record leaked into selection")
                record = check_record(archive, scenario, family)
                for age, noise in itertools.product(protocol["evaluation_ages_raw"],
                                                     protocol["noise_quantum_multipliers"]):
                    packet = build_packet(record, scenario, axis, idx, age, noise, protocol, horizons, task)
                    target = record[axis][packet["k"][:, None]+horizons]
                    validate_archived_packet(archived_replay/f"{name}_s{scenario}_a{age}_n{noise}.npz",
                                             packet, target)
                    comparisons += 1
                    prep = prepare(packet, calibration)
                    prediction = {
                        "zero": np.zeros_like(target),
                        "persistence": np.repeat(packet["angle"][:, -1, None], len(horizons), axis=1),
                        "calibrated_oscillator": harmonic(packet, prep[0], horizons),
                        "local_ar": autoregression(packet, ar["order"], ar["ridge"], horizons),
                        "delayed_kalman": forecast_kalman(packet, prep, np.array(fit["drive_coefficient"]),
                                              fit["damping"], fit["acceleration_std"], horizons,
                                              [fit["continuation"]])[fit["continuation"]],
                    }
                    rows.extend(metric_rows(prediction, target, packet, name, scenario, horizons))
            idx += 1
            print(f"{name}: archived packet origins/regimes matched", flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(destination/"replay_metrics.csv", index=False)
    all_metrics = metrics[metrics.group=="all"]
    record = all_metrics.groupby(["method","age_ms","noise_quantum","scenario","horizon_s"], as_index=False).mae_deg.mean()
    record.to_csv(destination/"record_horizon_summary.csv", index=False)
    score = record.groupby(["method","age_ms","noise_quantum","horizon_s"], as_index=False).mae_deg.mean()
    score.to_csv(destination/"horizon_summary.csv", index=False)
    age_score = score.groupby(["method","age_ms","noise_quantum"], as_index=False).mae_deg.mean()
    age_score.to_csv(destination/"age_summary.csv", index=False)
    write_json(destination/"replay_manifest.json", dict(
        inputs_sha256=hashes, selection_sha256=selection_hash, matched_packet_conditions=comparisons,
        replay_record_count=int(record.scenario.nunique()), archive_origins_and_regimes_exact=True,
        engine=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        generated_sha256={name:digest(destination/name) for name in
                          ["replay_metrics.csv","record_horizon_summary.csv","horizon_summary.csv","age_summary.csv"]}))
    print(age_score[age_score.noise_quantum==0].pivot(index="age_ms", columns="method", values="mae_deg").round(4))


def run_checks():
    """Synthetic clock/causality tests and a standard sequential Kalman reference check."""
    packet = dict(angle=np.array([[1., 2., 3.]]), noise=0)
    stiffness = np.full((1, 4), 9.81)
    damping = np.full((1, 4), .5)
    drive = np.array([[.25, .25, 1., 0.]])
    a, b = transition(stiffness, damping)
    scalar = estimate_at_origin(packet, (a, b), drive, .03)
    # Compare against the Joseph-form covariance update on the tiny dated packet.
    q = np.array([[.03**2*DT**3/3, .03**2*DT**2/2, 0],
                  [.03**2*DT**2/2, .03**2*DT, 0], [0,0,np.deg2rad(.05)**2*DT]])
    r = ENCODER_STEP**2/12
    state = np.array([0., 0., 1.])
    covariance = np.diag(np.deg2rad([5.,30.,5.])**2)
    for index in range(4):
        if index < 3:
            gain = (covariance@OBSERVE)/(OBSERVE@covariance@OBSERVE + r)
            state += gain*(packet["angle"][0,index]-OBSERVE@state)
            residual = IDENTITY - np.outer(gain, OBSERVE)
            covariance = residual@covariance@residual.T + r*np.outer(gain,gain)
        if index < 3:
            state = a[0,index]@state + b[0,index]*drive[0,index]
            covariance = a[0,index]@covariance@a[0,index].T + q
    np.testing.assert_allclose(scalar[0], state, rtol=1e-11, atol=1e-11)
    stationary = np.ones((2, 401))
    p = dict(angle=stationary*.25, measured={"L": stationary, "x": stationary}, age=20, noise=0)
    y = forecast_kalman(p, prepare(p, [1., 0.]), np.zeros(1), .5, .03, [10,20], ["coast"])["coast"]
    np.testing.assert_allclose(y, .25, rtol=0, atol=1e-12)
    np.testing.assert_allclose(autoregression(p, 4, .01, [10,20]), .25, atol=1e-12)
    t = np.arange(1000)*DT
    signals = {"L": np.ones(1000), "x": np.sin(t), "theta": np.cos(t)}
    protocol = dict(origin_start_s=4.25, origin_stride_raw=100, noise_seed=20261108)
    thresholds = dict(past_position_range=.002, past_length_range_m=.002)
    original = build_packet(signals, 9, "theta", 0, 25, 1, protocol, [10,20], thresholds)
    altered = {name: values.copy() for name, values in signals.items()}
    for name in ("L", "x"):
        altered[name][426:] += 100.
    altered["theta"][401:] += 100.
    masked = build_packet(altered, 9, "theta", 0, 25, 1, protocol, [10,20], thresholds)
    np.testing.assert_array_equal(original["angle"][0], masked["angle"][0])
    for name in original["measured"]:
        np.testing.assert_array_equal(original["measured"][name][0], masked["measured"][name][0])
    print("Delayed Kalman Joseph-form, stationary gap, stationary local AR, and packet clock checks passed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check", "develop", "replay"])
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    run_checks()
    if args.stage == "check":
        return
    if any(p is None for p in (args.repo, args.dataset, args.out)):
        parser.error("--repo, --dataset, and --out are required")
    args.out.mkdir(parents=True, exist_ok=True)
    protocol_path = args.repo/"configs/dated_forecast_20261108.json"
    task_path = args.repo/"artifacts/modal_bundle_20261107/task.json"
    protocol, task, design = map(lambda p: json.loads(p.read_text()), [protocol_path, task_path, DESIGN])
    archive_root = args.repo/"results/dense_reference_20261108"
    source_hashes = {p.name: digest(p) for p in [Path(__file__), DESIGN, protocol_path, task_path, args.dataset]}
    lock = args.out/"design_lock.json"
    with np.load(args.dataset, allow_pickle=False) as archive:
        np.testing.assert_allclose(np.diff(archive["time"]), DT, rtol=0, atol=1e-10)
        if args.stage == "develop":
            if lock.exists():
                raise FileExistsError("Use a new run output directory; retain the original design lock")
            write_json(lock, dict(hashes=source_hashes, stage="before_development", design=design))
            develop(archive, protocol, task, archive_root/"frozen_development", design, args.out, source_hashes)
        else:
            if json.loads(lock.read_text())["hashes"] != source_hashes:
                raise ValueError("Code or protocol changed after design lock")
            if (args.out/"replay_manifest.json").exists():
                raise FileExistsError("Preserve the existing replay")
            replay(archive, protocol, task, archive_root/"frozen_replay", design, args.out, source_hashes)


if __name__ == "__main__":
    main()
