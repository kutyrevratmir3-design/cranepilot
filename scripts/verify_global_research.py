"""Replay every retained real record from saved global-research weights.

Run only after the research provenance says complete. All guarded inputs,
checkpoints, retained results and the locked selection are hashed before model
loading. Prediction inputs are rebuilt as history-only packets; measured future
angles are used solely to verify the retained targets. No fitting is performed.
Paths passed on the command line are relative to the main project directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

import numpy as np

from cranepilot.data import load_dataset, sha256


TOLERANCE_DEG = 5e-5
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSFER_ROOT = PROJECT_ROOT.parent.parent
ORIGINAL_COUNTS = {"main": 1111, "aphynity": 170}


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def verify_hash_mapping(mapping, base):
    """Return every missing/changed file, without modifying the evidence."""
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("A nonempty retained hash mapping is required")
    missing, changed, errors = [], [], []
    checked = 0
    for name, expected in mapping.items():
        path = base / name.replace("\\", "/")
        if not path.is_file():
            missing.append(name)
            continue
        try:
            actual = sha256(path)
        except OSError as exc:
            errors.append({"path": name, "error": str(exc)})
            continue
        checked += 1
        if actual != expected:
            changed.append({"path": name, "expected": expected, "actual": actual})
    return {"passed": not (missing or changed or errors), "expected_files": len(mapping),
            "checked_files": checked, "missing": missing, "changed": changed,
            "read_errors": errors}


def verify_original_files():
    snapshot = TRANSFER_ROOT / "handoff/restoration_20261107/original_file_hashes.json"
    original = json.loads(snapshot.read_text())
    if set(original) != set(ORIGINAL_COUNTS):
        raise ValueError("Unexpected original-file manifest sections")
    roots = {"main": PROJECT_ROOT,
             "aphynity": TRANSFER_ROOT / "aphynity_pilot/cranepilot"}
    result = {"manifest": str(snapshot), "manifest_sha256": sha256(snapshot),
              "roots": {name: str(root) for name, root in roots.items()}, "projects": {}}
    for name, expected_count in ORIGINAL_COUNTS.items():
        if len(original[name]) != expected_count:
            raise ValueError(f"Expected {expected_count} original {name} files")
        result["projects"][name] = verify_hash_mapping(original[name], roots[name])
    result["passed"] = all(value["passed"] for value in result["projects"].values())
    result["checked_files"] = sum(value["checked_files"] for value in result["projects"].values())
    return result


def _finite_matrix(value, rows, label):
    value = np.asarray(value)
    if value.shape != (rows, 30) or not np.isfinite(value).all():
        raise AssertionError(f"{label}: finite ({rows}, 30) matrix required")
    return value


def replay(run, provenance, selection, report):
    # These frozen implementations are imported only after the evidence hashes
    # and the original-file snapshot have been verified successfully.
    import torch
    from cranepilot.rollout import command_windows
    from scripts.global_research_candidate import GlobalResearchCandidate, CONTROLS
    from scripts.predict_cranepilot import build_packet

    torch.set_num_threads(4)
    cfg = provenance["config"]
    if len(cfg["folds"]) != 3 or len(set(cfg["folds"])) != 3:
        raise ValueError("This verification requires the three frozen research folds")
    expected_new = {f"weak_r{i}" for i in range(len(cfg["weak_ridges"]))}
    expected_new.update(f"{basis}_{mode}_r{i}" for basis in cfg["bases"]
                        for mode in cfg["input_modes"] for i in range(len(cfg["alphas"])))
    expected_models = set(CONTROLS) | expected_new
    if len(expected_new) != 15 or len(CONTROLS) != 4 or len(expected_models) != 19:
        raise ValueError("Expected 15 fitted methods and four retained controls per fold")
    source = json.loads((Path(cfg["source_run"]) / "provenance.json").read_text())
    if source["status"] != "complete":
        raise ValueError("The retained split source must be complete")
    data = load_dataset(PROJECT_ROOT / "data/processed")
    report.update(folds=list(cfg["folds"]), fitted_models_per_fold=15,
                  methods_per_fold=19, fitted_models_total=45,
                  history_only_inputs=True, history_samples=41, prediction_steps=30,
                  target_comparison="exact_array_equal", records=[], checks=[])

    for fold in cfg["folds"]:
        folder = run / fold
        split = source["config"]["folds"][fold]
        groups = {}
        for group in ("validation", "evaluation", "load_stress"):
            for sid in split[group]:
                if sid in groups:
                    raise ValueError(f"Repeated real scenario in fold {fold}: {sid}")
                groups[sid] = group
        files = list(folder.glob("predictions_s*.npz"))
        by_sid = {}
        for path in files:
            match = re.fullmatch(r"predictions_s([0-9]+)\.npz", path.name)
            if match is None:
                raise ValueError(f"Unexpected real prediction filename: {path}")
            sid = int(match.group(1))
            if sid in by_sid:
                raise ValueError(f"Repeated retained scenario: {fold}/{sid}")
            by_sid[sid] = path
        if set(by_sid) != set(groups):
            raise AssertionError(f"Real-record coverage mismatch in {fold}: "
                                 f"expected {sorted(groups)}, found {sorted(by_sid)}")
        candidate = GlobalResearchCandidate(cfg, fold, folder)
        if set(candidate.manifest) != expected_new:
            raise AssertionError(f"Unexpected fitted model manifest in {fold}")
        for sid in sorted(by_sid):
            path = by_sid[sid]
            relative_path = str(path.relative_to(run))
            if relative_path not in provenance["result_hashes"]:
                raise AssertionError(f"Real predictions lack a retained result hash: {relative_path}")
            scenario = data[sid]
            with np.load(path, allow_pickle=False) as saved:
                k = saved["k"]
                if (k.ndim != 1 or not len(k) or not np.issubdtype(k.dtype, np.integer)
                        or np.any(np.diff(k) <= 0)):
                    raise AssertionError(f"Invalid saved origins: {relative_path}")
                expected_k = np.arange(40, len(scenario.time) - 150, cfg["evaluation_stride"])
                np.testing.assert_array_equal(k, expected_k, err_msg=relative_path)
                raw = command_windows(scenario, candidate.calibration, k)
                target = _finite_matrix(saved["target"], len(k), relative_path + "/target")
                np.testing.assert_array_equal(raw["target_q"], target, err_msg=relative_path)
                saved_acc = _finite_matrix(saved["acceleration"], len(k), relative_path + "/acceleration")

                # Build the deployed interface directly from the physical data,
                # excluding targets, scenario IDs and future measured positions.
                past = k[:, None] - np.arange(41)
                future = k[:, None] + np.arange(30) * 5
                packet = build_packet(scenario.signals["theta"][past],
                                      scenario.signals["x"][past],
                                      scenario.signals["xr"][past],
                                      scenario.signals["L"][k],
                                      scenario.signals["xr"][future], candidate.calibration)
                actual, acceleration = candidate.forecast(packet)
                if set(actual) != expected_models:
                    raise AssertionError(f"Unexpected forecast model set in {fold}")
                acceleration = _finite_matrix(acceleration, len(k), "replayed acceleration")
                saved_models = set(saved.files) - {"k", "target", "acceleration"}
                aliases = saved_models & {"selected", "common"}
                if saved_models != expected_models | aliases:
                    raise AssertionError(f"Missing/extra saved methods in {relative_path}")
                if groups[sid] != "validation" and aliases != {"selected", "common"}:
                    raise AssertionError(f"Evaluation policy outputs missing in {relative_path}")
                resolved = {"selected": selection["folds"][fold]["model"],
                            "common": selection["common"]["model"]}
                for alias in aliases:
                    actual[alias] = actual[resolved[alias]]
                maximum = 0.
                for name in sorted(saved_models):
                    expected = _finite_matrix(saved[name], len(k), relative_path + "/" + name)
                    predicted = _finite_matrix(actual[name], len(k), "replayed " + name)
                    error = float(np.max(np.abs(np.rad2deg(predicted - expected))))
                    if error > TOLERANCE_DEG:
                        raise AssertionError(f"Replay error {error} deg for {relative_path}/{name}")
                    maximum = max(maximum, error)
                    report["checks"].append({"fold": fold, "scenario": sid,
                                             "group": groups[sid], "model": name,
                                             "resolved_model": resolved.get(name, name),
                                             "packets": len(k), "max_error_deg": error})
                report["records"].append({"fold": fold, "scenario": sid,
                                           "group": groups[sid], "file": relative_path,
                                           "packets": len(k), "methods": len(saved_models),
                                           "targets_exact": True, "all_shapes_finite": True,
                                           "max_error_deg": maximum,
                                           "acceleration_max_abs_difference_m_s2": float(
                                               np.max(np.abs(acceleration - saved_acc)))})
            print(f"REPLAY {fold} scenario={sid} packets={len(k)} "
                  f"methods={len(saved_models)} max_error_deg={maximum:.9g}", flush=True)
    report.update(real_record_files=len(report["records"]),
                  packet_count=sum(item["packets"] for item in report["records"]),
                  prediction_comparisons=len(report["checks"]),
                  max_error_deg=max(item["max_error_deg"] for item in report["checks"]))


def main():
    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
                        default=Path("artifacts/runs/global_research_20261107"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/global_research_20261107/replay_verification.json"))
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve previous verification; choose a fresh --output")
    provenance_path = run / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if provenance["status"] != "complete":
        raise ValueError("Wait for completed research; replay does not run during training/evaluation")
    report = {"passed": False, "run": str(run), "started_utc": _utc_now(),
              "tolerance_deg": TOLERANCE_DEG, "relative_tolerance": 0.,
              "fitting_performed": False, "verifier_sha256": sha256(Path(__file__)),
              "provenance_sha256": sha256(provenance_path)}
    try:
        integrity = {}
        for key, base in (("guarded_hashes", PROJECT_ROOT), ("checkpoint_hashes", run),
                          ("result_hashes", run)):
            integrity[key] = verify_hash_mapping(provenance[key], base)
        integrity["selection"] = verify_hash_mapping(
            {"selection.json": provenance["selection_sha256"]}, run)
        report["integrity"] = integrity
        report["original_files"] = verify_original_files()
        if not all(item["passed"] for item in integrity.values()):
            raise AssertionError("Retained research evidence failed SHA-256 verification")
        if not report["original_files"]["passed"]:
            raise AssertionError("Original main/APhYNITY files differ from the restoration snapshot")
        print("HASHES verified; original files main=1111 aphynity=170", flush=True)
        selection = json.loads((run / "selection.json").read_text())
        replay(run, provenance, selection, report)
        report["passed"] = True
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["finished_utc"] = _utc_now()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"passed": report["passed"], "output": str(output),
                      "real_record_files": report["real_record_files"],
                      "prediction_comparisons": report["prediction_comparisons"],
                      "max_error_deg": report["max_error_deg"],
                      "original_files_unchanged": report["original_files"]["checked_files"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
