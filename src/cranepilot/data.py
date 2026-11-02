"""Immutable import of the published 33-scenario Inteco tower-crane dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd

DATASET_DOI = "10.17632/x69hyyv3zw.2"
DT = 0.01
QUANTIZATION_RAD = 2 * np.pi / 4096


@dataclass
class Scenario:
    sid: int
    time: np.ndarray
    signals: dict[str, np.ndarray]
    metadata: dict

    @property
    def family(self) -> str:
        return self.metadata["family"]

    @property
    def fixed_length(self) -> bool:
        return bool(np.ptp(self.signals["L"]) < 1e-6)

    def angle(self, axis: str = "theta") -> np.ndarray:
        return self.signals[axis]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_dataset(source: Path, destination: Path) -> dict:
    """Read original XLSX bytes, validate all rows, retain checksums and SI units."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for path in source.glob("*.xlsx"):
        match = re.match(r"^\(?(\d+)", path.name)
        if match:
            sid = int(match.group(1))
            if sid in files:
                raise ValueError(f"Duplicate scenario {sid}")
            files[sid] = path
    if set(files) != set(range(1, 35)):
        raise ValueError(f"Expected scenarios 1..33 and time file 34; found {sorted(files)}")
    time = pd.read_excel(files[34], engine="openpyxl").to_numpy(dtype=float).ravel()
    if len(time) != 9901 or not np.allclose(np.diff(time), DT, atol=1e-10):
        raise ValueError("Unexpected time grid")
    arrays = {"time": time}
    manifest = {"dataset_doi": DATASET_DOI, "license": "CC-BY-4.0", "dt_s": DT,
                "time_file": {"name": files[34].name, "sha256": sha256(files[34])},
                "units": {"L": "m", "x": "m", "theta": "rad", "beta": "rad",
                          "J": "rad", "xr": "m (reported reference; servo scaling unverified)",
                          "Jr": "rad (reported reference; servo scaling unverified)"},
                "scenarios": []}
    for sid in range(1, 34):
        path = files[sid]
        frame = pd.read_excel(path, engine="openpyxl")
        names = (["L", "xr", "x", "theta"] if sid <= 13 else
                 ["L", "Jr", "J", "theta", "beta"] if sid <= 23 else
                 ["L", "xr", "x", "theta", "Jr", "J", "beta"])
        values = frame.to_numpy(dtype=float)
        if values.shape != (len(time), len(names)) or not np.isfinite(values).all():
            raise ValueError(f"Invalid data in scenario {sid}: {values.shape}")
        lower = path.name.lower()
        mass = next((n for n in (75, 170, 500) if f"{n} g" in lower), 300)
        meta = {"sid": sid, "file_name": path.name, "sha256": sha256(path),
                "family": "trolley" if sid <= 13 else "jib" if sid <= 23 else "simult",
                "mass_g": mass, "mass_source": "file name; nominal 300 g from dataset description",
                "shape": "beam" if "beam" in lower else "ball" if "ball" in lower else "cylinder",
                "n_samples": len(time), "signals": names, "original_headers": list(map(str, frame.columns)),
                "L_min_m": float(values[:, 0].min()), "L_max_m": float(values[:, 0].max()),
                "fixed_length": bool(np.ptp(values[:, 0]) < 1e-6)}
        manifest["scenarios"].append(meta)
        for j, name in enumerate(names):
            arrays[f"s{sid}_{name}"] = values[:, j]
    np.savez_compressed(destination / "dataset.npz", **arrays)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_dataset(path: Path) -> dict[int, Scenario]:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    with np.load(path / "dataset.npz", allow_pickle=False) as z:
        return {m["sid"]: Scenario(m["sid"], z["time"].copy(),
                 {name: z[f's{m["sid"]}_{name}'].copy() for name in m["signals"]}, m)
                for m in manifest["scenarios"]}


def make_windows(scenario: Scenario, axis: str, horizons: list[int], history: int = 41,
                 stride: int = 1, start_s: float = 0.0, end_s: float | None = None):
    """Newest available angle is theta[k]; target is theta[k+H], exactly H*dt later."""
    theta = scenario.angle(axis)
    lo = max(history - 1, int(np.ceil(start_s / DT)))
    hi = min(len(theta) - max(horizons), len(theta) if end_s is None else int(end_s / DT))
    k = np.arange(lo, hi, stride, dtype=int)
    if len(k) == 0:
        raise ValueError("No forecast windows in the requested interval")
    past = theta[k[:, None] - np.arange(history)]
    y = theta[k[:, None] + np.asarray(horizons)]
    return {"past": past, "L": scenario.signals["L"][k], "y": y, "k": k,
            "time": scenario.time[k], "sid": np.full(len(k), scenario.sid, dtype=int)}


def combine_windows(scenarios, ids, axis, horizons, history=41, stride=5):
    parts = [make_windows(scenarios[s], axis, horizons, history, stride) for s in ids]
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def fixed_folds():
    """Scenario-level LOSO with a whole, distinct middle-length validation record."""
    groups = {"trolley": list(range(1, 9)), "jib": list(range(14, 21)), "simult": list(range(24, 32))}
    folds = []
    for family, ids in groups.items():
        for test in ids:
            rest = [s for s in ids if s != test]
            val = rest[(len(rest) - 1) // 2]
            for axis in (["theta"] if family == "trolley" else ["theta", "beta"]):
                folds.append({"name": f"loso_{family}_{axis}_{test}", "family": family,
                              "axis": axis, "train": [s for s in rest if s != val],
                              "validation": [val], "test": [test], "evaluation": "retrospective_loso"})
    return folds


def extrapolation_fold():
    return {"name": "trolley_length_extrapolation", "family": "trolley", "axis": "theta",
            "train": [3, 5, 6], "validation": [4], "test": [1, 2, 7, 8],
            "evaluation": "retrospective_length_extrapolation"}

