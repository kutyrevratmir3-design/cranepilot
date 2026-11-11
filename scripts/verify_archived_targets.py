"""Check target timestamp parity by reproducing the existing dense ARX metrics."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from delayed_baselines import build_packet, check_record, digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    task = json.loads((args.repo/"artifacts/modal_bundle_20261107/task.json").read_text())
    protocol = json.loads((args.repo/"configs/dated_forecast_20261108.json").read_text())
    horizon = np.array(task["horizons_raw"], dtype=int)
    root = args.repo/"results/dense_reference_20261108/frozen_replay"
    metrics_path = root/"metrics.csv"
    metrics = pd.read_csv(metrics_path)
    metrics = metrics[(metrics.group=="all") & (metrics.method=="dense_matched_arx")]
    grouped = metrics.groupby(["case", "scenario", "age_ms", "noise_quantum"])
    maximum = 0.
    packets, forecast_errors = 0, 0
    idx = 0
    with np.load(args.dataset, allow_pickle=False) as archive:
        for family, fold in task["families"].items():
            for axis in fold["axes"]:
                name = f"{family}_{axis}"
                for sid in fold["confirmation"]:
                    record = check_record(archive, sid, family)
                    for age, noise in itertools.product(protocol["evaluation_ages_raw"],
                                                         protocol["noise_quantum_multipliers"]):
                        packet = build_packet(record, sid, axis, idx, age, noise, protocol, horizon, task)
                        with np.load(root/f"{name}_s{sid}_a{age}_n{noise}.npz", allow_pickle=False) as saved:
                            np.testing.assert_array_equal(packet["k"], saved["k"])
                            target = record[axis][packet["k"][:, None]+horizon]
                            error = np.abs(np.rad2deg(saved["dense_matched_arx"]-target))
                        archived = grouped.get_group((name,sid,age*10,noise)).sort_values("horizon_s")
                        np.testing.assert_allclose(archived.horizon_s.to_numpy(), horizon*.01, atol=1e-12)
                        np.testing.assert_allclose(archived.mae_deg.to_numpy(), error.mean(0), atol=1e-12, rtol=0)
                        maximum = max(maximum, float(np.max(np.abs(archived.mae_deg.to_numpy()-error.mean(0)))))
                        forecast_errors += error.size
                        packets += 1
                idx += 1
    write_json(args.out, dict(max_absolute_archived_mae_difference_deg=maximum, matched_packets=packets,
                              target_prediction_pairs_checked=forecast_errors, source_metrics_sha256=digest(metrics_path),
                              dataset_sha256=digest(args.dataset)))


if __name__ == "__main__":
    main()
