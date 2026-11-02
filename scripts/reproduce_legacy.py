"""Reproduce historical OLS results without claiming this is the corrected protocol."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from cranepilot.data import load_dataset, DT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--legacy-results", type=Path)
    p.add_argument("--output", type=Path, default=Path("results/legacy_replication"))
    args = p.parse_args()
    scenarios = load_dataset(args.data)
    args.output.mkdir(parents=True, exist_ok=True)

    def build(sid, h):
        s = scenarios[sid]
        th = s.angle()
        i = np.arange(3, len(th)-h)
        recent = np.column_stack([th[i-1], th[i-2], th[i-3]])
        state = np.column_stack([recent[:,0], (recent[:,0]-recent[:,1])/DT,
                                (recent[:,0]-2*recent[:,1]+recent[:,2])/DT**2])
        return np.column_stack([np.ones(len(i)), state, s.signals["L"][i]]), th[i+h]

    rows = []
    for sid in range(1, 9):
        rest = [s for s in range(1,9) if s != sid]
        val = rest[(len(rest)-1)//2]
        train = [s for s in rest if s != val]
        for h in [10,50,100]:
            parts = [build(s,h) for s in train]
            x = np.concatenate([a for a,b in parts])[::4]
            y = np.concatenate([b for a,b in parts])[::4]
            xt, yt = build(sid,h)
            coef = np.linalg.lstsq(x,y,rcond=None)[0]
            rows.append({"Scenario":sid,"h_sec":h*DT,"actual_horizon_s":(h+1)*DT,
                         "MAE_deg":float(np.rad2deg(np.mean(np.abs(yt-xt@coef))))})
    table = pd.DataFrame(rows)
    report = {"status":"completed", "n_folds":len(table),
              "note":"Historical indexing retained here only; newest input is at i-1, target at i+H."}
    if args.legacy_results:
        old = pd.read_csv(args.legacy_results)
        old = old[old.Model == "linear"][["Scenario","h_sec","MAE_deg"]]
        table = table.merge(old,on=["Scenario","h_sec"], suffixes=("_python","_saved"), validate="one_to_one")
        table["absolute_difference_deg"] = np.abs(table.MAE_deg_python-table.MAE_deg_saved)
        report["max_absolute_difference_deg"] = float(table.absolute_difference_deg.max())
        if len(table) != 24 or report["max_absolute_difference_deg"] > 1e-8:
            raise RuntimeError(f"Legacy reproduction failed: {report}")
    table.to_csv(args.output/"per_fold.csv",index=False)
    (args.output/"summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()

