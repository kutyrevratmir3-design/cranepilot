"""Compare fixed prefix acceptance against retained ungated and zero-shot physics."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import subprocess
import numpy as np
import pandas as pd
from cranepilot.data import DT, load_dataset, make_windows
from cranepilot.physics import LengthCalibration, oscillator_forecast
from cranepilot.adaptation import fit_prefix_oscillator, fit_validated_prefix
from cranepilot.metrics import regression_metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=Path("configs/validated_adaptation_pilot.json"))
    parser.add_argument("--output",type=Path,default=Path("artifacts/runs/validated_adaptation"))
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserve prior evidence: output already exists: {args.output}")
    args.output.mkdir(parents=True)
    config=json.loads(args.config.read_text())
    scenarios=load_dataset(Path("data/processed"))
    calibration=LengthCalibration.fit(scenarios,config["train"],"theta")
    files=[*Path("src").rglob("*.py"),Path(__file__),args.config]
    provenance={"status":"running","config":config,
                "config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                "data_manifest_sha256":hashlib.sha256(Path("data/processed/manifest.json").read_bytes()).hexdigest(),
                "git_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                "git_status":subprocess.check_output(["git","status","--short"],text=True),
                "python":platform.python_version(),
                "information":"ungated uses angles and measured trolley prefix; validated uses angles only; same scored origins",
                "validation_policy":"scenario 4 processed and recorded before test; no thresholds retuned from its result"}
    (args.output/"provenance.json").write_text(json.dumps(provenance,indent=2))
    rows=[]; fits=[]
    for group in ("validation","test"):
        for sid in config[group]:
            scenario=scenarios[sid]
            prior=float(calibration.omega(scenario.signals["L"][0]))
            period=2*np.pi/prior
            horizons=config["horizons_s"]
            windows=make_windows(scenario,"theta",np.rint(np.array(horizons)/DT).astype(int),
                                 history=config["history_samples"],stride=2,start_s=config["score_start_s"])
            for cycles in config["prefix_period_budgets"]:
                budget=cycles*period
                if budget>config["score_start_s"]-(config["history_samples"]-1)*DT:
                    raise ValueError("Prefix overlaps scored observation histories")
                base={"omega":prior,"damping":0.,"status":"zero_shot"}
                methods={"zero_shot":base,
                         "ungated":fit_prefix_oscillator(scenario.angle(),scenario.signals["x"],prior,budget),
                         "validated":fit_validated_prefix(scenario.angle(),prior,budget,
                             history=config["history_samples"],train_fraction=config["train_fraction"],
                             minimum_improvement=config["minimum_improvement"],minimum_blocks=config["minimum_blocks"])}
                for model,fit in methods.items():
                    fits.append({"group":group,"scenario":sid,"cycles":cycles,"model":model,**fit})
                    prediction,_=oscillator_forecast(windows["past"],fit["omega"],horizons,damping=fit["damping"])
                    for j,horizon in enumerate(horizons):
                        rows.append({"group":group,"scenario":sid,"cycles":cycles,"budget_s":budget,
                                     "model":model,"horizon_s":horizon,
                                     **regression_metrics(windows["y"][:,j],prediction[:,j])})
        # Persist chronological development evidence before the next group is run.
        pd.DataFrame(rows).to_csv(args.output/"metrics.csv",index=False)
        (args.output/"fits.json").write_text(json.dumps(fits,indent=2))
    provenance["status"]="complete"
    (args.output/"provenance.json").write_text(json.dumps(provenance,indent=2))
    metrics=pd.DataFrame(rows)
    summary=metrics.groupby(["group","model","cycles","horizon_s"],as_index=False).mae_deg.mean()
    summary.to_csv(args.output/"summary.csv",index=False)
    print(summary.to_string(index=False))


if __name__=="__main__":
    main()
