"""Equal-scoring-tail calibration curve; all budgets evaluated after t=25 s."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from cranepilot.data import load_dataset, make_windows
from cranepilot.physics import LengthCalibration, oscillator_forecast
from cranepilot.adaptation import fit_prefix_oscillator
from cranepilot.metrics import regression_metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=Path("data/processed"))
    p.add_argument("--output",type=Path,default=Path("artifacts/runs/adaptation_pilot"))
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    scenarios=load_dataset(args.data)
    cal=LengthCalibration.fit(scenarios,[3,5,6],"theta")
    provenance={"status":"running","train":[3,5,6],"validation":[4],"test":[1,2,7,8],
                "prefix_period_budgets":[0,1,3,5,10],"score_start_s":25.0,"history_samples":41,
                "information":"angle and measured trolley motion inside calibration prefix; angle-only scored forecasts",
                "scope":"retrospective adaptive physical reference; not a new method claim"}
    (args.output/"provenance.json").write_text(json.dumps(provenance,indent=2))
    rows=[]; fits=[]
    for sid in provenance["test"]:
        s=scenarios[sid]; prior=float(cal.omega(s.signals["L"][0])); period=2*np.pi/prior
        windows=make_windows(s,"theta",[10,50,100],history=41,stride=2,start_s=25.)
        for cycles in provenance["prefix_period_budgets"]:
            budget=cycles*period
            if budget>25.-.4:
                raise ValueError("Calibration must end before all scored origins and their history")
            fit=fit_prefix_oscillator(s.angle(),s.signals["x"],prior,budget)
            fits.append({"scenario":sid,"cycles":cycles,"prior_omega":prior,**fit})
            prediction,_=oscillator_forecast(windows["past"],fit["omega"],[.1,.5,1.],damping=fit["damping"])
            for j,h in enumerate([.1,.5,1.]):
                rows.append({"scenario":sid,"cycles":cycles,"budget_s":budget,"horizon_s":h,
                             **regression_metrics(windows["y"][:,j],prediction[:,j])})
    pd.DataFrame(rows).to_csv(args.output/"metrics.csv",index=False)
    (args.output/"fits.json").write_text(json.dumps(fits,indent=2))
    provenance["status"]="complete"
    (args.output/"provenance.json").write_text(json.dumps(provenance,indent=2))
    print(pd.DataFrame(rows).groupby(["cycles","horizon_s"]).mae_deg.mean().unstack().round(4).to_string())


if __name__=="__main__":
    main()

