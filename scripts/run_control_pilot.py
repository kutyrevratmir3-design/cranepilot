"""Small paired synthetic model-mismatch experiment; NO hardware control claims."""
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from cranepilot.control import PlantParameters, LinearMPC, simulate_episode
from cranepilot.metrics import paired_bootstrap


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--forecast-run",type=Path,default=Path("artifacts/runs/extrapolation_pilot"))
    p.add_argument("--output",type=Path,default=Path("artifacts/runs/control_pilot"))
    p.add_argument("--episodes",type=int,default=16)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    c=json.loads((args.forecast_run/"calibration_trolley_length_extrapolation_h41.json").read_text())["calibration"]
    rng=np.random.default_rng(20261102)
    contexts=[]
    for i in range(args.episodes):
        cable=float(rng.choice([.1,.2,.7,.8]))
        # Synthetic mismatch range, not an identified probability distribution.
        true_length=float(rng.uniform(.85,1.05)*cable+rng.uniform(.10,.22))
        contexts.append({"episode":i,"cable_m":cable,"seed":1000+i,
                         "plant":asdict(PlantParameters(true_length,actuator_tau=float(rng.uniform(.07,.14)),
                              damping=float(rng.uniform(.01,.07)),deadzone=float(rng.uniform(0,.04)),
                              acceleration_gain=float(rng.uniform(.9,1.1))))})
    config={"status":"running","scope":"simulation-only; exact state feedback; two MPC model lengths",
            "calibration":c,"contexts":contexts,"dt_s":.05,"duration_s":12.,"horizon_steps":40,
            "warning":"Does not evaluate the neural forecaster inside MPC; no certified state constraints."}
    (args.output/"provenance.json").write_text(json.dumps(config,indent=2))
    rows=[]
    for context in contexts:
        for name,length in [("nominal_mpc",context["cable_m"]),
                            ("calibrated_mpc",c["a"]*context["cable_m"]+c["b"])]:
            controller=LinearMPC(length)
            metrics,trace=simulate_episode(controller,PlantParameters(**context["plant"]),seed=context["seed"])
            rows.append({"episode":context["episode"],"model":name,"cable_m":context["cable_m"],**metrics})
            np.savez_compressed(args.output/f"episode_{context['episode']}_{name}.npz",**trace)
        pd.DataFrame(rows).to_csv(args.output/"metrics.csv",index=False)
        print(f"Completed paired control context {context['episode']+1}/{len(contexts)}",flush=True)
    table=pd.DataFrame(rows)
    paired=table.pivot(index="episode",columns="model",values="residual_rms_deg")
    statistics=paired_bootstrap(paired.calibrated_mpc-paired.nominal_mpc)
    (args.output/"paired_statistics.json").write_text(json.dumps(statistics,indent=2))
    config["status"]="complete"
    (args.output/"provenance.json").write_text(json.dumps(config,indent=2))
    print(table.groupby("model").mean(numeric_only=True).to_string())


if __name__=="__main__":
    main()

