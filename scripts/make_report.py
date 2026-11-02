"""Build paper figures, tables and a factual checkpoint from completed runs."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS={"persistence":"Persistence","ideal_oscillator":"Ideal oscillator",
        "calibrated_oscillator":"Calibrated oscillator","ols":"OLS","lpv":"LPV",
        "boosted_trees":"Boosted trees","mlp":"MLP ensemble","gp_residual":"GP residual",
        "phase_ridge_residual":"Phase ridge residual","phase_nn_residual":"Phase NN residual",
        "projected_phase_residual":"Projected phase residual"}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,default=Path("artifacts/runs/extrapolation_pilot"))
    p.add_argument("--output",type=Path,default=Path("results/latest"))
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    paper=Path("paper"); figures=paper/"figures"; figures.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((args.run/"provenance.json").read_text())
    if provenance["status"] != "complete":
        raise RuntimeError("Refusing to report an incomplete forecast run")
    metrics=pd.read_csv(args.run/"metrics.csv")
    metrics.to_csv(args.output/"forecast_metrics.csv",index=False)
    all_records=metrics[metrics.regime=="all"]
    summary=all_records.groupby(["model","horizon_s"])[["mae_deg","rmse_deg"]].mean().reset_index()
    summary.to_csv(args.output/"forecast_summary.csv",index=False)
    data=summary.pivot(index="model",columns="horizon_s",values="mae_deg")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "figure.dpi":130,"savefig.dpi":220,"axes.grid":True,"grid.alpha":.18})

    def save(fig,name):
        fig.tight_layout()
        fig.savefig(figures/f"{name}.png",facecolor="white",bbox_inches="tight")
        fig.savefig(figures/f"{name}.svg",facecolor="white",bbox_inches="tight")
        plt.close(fig)

    order=data[1.0].sort_values().index
    fig,ax=plt.subplots(figsize=(7.5,5))
    colors=["#176b87" if x=="calibrated_oscillator" else "#b85335" if x=="projected_phase_residual" else "#8d9baa" for x in order]
    ax.barh([LABELS[x] for x in order],data.loc[order,1.0],color=colors)
    ax.invert_yaxis(); ax.set_xlabel("MAE at 1.0 s (degrees)")
    ax.set_title("Length extrapolation: mean of four held-out records")
    save(fig,"forecast_comparison")

    fig,ax=plt.subplots(figsize=(7.3,4.3))
    chosen=["ols","mlp","calibrated_oscillator","phase_nn_residual","projected_phase_residual"]
    for name in chosen:
        ax.plot(data.columns,data.loc[name],marker="o",label=LABELS[name],lw=1.7)
    ax.set(xlabel="Forecast horizon (s)",ylabel="MAE (degrees)",
           title="Forecast accuracy across horizons; same causal angle history")
    ax.legend(ncols=2,fontsize=8)
    save(fig,"horizon_comparison")

    fig,axes=plt.subplots(1,2,figsize=(9,3.7),sharey=False)
    for ax,regime in zip(axes,["active","quiet"]):
        for name in ["calibrated_oscillator","phase_nn_residual","projected_phase_residual"]:
            d=metrics[(metrics.regime==regime)&(metrics.model==name)].groupby("horizon_s").mae_deg.mean()
            ax.plot(d.index,d.values,marker="o",label=LABELS[name])
        ax.set(title=f"{regime.capitalize()} at forecast origin",xlabel="Horizon (s)",ylabel="MAE (degrees)")
    axes[1].legend(fontsize=7)
    save(fig,"activity_comparison")

    fig,axes=plt.subplots(2,1,figsize=(8,5.5))
    horizons=provenance["config"]["horizons_steps"]
    j=horizons.index(100)
    for ax,sid in zip(axes,[1,8]):
        file=next(args.run.glob(f"predictions_*_s{sid}_h41.npz"))
        with np.load(file) as z:
            t=(z["k"]+100)*.01
            mask=t<14
            ax.plot(t[mask],np.rad2deg(z["target"][mask,j]),color="#222222",label="Measured target",lw=1)
            for name,color in [("calibrated_oscillator","#176b87"),("projected_phase_residual","#b85335")]:
                ax.plot(t[mask],np.rad2deg(z[name][mask,j]),label=LABELS[name],lw=.9,color=color,alpha=.85)
        ax.set(title=f"Scenario {sid}: 1 s forecast during early motion",ylabel="Angle (degrees)")
    axes[-1].set_xlabel("Target time (s)"); axes[0].legend(fontsize=8,ncols=3)
    save(fig,"forecast_traces")

    table=[r"\begin{tabular}{lrrr}",r"\toprule",r"Model & 0.1 s & 0.5 s & 1.0 s \\",r"\midrule"]
    for name in order:
        table.append(LABELS[name]+" & "+" & ".join(f"{data.loc[name,h]:.3f}" for h in [.1,.5,1.0])+r" \\")
    table.extend([r"\bottomrule",r"\end{tabular}"])
    (paper/"forecast_table.tex").write_text("\n".join(table)+"\n")

    overall={"status":"exploratory", "forecast_test_scenarios":sorted(all_records.scenario.unique().tolist()),
             "n_forecast_models":int(all_records.model.nunique()),
             "calibrated_mae_1s":float(data.loc["calibrated_oscillator",1.0]),
             "projected_mae_1s":float(data.loc["projected_phase_residual",1.0]),
             "calibrated_mae_0p1s":float(data.loc["calibrated_oscillator",.1]),
             "projected_mae_0p1s":float(data.loc["projected_phase_residual",.1]),
             "neural_seeds":provenance["config"]["seeds"],
             "config_sha256":provenance["config_sha256"]}
    baseline_dir=Path("artifacts/runs/loso_baselines")
    if (baseline_dir/"provenance.json").exists() and json.loads((baseline_dir/"provenance.json").read_text())["status"]=="complete":
        base=pd.read_csv(baseline_dir/"metrics.csv")
        bs=base[base.regime=="all"].groupby(["family","axis","history_samples","horizon_s","model"]).mae_deg.mean().reset_index()
        bs.to_csv(args.output/"all_family_baselines.csv",index=False)
        base.to_csv(args.output/"all_family_metrics.csv",index=False)
        overall["baseline_scenario_axis_folds"]=int(base.fold.nunique())
        overall["baseline_histories"]=sorted(base.history_samples.unique().tolist())
        fig,axes=plt.subplots(1,3,figsize=(10,3.3))
        for ax,family in zip(axes,["trolley","jib","simult"]):
            q=bs[(bs.family==family)&(bs.horizon_s==1.0)]
            for name in ["ols","calibrated_oscillator","phase_ridge_residual"]:
                z=q[q.model==name].groupby("history_samples").mae_deg.mean()
                ax.plot(z.index,z.values,marker="o",label=LABELS[name])
            ax.set(title=family.capitalize(),xlabel="History samples",ylabel="1 s MAE (degrees)",xticks=[3,41])
        axes[-1].legend(fontsize=7)
        save(fig,"all_family_baselines")
    neural_dir=Path("artifacts/runs/neural_loso")
    if (neural_dir/"provenance.json").exists() and json.loads((neural_dir/"provenance.json").read_text())["status"]=="complete":
        neural=pd.read_csv(neural_dir/"metrics.csv")
        neural.to_csv(args.output/"all_family_neural_metrics.csv",index=False)
        ns=neural[neural.regime=="all"].groupby(["family","axis","horizon_s","model"]).mae_deg.mean().reset_index()
        ns.to_csv(args.output/"all_family_neural_summary.csv",index=False)
        training=pd.read_csv(neural_dir/"training.csv")
        training.to_csv(args.output/"all_family_neural_training.csv",index=False)
        overall["neural_scenario_axis_folds"]=int(neural.fold.nunique())
        overall["trained_neural_networks"]=len(training)
        pivot=ns[ns.horizon_s==1.].pivot(index=["family","axis"],columns="model",values="mae_deg")
        t=[r"\begin{tabular}{llrrr}",r"\toprule",r"Family & Axis & Calibrated & GP residual & Projected NN \\",r"\midrule"]
        for (family,axis),row in pivot.iterrows():
            t.append(family+" & "+axis+" & "+" & ".join(f"{row[m]:.3f}" for m in
                ["calibrated_oscillator","gp_residual","projected_phase_residual"])+r" \\")
        t.extend([r"\bottomrule",r"\end{tabular}"])
        (paper/"all_family_neural_table.tex").write_text("\n".join(t)+"\n")
    adaptation_dir=Path("artifacts/runs/adaptation_pilot")
    if (adaptation_dir/"provenance.json").exists() and json.loads((adaptation_dir/"provenance.json").read_text())["status"]=="complete":
        adaptation=pd.read_csv(adaptation_dir/"metrics.csv")
        adaptation.to_csv(args.output/"adaptation_metrics.csv",index=False)
        aa=adaptation.groupby(["cycles","horizon_s"]).mae_deg.mean().unstack()
        fig,ax=plt.subplots(figsize=(6.5,3.5))
        for horizon in [.1,.5,1.]:
            ax.plot(aa.index,aa[horizon],marker="o",label=f"{horizon:g} s horizon")
        ax.set(xlabel="Calibration budget (prior oscillator periods)",ylabel="MAE (degrees)",
               title="Adaptive-physics pilot: same scored tail after 25 s")
        ax.legend(); save(fig,"adaptation_curve")
        overall["adaptation_1s_mae_by_cycles"]={str(int(k)):float(v) for k,v in aa[1.].items()}
    control_dir=Path("artifacts/runs/control_pilot")
    if (control_dir/"provenance.json").exists() and json.loads((control_dir/"provenance.json").read_text())["status"]=="complete":
        control=pd.read_csv(control_dir/"metrics.csv")
        control.to_csv(args.output/"control_metrics.csv",index=False)
        cs=control.groupby("model").mean(numeric_only=True)
        cs.to_csv(args.output/"control_summary.csv")
        overall["control_paired_contexts"]=int(control.episode.nunique())
        overall["control_summary"]=cs.to_dict(orient="index")
        overall["control_paired_statistics"]=json.loads((control_dir/"paired_statistics.json").read_text())
        fig,axes=plt.subplots(1,2,figsize=(8,3.5))
        for ax,metric,title in zip(axes,["residual_rms_deg","sway_ise_deg2_s"],["Residual sway (degrees)","Integrated sway (degrees squared s)"]):
            pair=control.pivot(index="episode",columns="model",values=metric)
            for _,row in pair.iterrows():
                ax.plot([0,1],[row.nominal_mpc,row.calibrated_mpc],color="#b7c0c7",lw=.8,marker="o",ms=3)
            ax.plot([0,1],[pair.nominal_mpc.mean(),pair.calibrated_mpc.mean()],color="#176b87",lw=2.5,marker="o",label="Mean")
            ax.set(xticks=[0,1],xticklabels=["Nominal MPC","Calibrated MPC"],ylabel=title,title="16 paired synthetic contexts")
        axes[0].legend()
        save(fig,"control_paired")
        t=[r"\begin{tabular}{lrrrr}",r"\toprule",r"Controller & Residual RMS & Sway integral & Settle time & p99 solve \\",
           r" & (deg) & (deg$^2$s) & (s) & (ms) \\",r"\midrule"]
        for name in ["nominal_mpc","calibrated_mpc"]:
            row=cs.loc[name]
            t.append(name.replace("_"," ")+" & "+" & ".join(f"{row[m]:.3f}" for m in
                ["residual_rms_deg","sway_ise_deg2_s","settling_time_s","solve_p99_ms"])+r" \\")
        t.extend([r"\bottomrule",r"\end{tabular}"])
        (paper/"control_table.tex").write_text("\n".join(t)+"\n")
    (args.output/"summary.json").write_text(json.dumps(overall,indent=2))
    (args.output/"source_snapshot.json").write_text(json.dumps({"recorded_after_pilot":True,
        "sha256":{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path("src").rglob("*.py")}},indent=2))
    print(json.dumps(overall,indent=2))


if __name__=="__main__":
    main()

