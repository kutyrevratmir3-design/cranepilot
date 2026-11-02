"""Report every completed variant, preserving unfavourable results."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'results/latest'; run=root/'artifacts/runs/innovation_development'
    control=root/'artifacts/runs/uniform_passive_control'
    for r in (run,control):
        if json.loads((r/'provenance.json').read_text())['status']!='complete':raise RuntimeError('Incomplete run')
    old=pd.read_csv(out/'passive_rollout_metrics.csv').rename(columns={'group':'original_group'})
    old['group']=old.original_group.replace({'test':'evaluation'})
    frames=[old.drop(columns='original_group'),pd.read_csv(run/'evaluation_metrics.csv'),pd.read_csv(control/'evaluation_metrics.csv')]
    metrics=pd.concat(frames,ignore_index=True)
    summary=metrics.groupby(['group','model','regime','horizon_s'],as_index=False)[['mae_deg','rmse_deg']].mean()
    metrics.to_csv(out/'innovation_comparison_metrics.csv',index=False)
    summary.to_csv(out/'innovation_comparison_summary.csv',index=False)
    for source,destination in [(run/'training.csv','innovation_training.csv'),(run/'validation_seeds.csv','innovation_validation_seeds.csv'),
        (run/'provenance.json','innovation_provenance.json'),(run/'diagnostics.csv','innovation_diagnostics.csv'),
        (control/'provenance.json','uniform_control_provenance.json'),(control/'training.csv','uniform_control_training.csv')]:shutil.copyfile(source,out/destination)
    names={'action_physics':'Action physics','passive_m4':'Original PEPM',
        'gp_action_residual':'GP reference','uniform_control':'PEPM, matched training',
        'observer':'Refined observer','driven':'Observer + driven innovation'}
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,regime,title in zip(axes,['all','active_at_origin'],['All forecast origins','Active at origin']):
        for m,label in names.items():
            part=summary.query("group == 'evaluation' and regime == @regime and model == @m")
            ax.plot(part.horizon_s,part.mae_deg,'o-',label=label,ms=3,lw=1.2)
        ax.set(title=title,xlabel='Forecast horizon (s)',ylabel='Mean scenario MAE (degrees)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    for ext in ['png','svg']:fig.savefig(root/f'paper/figures/innovation_comparison.{ext}',dpi=190)
    plt.close(fig)
    table=summary.query("group == 'evaluation' and regime == 'all'").pivot(index='model',columns='horizon_s',values='mae_deg')
    details=metrics.query("group == 'evaluation' and regime == 'all'").pivot(index=['scenario','horizon_s'],columns='model',values='mae_deg')
    report={'scope':'retrospective development; same measurement and prescribed-command information; four records',
        'tests_passed':28,'fits_this_campaign':9,'seeds_per_variant':3,
        'mean_across_seven_horizons_deg':{m:float(table.loc[m].mean()) for m in names},
        'improvement_vs_original_percent':{m:float(100*(1-table.loc[m].mean()/table.loc['passive_m4'].mean())) for m in ['uniform_control','observer','driven']},
        'wins_of_28_scenario_horizon_pairs_vs_gp':{m:int((details[m]<details.gp_action_residual).sum()) for m in ['uniform_control','observer','driven']},
        'caution':'Model variants and protocol changed; matched-training control separates part of that effect. No new independent physical trials. Learned innovation is an additional energy port, not a physical force identification or closed-loop safety claim.'}
    (out/'innovation_report.json').write_text(json.dumps(report,indent=2))
    print(table.round(5).to_string());print(json.dumps(report,indent=2))


if __name__=='__main__':main()
