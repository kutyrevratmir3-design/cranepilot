"""Matched-origin report for every completed force-aware variant."""
from pathlib import Path
import argparse
import itertools
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=Path('artifacts/runs/forceaware_campaign_v2'))
    args=p.parse_args();run=args.run;out=Path('results/latest')
    provenance=json.loads((run/'provenance.json').read_text())
    if provenance['status']!='complete':raise RuntimeError('Completed evaluation required')
    cfg=provenance['config'];metrics=pd.read_csv(run/'metrics.csv')
    previous=pd.read_csv(out/'innovation_comparison_metrics.csv')
    old_names=['passive_m4','uniform_control','driven','observer','gp_action_residual','mlp_action_residual']
    previous=previous.query('model in @old_names')
    for sid in cfg['evaluation']:
        old=np.load(Path('artifacts/runs/innovation_development')/f'predictions_s{sid}.npz')
        new=np.load(run/f'predictions_s{sid}.npz')
        if not np.array_equal(old['k'],new['k']):raise ValueError('Forecast origins differ')
        np.testing.assert_allclose(old['target'],new['target'],atol=0,rtol=0)
    metrics=pd.concat([metrics,previous],ignore_index=True)
    metrics.to_csv(out/'forceaware_metrics.csv',index=False)
    summary=metrics.groupby(['group','model','regime','horizon_s'],as_index=False)[['mae_deg','rmse_deg']].mean()
    summary.to_csv(out/'forceaware_summary.csv',index=False)
    val=summary.query("group == 'validation' and regime == 'all'").groupby('model').mae_deg.mean()
    candidate=val.loc[list(cfg['variants'])].idxmin()
    paired=metrics.query("group == 'evaluation' and regime == 'all'").groupby(['scenario','model']).mae_deg.mean().unstack()
    difference=(paired[candidate]-paired.full_kernel).to_numpy()
    bootstrap=np.array([np.mean(difference[list(i)]) for i in itertools.product(range(4),repeat=4)])
    aggregate=summary.query("group == 'evaluation' and regime == 'all'").groupby('model').mae_deg.mean()
    report={'scope':cfg['scope'],'best_neural_by_validation':candidate,
        'primary_macro_mae_deg':aggregate.to_dict(),
        'worst_scenario_macro_mae_deg':paired.max().to_dict(),
        'candidate_minus_full_kernel_per_record_deg':difference.tolist(),
        'four_record_descriptive_bootstrap_95pct':np.quantile(bootstrap,[.025,.975]).tolist(),
        'bootstrap_caution':'Four retrospective records only; not a reliable population confidence guarantee.',
        'all_seeds_and_variants_retained':True,
        'failed_initial_run':'forceaware_campaign failed before fitting due to negative-stride tensor conversion; fixed in v2, original failure retained',
        'publication':'local only; not pushed to GitHub'}
    (out/'forceaware_report.json').write_text(json.dumps(report,indent=2))
    for name in ['provenance.json','validation_seeds.csv','kernel_validation.csv','training.csv','diagnostics.csv','timing.csv']:
        shutil.copyfile(run/name,out/('forceaware_'+name))
    names={'driven':'Previous driven NN','gp_action_residual':'Previous GP (512)',
        'full_kernel':'Stronger full GP (1024)','phase_kernel':'Physical-feature GP (1024)',candidate:'New NN: '+candidate}
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,regime,title in zip(axes,['all','active_at_origin'],['All forecast origins','Active at origin']):
        for model,label in names.items():
            part=summary.query("group == 'evaluation' and regime == @regime and model == @model")
            ax.plot(part.horizon_s,part.mae_deg,'o-',label=label,ms=3,lw=1.2)
        ax.set(title=title,xlabel='Forecast horizon (s)',ylabel='Mean scenario MAE (degrees)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    for ext in ['png','svg']:fig.savefig(f'paper/figures/forceaware_comparison.{ext}',dpi=190)
    plt.close(fig)
    print(summary.query("regime == 'all'").pivot(index=['group','model'],columns='horizon_s',values='mae_deg').round(5).to_string())
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
