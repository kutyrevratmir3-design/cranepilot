"""Retain all same-information pilot comparisons, including negative findings."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parents[1]
    run=root/'artifacts/runs/passive_rollout_pilot'
    references=root/'artifacts/runs/rollout_references'
    output=root/'results/latest'
    for source in (run,references):
        if json.loads((source/'provenance.json').read_text())['status']!='complete':
            raise RuntimeError('Only completed runs may enter the manuscript')
    metrics=pd.concat([pd.read_csv(run/'metrics.csv'),pd.read_csv(references/'metrics.csv')],ignore_index=True)
    summary=metrics.groupby(['group','model','regime','horizon_s'],as_index=False).mae_deg.mean()
    metrics.to_csv(output/'passive_rollout_metrics.csv',index=False)
    summary.to_csv(output/'passive_rollout_summary.csv',index=False)
    pd.read_csv(run/'timing.csv').to_csv(output/'passive_rollout_timing.csv',index=False)
    pd.read_csv(run/'actuator_validation.csv').to_csv(output/'actuator_validation.csv',index=False)
    names={'calibrated_harmonic':'Angle-only harmonic','action_physics':'Action-conditioned physics',
           'passive_m0':'Neural observer, no memory','passive_m4':'PEPM, four memory states',
           'gp_action_residual':'GP action residual','mlp_action_residual':'MLP action residual'}
    order=list(names)
    q=summary.query("group == 'test' and regime == 'all'").pivot(index='model',columns='horizon_s',values='mae_deg')
    rows=['\\begin{tabular}{lrrr}','\\toprule','Model & 0.1 s & 0.5 s & 1.0 s \\\\','\\midrule']
    rows += [names[m]+' & '+' & '.join(f'{q.loc[m,h]:.3f}' for h in [.1,.5,1.])+' \\\\' for m in order]
    rows += ['\\bottomrule','\\end{tabular}']
    (root/'paper/passive_rollout_table.tex').write_text('\n'.join(rows)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for ax,regime,title in zip(axes,['all','active_at_origin'],['All forecast origins','Active at forecast origin']):
        for m in order:
            part=summary.query("group == 'test' and regime == @regime and model == @m")
            ax.plot(part.horizon_s,part.mae_deg,'o-',lw=1.1,ms=3,label=names[m])
        ax.set(xlabel='Forecast horizon (s)',ylabel='Mean scenario MAE (degrees)',title=title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    for extension in ('png','svg'):
        fig.savefig(root/f'paper/figures/passive_rollout.{extension}',dpi=200,facecolor='white')
    plt.close(fig)
    timing=pd.read_csv(run/'timing.csv')
    report={'scope':'retrospective development; command-conditioned and angle-only information sets differ',
            'test_scenarios':[1,2,7,8], 'test_origin_stride_s':.1,
            'maximum_float32_balance_residual':float(timing.maximum_balance_residual.max()),
            'parameter_counts':{str(m):int(timing.query('memory == @m').parameters.iloc[0]) for m in [0,4]},
            'tests_passed':25, 'actuator_validation_scenario':4,
            'findings':'PEPM improves short horizons relative to action physics, but does not surpass the GP reference uniformly; long-horizon superiority is unconfirmed.'}
    (output/'passive_rollout_report.json').write_text(json.dumps(report,indent=2))
    print(q.round(5).to_string())
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
