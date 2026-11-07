"""Retain every off-grid subgroup, including adaptive-physics failures."""
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cranepilot.data import sha256


def main():
    run = Path('artifacts/runs/combined_offgrid_20261107')
    out = Path('results/combined_stress_20261107/offgrid')
    if out.exists(): raise FileExistsError('Preserve prior report')
    provenance = json.loads((run/'provenance.json').read_text())
    if provenance['status'] != 'complete': raise ValueError('Completed run required')
    for name,digest in provenance['result_hashes'].items():
        if sha256(run/name) != digest: raise ValueError(f'Changed result: {name}')
    out.mkdir(parents=True)
    groups = pd.read_csv(run/'group_summary.csv')
    overall = groups.groupby(['fold','model']).mean_mae_deg.mean().unstack('model')
    overall.to_csv(out/'overall.csv')
    pairs = groups.pivot(index=['fold','condition','measurement'],columns='model',values='mean_mae_deg').reset_index()
    pairs['adaptive_change_percent_vs_combined'] = 100*(pairs.adaptive_physics/pairs.combined-1)
    pairs.to_csv(out/'paired_groups.csv',index=False)
    failures = pairs[pairs.adaptive_physics>pairs.combined]
    failures.to_csv(out/'adaptive_failure_groups.csv',index=False)
    for name in ['group_summary.csv','gate_statistics.csv','verification.json']:
        shutil.copy2(run/name,out/name)
    info = {'parent_provenance_sha256':sha256(run/'provenance.json'), 'posthoc_hypothesis_test':True,
            'cases_per_fold':512,'same_targets_between_folds':False,'model_parameters_changed':False,
            'adaptive_wins_vs_combined':{f:int((p.adaptive_physics<p.combined).sum()) for f,p in pairs.groupby('fold')},
            'groups_per_fold':32,'adaptive_failure_groups':len(failures),'real_record_adaptive_evidence':False}
    (out/'report.json').write_text(json.dumps(info,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    fig,ax = plt.subplots(figsize=(9,4.6))
    order = ['primary','short_transfer','long_transfer']; x=np.arange(3)
    for i,(name,label,color) in enumerate([('incumbent','Исходная сеть','#6f8293'),('combined','Симметрия + gate','#408c81'),('adaptive_physics','Адаптивная физика','#bd713d')]):
        values=overall.loc[order,name].to_numpy();xx=x+(i-1)*.24
        ax.bar(xx,values,width=.22,label=label,color=color)
        for j,v in zip(xx,values): ax.text(j,v+.025,f'{v:.3f}',ha='center',fontsize=10)
    ax.set_xticks(x,['Основной набор','Короткий перенос','Длинный перенос'])
    ax.set_ylabel('MAE угла, °');ax.set_ylim(0,1.85);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.legend(frameon=False,ncol=3,loc='upper center',fontsize=9)
    ax.set_title('Новые частоты и затухание между узлами фиксированной сетки')
    fig.text(.04,.025,'512 случаев на набор. Цели между наборами различаются. Симуляция; методы не менялись.',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(out/'offgrid_findings.png',dpi=180);fig.savefig(out/'offgrid_findings.svg');plt.close(fig)
    print(overall.to_string());print(json.dumps(info,indent=2))


if __name__ == '__main__': main()
