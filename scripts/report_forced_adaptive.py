"""Report every frozen forced-adaptation strategy and operational acceptance gate."""
import argparse
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from cranepilot.data import sha256

METHODS = ['incumbent','soft_symmetry','combined','physical','damping_only','adaptive',
           'adaptive_strong','adaptive_fine','residual_transfer','residual_gate','soft_adaptive_gate']
LABELS = {'incumbent':'Исходная сеть','soft_symmetry':'Частичная симметрия','combined':'Прежняя комбинация',
          'physical':'Номинальная физика','damping_only':'Только затухание','adaptive':'Адаптивная физика',
          'adaptive_strong':'Сильный штраф','adaptive_fine':'Мелкая сетка',
          'residual_transfer':'Перенос поправки','residual_gate':'Поправка × gate','soft_adaptive_gate':'Симметрия + адапт. gate'}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def compare(frame, keys, references):
    result = frame.copy()
    for name in references:
        reference = frame[frame.model == name][keys+['mae_deg']].rename(columns={'mae_deg':f'{name}_mae_deg'})
        result = result.merge(reference,on=keys,how='left',validate='many_to_one')
        if result[f'{name}_mae_deg'].isna().any(): raise ValueError(f'Missing reference {name}')
        result[f'change_vs_{name}_percent'] = 100*(result.mae_deg/result[f'{name}_mae_deg']-1)
    return result


def aggregate_cases(cases, keys):
    return cases.groupby(keys,as_index=False).agg(mae_deg=('mae_deg','mean'),
        median_case_mae_deg=('mae_deg','median'),case_p95_mae_deg=('mae_deg',lambda x:float(x.quantile(.95))),
        worst_case_mae_deg=('mae_deg','max'),cases=('mae_deg','size'))


def make_figures(out, real, sim, cfg, selection):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    cols=[(f,r) for f in cfg['folds'] for r in ['all','active']]
    matrix=np.array([[float(real[(real.model==m)&(real.fold==f)&(real.regime==r)].change_vs_incumbent_percent.iloc[0])
                      for f,r in cols] for m in METHODS])
    fig,ax=plt.subplots(figsize=(10.5,6.8))
    norm=TwoSlopeNorm(vmin=-30,vcenter=0,vmax=30)
    im=ax.imshow(matrix,cmap='RdBu_r',norm=norm,aspect='auto')
    for i,m in enumerate(METHODS):
        for j,(f,r) in enumerate(cols):
            mark='*' if selection['folds'][f]['model']==m else ''
            ax.text(j,i,f'{matrix[i,j]:+.1f}%{mark}',ha='center',va='center',fontsize=9,
                    color='white' if abs(matrix[i,j])>=20 else '#22313a')
    ax.set_yticks(range(len(METHODS)),[LABELS[m]+(' †' if selection['common']['model']==m else '') for m in METHODS])
    ax.set_xticks(range(6),['Основное\nвсё','Основное\nактивное','Короткие\nвсё','Короткие\nактивное','Длинные\nвсё','Длинные\nактивное'])
    ax.set_title('Вынужденная адаптация: все стратегии на реальных записях',pad=16)
    fig.colorbar(im,ax=ax,shrink=.7,label='Изменение MAE, %; цвет ограничен ±30%')
    fig.text(.02,.025,'Минус — улучшение. * Локальный выбор по валидации; † общий выбор. Оценка ретроспективная.',fontsize=9)
    fig.tight_layout(rect=[0,.055,1,1]);fig.savefig(out/'real_strategies.png',dpi=180);fig.savefig(out/'real_strategies.svg');plt.close(fig)

    # Direct acceleration, nonzero trolley drive and matched future gain only.
    subset=sim[(sim.fold=='primary')&(sim.drive_amplitude_m>0)&np.isclose(sim.future_acceleration_gain,1.)]
    names=['incumbent','physical','adaptive','adaptive_strong','adaptive_fine','residual_transfer','residual_gate','soft_adaptive_gate']
    colors=['#657c90','#b3b7ab','#247c70','#68a49a','#afc8bf','#bd713d','#cf9b70','#8e5475']
    fig,axes=plt.subplots(1,2,figsize=(13,5.3),sharey=True)
    for ax,law,title in zip(axes,['effective','geometric'],['Связь ω²/g','Связь 1/L']):
        part=subset[subset.force_law==law]
        x=np.arange(2)
        for i,(name,color) in enumerate(zip(names,colors)):
            vals=[float(part[(part.model==name)&(part.measurement==mode)].mae_deg.iloc[0]) for mode in ['exact','noisy_quantized']]
            ax.bar(x+(i-3.5)*.105,vals,width=.1,color=color,label=LABELS[name])
        ax.set_xticks(x,['Точная история','Шум и квантование'])
        ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.set_xlabel('Ненулевое движение; будущий коэффициент = 1')
    axes[0].set_ylabel('MAE угла, °')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.035),fontsize=9)
    fig.suptitle('Свежая вынужденная симуляция: основной набор, разные законы связи',fontsize=13)
    fig.text(.02,.005,'Ускорение задано напрямую всем методам. Это угловой диагностический тест; актор не проверяется.',fontsize=9)
    fig.tight_layout(rect=[0,.20,1,.94]);fig.savefig(out/'forced_simulation.png',dpi=180);fig.savefig(out/'forced_simulation.svg');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('artifacts/runs/forced_adaptive_20261107'))
    p.add_argument('--output',type=Path,default=Path('results/forced_adaptive_20261107'))
    args=p.parse_args();run,out=args.run,args.output
    provenance=json.loads((run/'provenance.json').read_text());cfg=provenance['config']
    if provenance['status']!='complete':raise ValueError('Completed evaluation required')
    for name,digest in provenance['result_hashes'].items():
        if sha256(run/name)!=digest:raise ValueError(f'Changed result: {name}')
    if (out/'report.json').exists():raise FileExistsError('Preserve completed report')
    out.mkdir(parents=True,exist_ok=True)
    selection=json.loads((run/'selection.json').read_text())
    if sha256(run/'selection.json')!=provenance['selection_sha256']:raise ValueError('Selection changed')
    real=pd.concat([pd.read_csv(run/'select_metrics.csv'),pd.read_csv(run/'evaluate_metrics.csv')],ignore_index=True)
    keys=['fold','group','scenario','model','regime']
    if real.duplicated(keys+['horizon_s']).any() or not real.groupby(keys).horizon_s.nunique().eq(7).all():
        raise ValueError('Unique complete seven-horizon real metrics required')
    real.to_csv(out/'real_metrics_all.csv',index=False)
    records=real.groupby(keys+['extrapolation'],as_index=False).mae_deg.mean()
    records.to_csv(out/'real_record_means.csv',index=False)
    evaluated=real[(real.group=='evaluation')&real.extrapolation.astype(bool)]
    evaluation_records=records[(records.group=='evaluation')&records.extrapolation.astype(bool)]
    summary=evaluation_records.groupby(['fold','model','regime'],as_index=False).agg(mae_deg=('mae_deg','mean'),scenarios=('scenario','nunique'))
    summary=compare(summary,['fold','regime'],['incumbent','soft_symmetry','combined','physical'])
    summary.to_csv(out/'real_summary.csv',index=False)
    evaluated.groupby(['fold','model','regime','horizon_s'],as_index=False).mae_deg.mean().to_csv(out/'real_horizons.csv',index=False)
    load=records[records.group=='load_stress'].groupby(['fold','model','regime'],as_index=False).mae_deg.mean()
    load=compare(load,['fold','regime'],['incumbent','soft_symmetry','combined'])
    load.to_csv(out/'load_stress.csv',index=False)
    interpolation=records[(records.group=='evaluation')&~records.extrapolation.astype(bool)]
    interpolation.to_csv(out/'interpolation_record_means.csv',index=False)

    simulation=pd.read_csv(run/'simulation_metrics.csv')
    case_keys=['fold','model','force_law','drive_amplitude_m','future_acceleration_gain','measurement',
               'omega_ratio','damping_per_s','length_m','amplitude_deg','phase_rad','trajectory_id']
    if simulation.duplicated(case_keys+['horizon_s']).any() or not simulation.groupby(case_keys).horizon_s.nunique().eq(7).all():
        raise ValueError('Unique complete seven-horizon simulation cases required')
    cases=simulation.groupby(case_keys,as_index=False).absolute_error_deg.mean().rename(columns={'absolute_error_deg':'mae_deg'})
    cases.to_csv(out/'simulation_case_means.csv',index=False)
    groups=['fold','model','force_law','drive_amplitude_m','future_acceleration_gain','measurement']
    sim=aggregate_cases(cases,groups)
    sim=compare(sim,[n for n in groups if n!='model'],['incumbent','soft_symmetry','combined','physical','adaptive'])
    sim.to_csv(out/'simulation_groups.csv',index=False)
    aggregate_cases(cases,groups+['omega_ratio','damping_per_s']).to_csv(out/'simulation_frequency_damping.csv',index=False)
    aggregate_cases(cases,groups+['length_m']).to_csv(out/'simulation_lengths.csv',index=False)
    simulation.groupby(groups+['horizon_s'],as_index=False).absolute_error_deg.mean().to_csv(out/'simulation_horizons.csv',index=False)
    sim.sort_values('mae_deg',ascending=False).groupby(['fold','model'],sort=False).head(3).to_csv(out/'simulation_worst_groups.csv',index=False)

    estimates=pd.concat([pd.read_csv(run/'select_parameter_estimates.csv'),pd.read_csv(run/'evaluate_parameter_estimates.csv')],ignore_index=True)
    diag=[]
    for (fold,model,group),frame in estimates.groupby(['fold','model','group']):
        diag.append({'fold':fold,'model':model,'group':group,'packets':len(frame),
                     'boundary_rate_applicable':model!='damping_only',
                     'ratio_boundary_rate':float(frame.ratio_boundary_hit.mean()) if model!='damping_only' else np.nan,
                     'ratio_mean':float(frame.frequency_ratio.mean()),'ratio_std':float(frame.frequency_ratio.std(ddof=0)),
                     'ratio_min':float(frame.frequency_ratio.min()),'ratio_max':float(frame.frequency_ratio.max()),
                     'damping_mean_per_s':float(frame.damping_per_s.mean()),'damping_max_rate':float((frame.damping_per_s==max(cfg['variants'][model]['dampings'])).mean()),
                     'history_rms_median_rad':float(frame.history_rms_rad.median()),
                     'score_gap_median_rad2':float(frame.score_gap_rad2.median()),
                     'zero_score_gap_rate':float((frame.score_gap_rad2==0).mean())})
    pd.DataFrame(diag).to_csv(out/'parameter_diagnostics.csv',index=False)
    histogram=estimates.groupby(['fold','model','group','damping_per_s']).size().rename('packets').reset_index()
    histogram['fraction']=histogram.packets/histogram.groupby(['fold','model','group']).packets.transform('sum')
    histogram.to_csv(out/'damping_histogram.csv',index=False)
    ratio_histogram=estimates.groupby(['fold','model','group','frequency_ratio']).size().rename('packets').reset_index()
    ratio_histogram['fraction']=ratio_histogram.packets/ratio_histogram.groupby(['fold','model','group']).packets.transform('sum')
    ratio_histogram.to_csv(out/'frequency_histogram.csv',index=False)

    acceptance=[]
    for fold in cfg['folds']:
        reference='soft_symmetry' if fold=='primary' else 'incumbent'
        threshold=cfg['acceptance']['primary_all_active_vs_soft' if fold=='primary' else 'transfer_all_active_vs_incumbent']
        for regime in ['all','active']:
            row=summary[(summary.fold==fold)&(summary.model=='common')&(summary.regime==regime)]
            if len(row)!=1:raise ValueError('Missing common-strategy acceptance cell')
            value=row.iloc[0];ratio=float(value.mae_deg/value[reference+'_mae_deg'])
            acceptance.append({'scope':'evaluation','fold':fold,'regime':regime,'reference':reference,'ratio':ratio,'maximum_ratio':threshold,'passed':bool(ratio<=threshold)})
    for regime in ['all','active']:
        row=load[(load.fold=='primary')&(load.model=='common')&(load.regime==regime)]
        if len(row)!=1:raise ValueError('Missing primary load acceptance cell')
        value=row.iloc[0];ratio=float(value.mae_deg/value.incumbent_mae_deg);threshold=cfg['acceptance']['load_all_active_vs_incumbent']
        acceptance.append({'scope':'load_stress','fold':'primary','regime':regime,'reference':'incumbent','ratio':ratio,'maximum_ratio':threshold,'passed':bool(ratio<=threshold)})
    verdict={'common_model':selection['common']['model'],'cells':acceptance,'passed':all(x['passed'] for x in acceptance),
             'operational_thresholds_not_statistical_bounds':True}
    write(out/'acceptance.json',verdict)
    for name in ['selection.json','select_verification.json','evaluate_verification.json','select_timings.csv','evaluate_timings.csv']:
        shutil.copy2(run/name,out/name)
    make_figures(out,summary,sim,cfg,selection)
    info={'source_provenance_sha256':sha256(run/'provenance.json'),'report_source_sha256':sha256(Path(__file__)),
          'selection':selection,'acceptance_passed':verdict['passed'],'acceptance_failed_cells':[x for x in acceptance if not x['passed']],
          'real_records_retrospective':True,'neural_training':False,'simulation_actor_bypassed':True,
          'simulation_targets_common_between_folds':False,'hardware_evidence':False,
          'methods':METHODS,'case_definition':'mean absolute error over seven horizons; modes/gain/force laws kept separate',
          'simulation_cases_per_fold':{f:int(p[['trajectory_id','measurement']].drop_duplicates().shape[0]) for f,p in cases.groupby('fold')}}
    write(out/'report.json',info)
    print(summary[summary.model.isin(['selected','common','adaptive','adaptive_fine','adaptive_strong'])].to_string(index=False))
    print('ACCEPTANCE',json.dumps(verdict,indent=2))


if __name__=='__main__':main()
