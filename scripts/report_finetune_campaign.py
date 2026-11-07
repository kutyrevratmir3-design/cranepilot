"""Report the frozen fine-tuning campaign, preserving all strategies and tradeoffs."""
import argparse
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cranepilot.data import load_dataset, sha256
from cranepilot.physics import LengthCalibration
from cranepilot.rollout import command_windows, HORIZON_INDEX, HORIZONS_S

CONTROLS = ['incumbent', 'soft_symmetry', 'combined', 'physical', 'previous_local']
ALIASES = ['selected', 'common']
REFERENCES = ['incumbent', 'soft_symmetry', 'previous_local', 'continued']
RISK_COLUMNS = ['mean_packet_peak_deg', 'packet_mae_p95_deg', 'tail_mae_deg', 'normalized_increment_mae_deg']
SEED_NOTE = 'Descriptive variation across the three retained starting seeds; not confidence intervals or independent hardware repetitions.'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def verify_mapping(mapping, base):
    if not mapping: raise ValueError('Empty evidence mapping')
    for name, digest in mapping.items():
        if sha256(base/name.replace('\\', '/')) != digest:
            raise ValueError(f'Changed evidence: {base}/{name}')


def verify_evidence(root, run):
    provenance = json.loads((run/'provenance.json').read_text())
    if provenance['status'] != 'complete': raise ValueError('Completed evaluation required')
    counts = {}
    for key, base in [('guarded_hashes', root), ('checkpoint_hashes', run), ('result_hashes', run)]:
        verify_mapping(provenance[key], base); counts[key] = len(provenance[key])
    actual = {str(p.relative_to(run)) for p in run.rglob('*') if p.is_file() and p.name != 'provenance.json'}
    if actual != set(provenance['result_hashes']): raise ValueError('Completed result manifest membership changed')
    if sha256(run/'selection.json') != provenance['selection_sha256']: raise ValueError('Selection changed')
    cfg = provenance['config']
    expected = {f'{fold}/{name}_seed{seed}.pt' for fold in cfg['folds'] for name in cfg['variants'] for seed in cfg['seeds']}
    if set(provenance['checkpoint_hashes']) != expected or len(expected) != 90:
        raise ValueError('Expected exactly 90 fitted seed checkpoints')
    for key in ['source_run', 'broad_run', 'parent_run']:
        directory = root/cfg[key]; previous = json.loads((directory/'provenance.json').read_text())
        if previous['status'] != 'complete': raise ValueError(f'Incomplete parent: {key}')
        verify_mapping(previous['checkpoint_hashes'], directory)
        if key == 'parent_run': verify_mapping(previous['result_hashes'], directory)
        counts[key+'_checkpoints'] = len(previous['checkpoint_hashes'])
    for stage in ['train', 'evaluate']:
        verification = json.loads((run/f'{stage}_verification.json').read_text())
        if verification['passed'] is not True: raise ValueError(f'Failed stage checks: {stage}')
        counts[stage+'_checks'] = len(verification['checks'])
    return provenance, counts


def flag(series):
    result = series.astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})
    if result.isna().any(): raise ValueError('Invalid extrapolation flag')
    return result.astype(bool)


def scope(frame):
    frame = frame.copy(); frame['extrapolation'] = flag(frame.extrapolation)
    frame['scope'] = np.where(frame.group == 'evaluation',
        np.where(frame.extrapolation, 'evaluation_extrapolation', 'evaluation_interpolation'), frame.group)
    return frame


def compare(frame, keys, value='mae_deg'):
    result = frame.copy()
    for name in REFERENCES:
        ref = frame[frame.model == name][keys+[value]].rename(columns={value:name+'_'+value})
        result = result.merge(ref, on=keys, how='left', validate='many_to_one')
        denominator = result[name+'_'+value]
        if denominator.isna().any() or (denominator <= 0).any(): raise ValueError(f'Missing reference {name}')
        result['change_vs_'+name+'_percent'] = 100*(result[value]/denominator-1)
    return result


def check_horizons(frame, keys, value):
    if frame.empty or frame.duplicated(keys+['horizon_s']).any(): raise ValueError('Empty or duplicate horizon cells')
    if not np.isfinite(frame[value]).all() or (frame[value] < 0).any(): raise ValueError('Invalid errors')
    for _, part in frame.groupby(keys, dropna=False):
        if len(part) != len(HORIZONS_S) or not np.allclose(sorted(part.horizon_s), HORIZONS_S):
            raise ValueError('Expected exactly HORIZONS_S for every metric cell')


def read_real(run, cfg, out, selection):
    methods = CONTROLS+list(cfg['variants']); metrics=[]; risks=[]
    for stage in ['train','evaluate']:
        for target, suffix in [(metrics,'metrics'),(risks,'risk_metrics')]:
            frame = scope(pd.read_csv(run/f'{stage}_{suffix}.csv'))
            expected = set(methods if stage == 'train' else methods+ALIASES)
            for _, part in frame.groupby(['fold','group','scenario','regime']):
                if set(part.model) != expected: raise ValueError('Unmatched real strategy cells')
            if stage == 'train':
                copies=[]
                for fold, part in frame.groupby('fold'):
                    for alias,name in [('selected',selection['folds'][fold]['model']),('common',selection['common']['model'])]:
                        local=part[part.model==name].copy();local['model']=alias;copies.append(local)
                frame=pd.concat([frame,*copies],ignore_index=True)
            target.append(frame)
    real=pd.concat(metrics,ignore_index=True); risk=pd.concat(risks,ignore_index=True)
    keys=['scope','fold','group','scenario','extrapolation','model','regime']
    check_horizons(real,keys,'mae_deg')
    real.to_csv(out/'real_metrics_all.csv',index=False)
    records=real.groupby(keys,as_index=False).mae_deg.mean();records.to_csv(out/'real_record_means.csv',index=False)
    horizon=real.groupby(['scope','fold','model','regime','horizon_s'],as_index=False).agg(mae_deg=('mae_deg','mean'),scenarios=('scenario','nunique'))
    horizon.to_csv(out/'real_horizons.csv',index=False)
    summary=horizon.groupby(['scope','fold','model','regime'],as_index=False).agg(mae_deg=('mae_deg','mean'),scenarios=('scenarios','min'))
    summary=compare(summary,['scope','fold','regime']);summary.to_csv(out/'real_summary_all_scopes.csv',index=False)
    for name,part in summary.groupby('scope'): part.to_csv(out/f'{name}_summary.csv',index=False)
    if risk.duplicated(keys).any() or not np.isfinite(risk[RISK_COLUMNS]).all().all(): raise ValueError('Invalid risk cells')
    risk.to_csv(out/'risk_record_metrics.csv',index=False)
    risk_summary=risk.groupby(['scope','fold','model','regime'],as_index=False)[RISK_COLUMNS].mean()
    risk_summary=risk_summary.rename(columns={'packet_mae_p95_deg':'mean_of_record_packet_mae_p95_deg'})
    risk_summary.to_csv(out/'risk_summary.csv',index=False)
    return methods,real,summary,horizon,risk_summary


def fit_tables(run,cfg,out):
    rows=json.loads((run/'fits.json').read_text())
    expected={(f,n,s) for f in cfg['folds'] for n in cfg['variants'] for s in cfg['seeds']}
    if len(rows)!=90 or {(r['fold'],r['model'],r['seed']) for r in rows}!=expected:raise ValueError('Expected exactly 90 unique fits')
    fits=pd.DataFrame(rows)
    cols=['fold','model','seed','selected_epoch','epochs_executed','trainable_parameter_count','total_parameter_count',
          'optimizer','optimizer_steps','backward_passes','optimizer_closure_evaluations','fit_seconds','elapsed_seconds',
          'selection_score','selection_candidate']
    flat=fits[cols].copy();flat['retained_starting_checkpoint']=flat.selected_epoch==-1
    flat['frozen_parameter_count']=flat.total_parameter_count-flat.trainable_parameter_count
    flat.to_csv(out/'fits_flat.csv',index=False)
    aggregate=flat.groupby('model',as_index=False).agg(fits=('seed','size'),epoch_minus_one=('retained_starting_checkpoint','sum'),
        selected_epoch_min=('selected_epoch','min'),selected_epoch_median=('selected_epoch','median'),selected_epoch_max=('selected_epoch','max'),
        epochs_executed=('epochs_executed','sum'),trainable_min=('trainable_parameter_count','min'),trainable_max=('trainable_parameter_count','max'),
        total_parameter_count=('total_parameter_count','max'),optimizer_steps=('optimizer_steps','sum'),backward_passes=('backward_passes','sum'),
        closure_evaluations=('optimizer_closure_evaluations','sum'),fit_seconds=('fit_seconds','sum'),runner_elapsed_seconds=('elapsed_seconds','sum'))
    aggregate.to_csv(out/'fit_summary.csv',index=False)
    flat.groupby(['fold','model'],as_index=False).agg(epoch_minus_one=('retained_starting_checkpoint','sum'),
        best_epoch_mean=('selected_epoch','mean'),fit_seconds=('fit_seconds','sum')).to_csv(out/'fit_fold_summary.csv',index=False)
    return flat,aggregate


def seed_tables(root,run,cfg,out,real,summary):
    data=load_dataset(root/'data/processed')
    source=json.loads((root/cfg['source_run']/'provenance.json').read_text())
    rows=[]
    unique=real[['fold','group','scenario','extrapolation','scope']].drop_duplicates()
    for entry in unique.itertuples(index=False):
        cal=LengthCalibration(**source['inputs'][entry.fold]['calibration'])
        with np.load(run/entry.fold/f'predictions_s{entry.scenario}.npz',allow_pickle=False) as saved:
            raw=command_windows(data[entry.scenario],cal,saved['k'])
            np.testing.assert_array_equal(raw['target_q'],saved['target'])
            active=np.abs(raw['initial_velocity'])>=cfg['active_threshold']
            seed_names={name[:-7] for name in saved.files if name.endswith('__seeds')}
            if seed_names!=set(cfg['variants']):raise ValueError('Saved new-strategy seed set mismatch')
            for name in cfg['variants']:
                values=saved[name+'__seeds']
                if values.shape!=(len(cfg['seeds']),len(raw['omega']),30) or not np.isfinite(values).all():raise ValueError('Invalid seed predictions')
                if not np.allclose(values.mean(0),saved[name],atol=1e-8,rtol=1e-7):raise ValueError('Stored ensemble differs from seed mean')
                for i,seed in enumerate(cfg['seeds']):
                    error=np.abs(np.rad2deg(values[i][:,HORIZON_INDEX]-raw['target_q'][:,HORIZON_INDEX]))
                    for regime,mask in [('all',np.ones(len(active),bool)),('active',active),('quiet',~active)]:
                        if not mask.any():continue
                        for j,h in enumerate(HORIZONS_S):
                            rows.append(dict(scope=entry.scope,fold=entry.fold,group=entry.group,scenario=entry.scenario,
                                extrapolation=entry.extrapolation,model=name,seed=seed,regime=regime,horizon_s=h,mae_deg=float(error[mask,j].mean())))
    detail=pd.DataFrame(rows);detail.to_csv(out/'seed_record_horizons.csv',index=False)
    per_horizon=detail.groupby(['scope','fold','model','seed','regime','horizon_s'],as_index=False).mae_deg.mean()
    per_seed=per_horizon.groupby(['scope','fold','model','seed','regime'],as_index=False).mae_deg.mean()
    per_seed.to_csv(out/'seed_summary.csv',index=False)
    dispersion=per_seed.groupby(['scope','fold','model','regime'],as_index=False).agg(seed_mae_mean_deg=('mae_deg','mean'),
        seed_mae_std_deg=('mae_deg',lambda x:float(x.std(ddof=1))),seed_mae_min_deg=('mae_deg','min'),seed_mae_max_deg=('mae_deg','max'),seeds=('seed','size'))
    ensemble=summary[['scope','fold','model','regime','mae_deg']].rename(columns={'mae_deg':'ensemble_mae_deg'})
    dispersion=dispersion.merge(ensemble,on=['scope','fold','model','regime'],validate='one_to_one')
    dispersion.to_csv(out/'seed_dispersion.csv',index=False)
    write(out/'seed_scope.json',{'methods':list(cfg['variants']),'seeds':cfg['seeds'],'note':SEED_NOTE,
                               'seed_mae_mean_is_not_ensemble_mae':True,'aggregation':'Equal records at each horizon, then equal horizons, then descriptive seed statistics'})
    return dispersion


def adoption(summary,cfg,selection,out):
    cells=[]
    for fold in cfg['folds']:
        ref=cfg['acceptance'][fold+'_reference'];threshold=cfg['acceptance'][fold+'_max_ratio']
        for regime in ['all','active']:
            row=summary[(summary.scope=='evaluation_extrapolation')&(summary.fold==fold)&(summary.model=='common')&(summary.regime==regime)]
            if len(row)!=1:raise ValueError('Missing adoption cell')
            r=row.iloc[0];ratio=float(r.mae_deg/r[ref+'_mae_deg'])
            cells.append(dict(scope='evaluation_extrapolation',fold=fold,regime=regime,reference=ref,ratio=ratio,maximum_ratio=threshold,passed=ratio<=threshold))
    ref=cfg['acceptance']['load_reference'];threshold=cfg['acceptance']['load_max_ratio']
    for regime in ['all','active']:
        row=summary[(summary.scope=='load_stress')&(summary.fold=='primary')&(summary.model=='common')&(summary.regime==regime)]
        if len(row)!=1:raise ValueError('Missing load adoption cell')
        r=row.iloc[0];ratio=float(r.mae_deg/r[ref+'_mae_deg'])
        cells.append(dict(scope='load_stress',fold='primary',regime=regime,reference=ref,ratio=ratio,maximum_ratio=threshold,passed=ratio<=threshold))
    result=dict(common_model=selection['common']['model'],passed=all(r['passed'] for r in cells),cells=cells,
                operational_thresholds_not_statistical_bounds=True)
    write(out/'acceptance.json',result);pd.DataFrame(cells).to_csv(out/'acceptance.csv',index=False)
    return result


def simulation_tables(run,cfg,out,methods):
    frame=pd.read_csv(run/'simulation_metrics.csv')
    casekeys=['fold','model','force_law','drive_amplitude_m','future_acceleration_gain','measurement','omega_ratio',
              'damping_per_s','length_m','amplitude_deg','phase_rad','trajectory_id']
    check_horizons(frame,casekeys,'absolute_error_deg')
    expected=set(methods+ALIASES)
    counts={}
    for fold,part in frame.groupby('fold'):
        if set(part.model)!=expected:raise ValueError('Unmatched simulation method set')
        cases=part[[c for c in casekeys if c not in ['fold','model']]].drop_duplicates()
        if len(cases)!=1152:raise ValueError('Expected 1152 cases per fold')
        counts[fold]=len(cases)
    if set(counts)!=set(cfg['folds']):raise ValueError('Missing simulation fold')
    cases=frame.groupby(casekeys,as_index=False).absolute_error_deg.mean().rename(columns={'absolute_error_deg':'mae_deg'})
    expected_counts=cases.groupby([c for c in casekeys if c!='model']).model.nunique()
    if not (expected_counts==len(expected)).all():raise ValueError('Incomplete per-case model match')
    cases.to_csv(out/'simulation_case_means.csv',index=False)
    groups=['fold','model','force_law','drive_amplitude_m','future_acceleration_gain','measurement']
    summary=cases.groupby(groups,as_index=False).agg(mae_deg=('mae_deg','mean'),case_median_mae_deg=('mae_deg','median'),
        case_p95_mae_deg=('mae_deg',lambda x:float(x.quantile(.95))),worst_case_mae_deg=('mae_deg','max'),cases=('mae_deg','size'))
    summary=compare(summary,[k for k in groups if k!='model']);summary.to_csv(out/'simulation_groups.csv',index=False)
    for suffix,extras in [('frequency_damping',['omega_ratio','damping_per_s']),('lengths',['length_m'])]:
        cases.groupby(groups+extras,as_index=False).mae_deg.mean().to_csv(out/f'simulation_{suffix}.csv',index=False)
    frame.groupby(groups+['horizon_s'],as_index=False).absolute_error_deg.mean().to_csv(out/'simulation_horizons.csv',index=False)
    return counts


def table(frame):
    def cell(value):
        if isinstance(value,(float,np.floating)):return f'{value:.6g}'
        return str(value).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |',
                      *['| '+' | '.join(map(cell,row))+' |' for row in frame.itertuples(index=False,name=None)]])


def figures(out,summary,horizon,methods,selection):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,8),sharey=True)
    for ax,fold in zip(axes,selection['folds']):
        part=summary[(summary.scope=='evaluation_extrapolation')&(summary.fold==fold)&(summary.regime=='all')].set_index('model').loc[methods]
        colors=['#768195' if n in CONTROLS else '#18847c' for n in methods]
        ax.barh(np.arange(len(methods)),part.mae_deg,color=colors);ax.set_title(fold);ax.set_xlabel('MAE угла, °');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_yticks(np.arange(len(methods)),methods);axes[0].invert_yaxis()
    fig.suptitle('Все 15 методов: ретроспективная экстраполяция по длине',fontsize=13)
    fig.tight_layout();fig.savefig(out/'all_methods_extrapolation.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    for ax,fold in zip(axes,selection['folds']):
        local=horizon[(horizon.scope=='evaluation_extrapolation')&(horizon.fold==fold)&(horizon.regime=='all')]
        names=list(dict.fromkeys(['incumbent','soft_symmetry','previous_local',selection['folds'][fold]['model']]))
        for name in names:
            y=local[local.model==name].sort_values('horizon_s');ax.plot(y.horizon_s,y.mae_deg,'o-',ms=3,label=name)
        ax.set(title=fold,xlabel='Горизонт, с',ylabel='MAE, °');ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=7)
    fig.suptitle('Фиксированный локальный выбор по validation и сохранённые сравнения',fontsize=12)
    fig.tight_layout();fig.savefig(out/'selected_horizons.png',dpi=180);plt.close(fig)


def document(path,run,out,cfg,summary,risk,fits,fit_summary,seeds,selection,verdict,counts,hash_counts):
    headline=summary[(summary.scope=='evaluation_extrapolation')&summary.regime.isin(['all','active'])]
    matched=headline.pivot(index='model',columns=['fold','regime'],values='mae_deg');matched.columns=[' / '.join(c) for c in matched.columns]
    chosen=headline[headline.model=='selected'][['fold','regime','mae_deg','change_vs_incumbent_percent','change_vs_soft_symmetry_percent','change_vs_previous_local_percent']]
    load=summary[(summary.scope=='load_stress')&(summary.fold=='primary')].pivot(index='model',columns='regime',values='mae_deg').reset_index()
    risk_table=risk[(risk.scope=='evaluation_extrapolation')&(risk.regime=='all')&risk.model.isin(['incumbent','soft_symmetry','previous_local','selected','common'])]
    seed_table=seeds[(seeds.scope=='evaluation_extrapolation')&(seeds.regime=='all')][['fold','model','seed_mae_mean_deg','seed_mae_std_deg','seed_mae_min_deg','seed_mae_max_deg','ensemble_mae_deg']]
    status='выполнен' if verdict['passed'] else 'не выполнен'
    replay_path=Path(__file__).resolve().parents[1]/out/'replay_verification.json'
    replay_note='Полная проверка из сохранённых весов ещё не представлена.'
    if replay_path.exists():
        replay=json.loads(replay_path.read_text())
        if not replay['passed']:raise ValueError('Independent replay failed')
        replay_note=(f"Полный replay: {replay['prediction_comparisons']} сравнений на {replay['packet_count']} пакетах; "
                     f"максимальное расхождение {replay['max_error_deg']:.6g}°. "
                     f"Проверены все {replay['parameter_audit']['models_checked']} наборов весов, точная заморозка параметров "
                     "и возврат к эпохе −1. Исходный snapshot основной версии и APHYNITY совпадает. "
                     "Подробности: `replay_verification.json`.")
    latency_file=Path(__file__).resolve().parents[1]/out/'inference_latency.json'
    latency_note=('Дополнительный benchmark сохранён в `inference_latency.json`: прогретый CPU, batch=1, '
                  'ансамбль трёх seed с актором и observer. Измеренная задержка на этом устройстве '
                  'не является гарантией real-time или worst-case execution time.'
                  if latency_file.is_file() else 'Отдельный benchmark задержки batch=1 в этом отчёте отсутствует.')
    text=f'''# Fine-tuning и критерии из управления — 7 ноября 2026

Завершены 90 дообучений: 10 стратегий × 3 разбиения × 3 сохранённых seed. Общий выбор по validation — `{selection['common']['model']}`; заранее заданный критерий принятия **{status}**. Локальные решения: {', '.join(f"{fold}: `{value['model']}`" for fold,value in selection['folds'].items())}. В {int(fits.retained_starting_checkpoint.sum())} из 90 fit выбран исходный checkpoint эпохи −1; это отсутствие принятого обновления, а не улучшение от fine-tuning.

## Выбранные локальные решения

{table(chosen)}

Это ретроспективное исследование ранее использованных реальных записей. Стратегии и checkpoints выбраны по validation до evaluation и новой симуляции; нет нового независимого аппаратного теста или основания для SOTA. `selected`/`common` являются псевдонимами уже выбранных моделей. `previous_local` сохраняет прошлые решения, включая LPV на коротких тросах; одна эта строка не означает одинаковую архитектуру между разбиениями.

## Что менялось

Все стратегии начинают с соответствующего soft-symmetry checkpoint, архитектура и актор сохраняются. Проверены обычное продолжение; только observer; только innovation; L2-SP к исходным параметрам; привязка выходов к исходной модели; SAM; EMA; фазовый/конечный surrogate; minibatch group-DRO; L-BFGS polish. Предыдущие данные, веса и отдельный APHYNITY не смешиваются с обучением этой серии.

История — 41 отсчёт, 30 шагов до 1.5 с. Метрики используют горизонты {', '.join(f'{h:g}' for h in HORIZONS_S)} с. Усреднение: равные веса целых сценариев на каждом горизонте, затем равные веса горизонтов. Headline — evaluation/extrapolation; interpolation и нагрузка записаны отдельно.

Литературная основа: [методы fine-tuning](FINETUNING_METHODS_RESEARCH_20261107.md) и [выбор методов в управлении](CONTROL_METHOD_SELECTION_RESEARCH_20261107.md). [DiffTune-MPC](https://arxiv.org/html/2312.11384v2) различает внутреннюю стоимость планирования и оценку замкнутой траектории. [Sakib–Pan, раздел V](https://arxiv.org/html/2408.06607v6) отдельно проверяет optimizer switching, расписание и rollout; эти результаты служат мотивацией эксперимента, не обещанием переноса эффекта на CranePilot.

Дополнительный обзор постановки задачи и регуляризации: [данные для управления, DeePC и W-PGNN](CONTROL_DATA_AND_REGULARIZATION_20261107.md).

Существующий LinearMPC не использует этот нейронный forecaster. Здесь не измерялись settling time, управленческое усилие или замкнутый cost. Phase-space loss — ошибка нормированных приращений и конечного угла, а не точное измерение скорости и не градиент реального управления. Квантизация меток может влиять на такой loss.

## Все методы: MAE на экстраполяции, градусы

{table(matched.reset_index())}

## Нагрузка в основном разбиении

{table(load)}

## Сохранённый критерий принятия общей стратегии

{table(pd.DataFrame(verdict['cells']))}

Primary сравнивается с soft symmetry; short-transfer должен сохранить предыдущий LPV-результат с допуском 1%; long-transfer и нагрузка сравниваются с incumbent с тем же допуском. Это практические пороги, а не статистические границы. Все отрицательные результаты и начальные checkpoints сохранены.

## Фактическое обучение и стоимость вычислений

{table(fit_summary)}

Полные selected epochs и замороженные параметры каждого fit — в `fits_flat.csv`, исходные метаданные — `fits.json`. `fit_seconds` суммируют время fit; `runner_elapsed_seconds` дополнительно включает сохранение/перезагрузку и локальную проверку. Одинаковый максимум эпох не означает одинаковый compute: SAM имеет два backward на update; L-BFGS делает переменное число closure в line search. Поле `closure_evaluations` в исходной реализации считает оценивания training closure с backward для всех оптимизаторов; для обычного AdamW это один проход. EMA выбирает только усреднённые веса. Число параметров и фактические проходы приведены, без заявления о равном FLOP-бюджете.

{latency_note}

## Разброс по трём исходным seed

{table(seed_table)}

Это описательные mean/std/min/max MAE отдельных seed после равного усреднения сценариев и горизонтов. Средняя MAE seed отличается от MAE ансамбля, который сначала усредняет прогнозы. Три seed не являются независимыми аппаратными испытаниями и не дают доверительный интервал для новой длины. У пяти сохранённых контролей в этом run нет отдельных `__seeds` массивов; им не приписывается искусственный нулевой разброс.

## Диагностика крупных ошибок и фазы

{table(risk_table)}

`mean_of_record_packet_mae_p95_deg` — **среднее отдельных 95-х процентилей записей**, не pooled p95 всех окон. Peak — среднее максимума абсолютной ошибки каждого 30-шагового пакета; tail — MAE последних 10 шагов; increment — MAE разности ошибки, нормированной на omega×0.05. Эти показатели относятся к ошибке прогноза, не к нарушениям физических ограничений. В `risk_summary.csv` сохранены все методы и режимы, а не только таблица выше.

## Симуляция и проверка воспроизводимости

Новая сетка содержит {json.dumps(counts,ensure_ascii=False)} наблюдений по разбиениям: всего {sum(counts.values())} случаев. Все 15 методов и два псевдонима получают одинаковое заданное ускорение; актор обойдён только в этой угловой диагностике. Сохраняются отдельные force-law/drive/gain/noise группы, p95 и максимумы ошибок случаев, длины, частоты, затухания и горизонты. Это прежнее семейство симулятора и не проверка реального контроллера.

SHA-проверки и этапные проверки: {json.dumps(hash_counts,ensure_ascii=False)}. Проверены исходники/данные, 90 новых checkpoints, исходные контрольные веса, полный manifest результатов и selection. Дополнительно runner проверяет перезагрузку checkpoints, контрольные прогнозы, подмену future-target, историю без targets, сходимость интегратора и скрытый future-gain. {replay_note}

Run: `{run}`. Результаты: `{out}`. Архив этого этапа формируется отдельно после завершения отчёта; полный replay требует сохранённых upstream данных и весов. Ничего не публикуется в GitHub.
'''
    path.write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('artifacts/runs/finetune_campaign_20261107'))
    parser.add_argument('--output',type=Path,default=Path('results/finetune_campaign_20261107'))
    parser.add_argument('--document',type=Path,default=Path('docs/FINETUNE_CAMPAIGN_2026_11_07_RU.md'))
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];run=root/args.run;out=root/args.output;doc=root/args.document
    prov,counts=verify_evidence(root,run);cfg=prov['config']
    if (out/'report.json').exists() or doc.exists():raise FileExistsError('Preserve completed report')
    selection=json.loads((run/'selection.json').read_text())
    if selection['uses_evaluation_this_stage'] or selection['uses_simulation']:raise ValueError('Selection used evaluation')
    out.mkdir(parents=True,exist_ok=True)
    methods,real,summary,horizon,risks=read_real(run,cfg,out,selection)
    if len(methods)!=15:raise ValueError('Expected fixed 15 methods')
    fits,fit_summary=fit_tables(run,cfg,out)
    seeds=seed_tables(root,run,cfg,out,real,summary)
    verdict=adoption(summary,cfg,selection,out);simulation_counts=simulation_tables(run,cfg,out,methods)
    for name in ['selection.json','fits.json','training.csv','train_verification.json','evaluate_verification.json','train_timings.csv','evaluate_timings.csv']:
        shutil.copy2(run/name,out/name)
    figures(out,summary,horizon,methods,selection)
    document(doc,args.run,args.output,cfg,summary,risks,fits,fit_summary,seeds,selection,verdict,simulation_counts,counts)
    report=dict(source_provenance_sha256=sha256(run/'provenance.json'),report_source_sha256=sha256(Path(__file__)),
                document=str(doc.relative_to(root)),document_sha256=sha256(doc),verified_hash_counts=counts,
                new_fits=len(fits),methods=methods,selection=selection,acceptance=verdict,
                epoch_minus_one_count=int(fits.retained_starting_checkpoint.sum()),fit_seconds=float(fits.fit_seconds.sum()),
                optimizer_steps=int(fits.optimizer_steps.sum()),backward_passes=int(fits.backward_passes.sum()),
                closure_evaluations=int(fits.optimizer_closure_evaluations.sum()),simulation_cases_per_fold=simulation_counts,
                seed_note=SEED_NOTE,risk_quantile_aggregation='Mean of record-wise packet MAE p95; not pooled p95',
                real_records_retrospective=True,closed_loop_control_evaluated=False,headline_scope='evaluation_extrapolation')
    latency=out/'inference_latency.json'
    if latency.is_file():report['inference_latency_sha256']=sha256(latency)
    write(out/'report.json',report)
    print(summary[(summary.scope=='evaluation_extrapolation')&summary.model.isin(ALIASES)].to_string(index=False))
    print('ACCEPTANCE',json.dumps(verdict,indent=2));print('FITS',len(fits),'EPOCH_MINUS_ONE',report['epoch_minus_one_count'])


if __name__=='__main__':main()
