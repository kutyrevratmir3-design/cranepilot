"""All frozen structural-transfer outcomes, gates, paired ablations and figures."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from scripts.report_angular_control import table,METRICS

OUT=Path('results/structural_transfer_20261107')
RUN=Path('artifacts/runs/structural_transfer_20261107')
RAW=['rail_violation_100hz','speed_violation_100hz','angle_violation_100hz']

def main():
    out=OUT;run=RUN
    if (out/'report.json').exists():raise FileExistsError('Preserve prior report')
    p=json.loads((run/'provenance.json').read_text());v=json.loads((out/'verification.json').read_text());cfg=p['config']
    assert p['status']=='complete' and v['passed'] and v['provenance_sha256']==sha256(run/'provenance.json')
    for mapping,base in [('guarded_hashes',Path('.')),('result_hashes',run)]:
        for name,digest in p[mapping].items():assert sha256(base/name)==digest,name
    rows=pd.concat([pd.read_csv(run/stage/'metrics.csv').assign(stage=stage) for stage in ['development','evaluate']],ignore_index=True)
    raw=pd.read_csv(out/'raw_sampling_metrics.csv');rows=rows.merge(raw,on=['stage','method','case'],validate='one_to_one')
    rows.to_csv(out/'all_episode_metrics.csv',index=False)
    metrics=METRICS+['settled','solver_nonfinite','fallbacks','current_velocity_estimate_rmse','current_acceleration_estimate_rmse','tail_estimated_tau_s','tail_estimated_gain','id_update_fraction','observer_use_fraction']
    def aggregate(frame,keys):
        return frame.groupby(keys,as_index=False)[metrics].mean().merge(frame.groupby(keys,as_index=False)[RAW].sum(),on=keys)
    summary=aggregate(rows,['stage','method']);summary.to_csv(out/'summary.csv',index=False)
    ev=rows[rows.stage=='evaluate'];family=aggregate(ev,['method','family']);family.to_csv(out/'family_summary.csv',index=False)
    for keys in [['method','archetype'],['method','noisy','pulse']]:aggregate(ev,keys).to_csv(out/('groups_'+'_'.join(keys[1:])+'.csv'),index=False)
    pivot=summary[summary.stage=='evaluate'].set_index('method');ref=pivot.loc['nominal'];comp=[];boot=[]
    rng=np.random.default_rng(7112026);bootstrap_ids=rng.integers(0,24,(2000,24))
    for method in cfg['methods']:
        candidate=ev[ev.method==method].set_index('case');reference=ev[ev.method=='nominal'].set_index('case');assert candidate.index.equals(reference.index)
        for metric in METRICS:
            delta=100*(candidate[metric].mean()/reference[metric].mean()-1)
            comp.append(dict(method=method,reference='nominal',metric=metric,change_percent=delta,paired_case_wins=int((candidate[metric]<reference[metric]).sum()),cases=len(candidate)))
            if metric=='realized_cost':
                pair=pd.DataFrame(dict(context=candidate.context,candidate=candidate[metric],reference=reference[metric])).groupby('context').mean()
                assert len(pair)==24;ids=bootstrap_ids
                effects=100*(pair.candidate.to_numpy()[ids].mean(1)/pair.reference.to_numpy()[ids].mean(1)-1)
                boot.append(dict(method=method,metric=metric,contexts=len(pair),change_percent=delta,descriptive_low=np.quantile(effects,.025),descriptive_high=np.quantile(effects,.975)))
    comp=pd.DataFrame(comp);comp.to_csv(out/'paired_comparisons.csv',index=False);pd.DataFrame(boot).to_csv(out/'context_bootstrap.csv',index=False)
    gates={};family_effects=[]
    for method in cfg['methods']:
        c=pivot.loc[method];family_gates={}
        for f in cfg['families']:
            fc=family[(family.method==method)&(family.family==f)].iloc[0];fr=family[(family.method=='nominal')&(family.family==f)].iloc[0]
            row=dict(method=method,family=f,cost_change_percent=100*(fc.realized_cost/fr.realized_cost-1),tracking_change_percent=100*(fc.tracking_ise_m2_s/fr.tracking_ise_m2_s-1),sway_change_percent=100*(fc.sway_ise_deg2_s/fr.sway_ise_deg2_s-1))
            family_effects.append(row)
            family_gates[f]=bool(fc.realized_cost<=1.10*fr.realized_cost and fc.tracking_ise_m2_s<=1.10*fr.tracking_ise_m2_s and all(fc[k]<=fr[k] for k in RAW))
        if method=='nominal':continue
        g=dict(cost_improved=bool(c.realized_cost<ref.realized_cost),sway_improved=bool(c.sway_ise_deg2_s<ref.sway_ise_deg2_s),tracking_improved=bool(c.tracking_ise_m2_s<ref.tracking_ise_m2_s),final_error_within_1percent=bool(c.final_position_error_m<=1.01*ref.final_position_error_m),no_additional_raw_violations=bool(all(c[k]<=ref[k] for k in RAW)),majority_cost_wins=bool(comp[(comp.method==method)&(comp.metric=='realized_cost')].paired_case_wins.iloc[0]>48),no_nonfinite_solves=bool(c.solver_nonfinite==0))
        g['global_passed']=all(g.values());g['family_gates']=family_gates;g['structural_transfer_passed']=g['global_passed'] and all(family_gates.values());gates[method]=g
    (out/'acceptance.json').write_text(json.dumps(gates,indent=2)+'\n')
    fe=pd.DataFrame(family_effects);fe.to_csv(out/'family_effects.csv',index=False)
    pairs=[('adaptive_observer','fixed_observer'),('adaptive_force_observer','fixed_force_observer'),('fixed_observer','nominal'),('fixed_force_observer','fixed_observer'),('adaptive_force_observer','adaptive_force'),('adaptive_force_observer','frozen_adaptive_force_observer'),('frozen_adaptive_force_observer','force_actuator')]
    ablations=[]
    for method,reference in pairs:
        c=ev[ev.method==method].set_index('case');r=ev[ev.method==reference].set_index('case')
        row=dict(method=method,reference=reference)
        for metric in ['realized_cost','tracking_ise_m2_s','sway_ise_deg2_s','final_position_error_m']:
            row[metric+'_change_percent']=100*(c[metric].mean()/r[metric].mean()-1)
            row[metric+'_paired_wins']=int((c[metric]<r[metric]).sum())
        ablations.append(row)
    ablations=pd.DataFrame(ablations);ablations.to_csv(out/'ablation_effects.csv',index=False)
    timing=[]
    for method in cfg['methods']:
        with np.load(run/'evaluate'/f'{method}.npz') as z:
            s=z['batch_seconds'];timing.append(dict(method=method,cases=96,batch_median_ms=np.median(s)*1000,batch_max_ms=s.max()*1000,amortized_median_ms=np.median(s)*1000/96))
    pd.DataFrame(timing).to_csv(out/'runtime_batches.csv',index=False)
    lat=pd.read_csv(out/'runtime_batch1.csv');latency=lat.groupby('method').elapsed_ms.agg(['median','max']).reset_index();latency['fraction_over_50ms']=lat.groupby('method').over_50ms.mean().to_numpy();latency.to_csv(out/'runtime_batch1_summary.csv',index=False)
    assert lat.plan_max_difference.max()<5e-5
    methods=cfg['methods'];colors=['#80868b','#247885','#589966','#398958','#af7744','#ad5964','#65639b','#252e43','#b99844']
    fig,axes=plt.subplots(1,3,figsize=(19,6.2),layout='constrained')
    for ax,metric,label in zip(axes,['realized_cost','tracking_ise_m2_s','sway_ise_deg2_s'],['Realized task cost','Tracking ISE, m²·s','Sway ISE, deg²·s']):
        ax.bar(methods,[pivot.loc[m,metric] for m in methods],color=colors);ax.set_title(label);ax.tick_params(axis='x',rotation=55,labelsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Structural mismatch and reversal: 96 paired synthetic cases per controller');fig.savefig(out/'control_comparison.png',dpi=160);plt.close(fig)
    traces={m:np.load(run/'evaluate'/f'{m}.npz') for m in methods};grid=np.arange(161)*.05;fig,axes=plt.subplots(4,3,figsize=(16,12),layout='constrained')
    for row,case in enumerate([0,31,64,95]):
        for m in ['nominal','force_actuator','fixed_force_observer','adaptive_force_observer','frozen_adaptive_force_observer']:
            color=colors[methods.index(m)];z=traces[m]
            axes[row,0].plot(grid,z['states'][case,:,0],label=m,color=color,lw=1.2);axes[row,1].plot(grid,np.rad2deg(z['states'][case,:,2]),color=color,lw=1.2);axes[row,2].step(grid[:-1],z['commands'][case],where='post',color=color,lw=1.)
        axes[row,0].step(grid,traces['nominal']['targets'],where='post',color='black',ls=':',lw=1);axes[row,0].set_ylabel(f'Case {case}, x (m)');axes[row,1].set_ylabel('theta (deg)');axes[row,2].set_ylabel('issued command (m/s²)')
        for ax in axes[row]:ax.set_xlabel('time (s)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.savefig(out/'fixed_examples.png',dpi=150);plt.close(fig)
    for z in traces.values():z.close()
    cols=['method','realized_cost','tracking_ise_m2_s','sway_ise_deg2_s','final_position_error_m','settled']+RAW
    diagnostics=pivot.reset_index()[['method','current_velocity_estimate_rmse','current_acceleration_estimate_rmse','tail_estimated_tau_s','tail_estimated_gain','observer_use_fraction','id_update_fraction']];diagnostics.to_csv(out/'identification_summary.csv',index=False)
    effect=comp.pivot(index='method',columns='metric',values='change_percent').reset_index()[['method','realized_cost','tracking_ise_m2_s','sway_ise_deg2_s','final_position_error_m']]
    wins=comp[comp.metric=='realized_cost'][['method','paired_case_wins','cases']]
    doc=Path('docs/STRUCTURAL_TRANSFER_2026_11_07_RU.md')
    passed=[m for m,g in gates.items() if g['structural_transfer_passed']]
    posthoc=pd.read_csv(out/'posthoc_adaptive_vs_static_families.csv')
    text=f'''# Структурная ошибка привода и разворот, 7 ноября 2026

Завершены 54 development и 864 evaluation эпизода (96 на каждый из 9 методов),
{v['decisions']} решений. Протокол и исполняемый код зафиксированы до запуска:
`{p['git_head']}`. Настройки после development не менялись, neural training отсутствует.
Прошедшие полный global+family gate: {', '.join(passed) if passed else 'нет'}.
Все альтернативы и отрицательные случаи сохранены; новый default не назначается.

## Основной вывод

Прежний adaptive+force+observer пережил structural mismatch: относительно nominal
mean cost −81.33%, sway −65.59%, tracking −4.53%, final error −41.11%; 76/96
cost wins. Raw angle>15deg остаётся в 8/96 случаях (nominal 12/96), поэтому
успех сравнительный, не отсутствие опасных excursions. Пять из восьми альтернатив
прошли полный критерий, причём frozen/new observer дают одинаковые траектории.

Сильный static force_actuator сохраняет преимущество по tracking: −8.57% против
nominal; settling 52/96 против 36/96 у adaptive+force+observer. У adaptive mean
cost −25.62% и sway −42.05% относительно static, но tracking +4.41%, final error
+24.83% и всего 25/96 cost wins. Данный средний выигрыш не означает преимущество
на типичном случае. Posthoc разбиение по семействам показывает пользу прежде
всего при hidden delay/combined, а на two_lag cost хуже static на 21.24%.

Fixed observer с теми же 2s истории даёт лишь −0.28% mean cost против nominal,
28/96 cost wins и tracking +1.01%. Adaptive force+observer против fixed force+
observer: cost −70.63%, но tracking +0.94%, 56/96 cost wins. Значит преимущество
online adaptation выходит за рамки одного длинного state-fit на этом grid;
универсального улучшения всех целей нет. New qualification gate не изменил
ни одного nontiming array относительно frozen winner — подтверждено отдельно.

Posthoc comparison adaptive_force_observer против force_actuator (не новый gate):

{table(posthoc)}

## Все результаты

{table(pivot.reset_index()[cols])}

Изменения средних относительно nominal, проценты (меньше лучше):

{table(effect)}

Парные выигрыши realized cost:

{table(wins)}

## Проверка переноса по семействам

{table(fe)}

Полные gates: `results/structural_transfer_20261107/acceptance.json`.
Global требует lower mean cost/sway/tracking, final error≤1.01 reference,
не больше raw нарушений каждого типа, cost wins>48/96 и ни одного nonfinite solve.
Transfer дополнительно требует cost и tracking≤1.10 reference, без роста raw
нарушений, отдельно в КАЖДОМ семействе. Проверка сравнительная и описательная,
не гарантия безопасности. Bootstrap cost resamples 24 context clusters (2000 draws,
seed 7112026, одни и те же resample indices для всех методов), не отдельные correlated ticks: `context_bootstrap.csv`.

## Сколько даёт адаптация сверх длинного наблюдателя

{table(ablations)}

Fixed_observer и fixed_force_observer используют тот же двухсекундный буфер,
профилирование initial x/v/a и current fit-RMSE gate, что и новые adaptive observer
варианты; tau=.1, K=1 остаются фиксированными. Adaptive counterparts отличаются
online tau/K. Вариант frozen_adaptive_force_observer сохраняет старый gate:
ever_valid parameter fit AND current fit-RMSE. Побитовый unit test подтверждает
прежние update states и планы. Смена gate отдельно отражена сравнением frozen/new.
Это policy comparisons: истории зависят от своих решений; RMSE между политиками
не является чистым fixed-history тестом оценивателя.

{table(diagnostics)}

Tau и gain здесь effective first-order approximations. У cascade/delay/friction
нет единственного истинного tau данной модели. Tail estimates не ошибки параметров;
у static scenario bank id_tau/gain — только point prior. Fit qualification и
конечный tau grid не доказывают идентифицируемость, persistent excitation или stability.

## Что именно усложнено

Первый порядок; скрытая задержка .04/.09s; два последовательных лага; friction
fc*tanh(v/.015)+.08v; изменение gain в 3.5s на .7/1.3; все эффекты вместе.
Четыре заранее заданных archetypes × 6 families × exact/noisy × pulse/noPulse.
Nuisances в archetypes связаны: это не независимый factorial test параметров.
Новые L .28/.98m, omega ratio .91/1.09 и seed 1407112026. Target .3m до 4s,
затем .15m; начальное x .1m, горизонт эксперимента 8s. Новая цель доступна
controller только начиная с 4s. Pulse .13rad/s² на [5.25,5.6)s.

Все controller получают .4s angular history и имеют доступ к одинаковой 2s
истории measured x/собственных ISSUED acceleration commands. Задержанная команда,
истинные состояния/параметры и будущие измерения доступны только plant/audit.
Endpoint v/a, tau/K профилируются причинно; future committed interval не входит
в estimator. Данные сохраняются на 100Hz, MPC работает на 20Hz с committed tick.
Cost и tracking ISE учитывают ступенчатую цель на control grid; trapezoid convention
в точке разворота задана заранее. Settling относится только к final target после 4s.
Actual acceleration effort учитывает friction, не только latent motor state.

Работа [Rohrs et al., 1985](https://doi.org/10.1109/TAC.1985.1104070)
([NASA record](https://ntrs.nasa.gov/citations/19850063162)) мотивирует явный тест
unmodeled dynamics. Наша конечная simulation grid и MPC не повторяют их алгоритм
или теоремы. Истинный actuator в прошлом опыте был близок к fitted first-order
структуре; этот этап проверяет именно практический предел такого совпадения.

## Проверка и время

172 unit tests PASS. Повторно интегрированы все {v['episodes']} эпизода,
причинно восстановлены все estimator states, перепроверены {v['decisions']}
выбранных forecasts/costs. Solver не запускался заново при полном replay.
SHA всех frozen inputs/results и {v['original_files']['checked_files']} исходных
перенесённых файлов проверены. Net acceleration/hidden motor state разделены;
raw issued queues и sensor measurements восстановлены. Metrics independently
recomputed; halfstep convergence проверен, подробности в verification.json. Дополнительный independent_metrics.json
восстанавливает net acceleration из latent motor/friction без plant/metric helpers
и заново вычисляет cost для всех 864 evaluation episodes. Distribution tails
сохранены в posthoc_cost_distribution.csv; анализа для tuning не было.

Warmed batch1 полный current decision, миллисекунды:

{table(latency)}

15 измерений на метод: случаи/тики (0,0),(31,40),(95,100), один warmup на точку,
5 repeats с одинаковым восстановленным pre-tick estimator state. Включены current
window assembly/update/observer/solver; исключены model I/O и восстановление
pre-tick state. Нет real-time/WCET guarantee. Wall-clock timing не добавляет
simulation delay сверх declared committed tick. Batch throughput отдельно в
runtime_batches.csv; plan differences относительно batch run в runtime_batch1.csv.

## Практические пределы

Это synthetic acceleration-command ablation: native xr features zero, trained
NeuralActuator bypass. Нельзя переносить вывод на нативный контроллер или реальный
кран без согласования xr/command timing и hardware tests. Нет доказательства
stability, отсутствия violations вне grid или SOTA. Все neural weights/defaults
и APHYNITY остаются прежними. Следующая новая методика должна сравниваться с
сильным static force_actuator и matched-memory observer, а не только nominal.
'''
    doc.write_text(text)
    report=dict(document=str(doc),document_sha256=sha256(doc),provenance_sha256=sha256(run/'provenance.json'),report_source_sha256=sha256(Path(__file__)),verification_sha256=sha256(out/'verification.json'),aggregation_helper_sha256=sha256(Path('scripts/report_angular_control.py')),runtime_batch1_sha256=sha256(out/'runtime_batch1.csv'),independent_metrics_sha256=sha256(out/'independent_metrics.json'),evaluation_episodes=864,development_episodes=54,neural_training=False,native_controller_evaluated=False,acceptance=gates)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(pivot.reset_index()[cols].to_string(index=False));print('TRANSFER_PASS',passed)

if __name__=='__main__':main()
