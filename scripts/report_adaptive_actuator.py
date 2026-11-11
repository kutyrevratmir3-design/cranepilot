"""All scenario-control outcomes; fixed gates, paired cases and context resampling."""
import argparse,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from scripts.report_angular_control import table,METRICS


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,default=Path('artifacts/runs/adaptive_actuator_20261107'))
    ap.add_argument('--output',type=Path,default=Path('results/adaptive_actuator_20261107'));a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise FileExistsError('Preserve prior report')
    p=json.loads((a.run/'provenance.json').read_text());v=json.loads((out/'verification.json').read_text())
    if p['status']!='complete' or not v['passed']:raise ValueError('Complete verified run required')
    assert v['provenance_sha256']==sha256(a.run/'provenance.json')
    for key,base in [('guarded_hashes',Path('.')),('result_hashes',a.run)]:
        for path,digest in p[key].items():
            if sha256(base/path)!=digest:raise ValueError(f'Changed {path}')
    cfg=p['config'];frames=[];timing=[]
    for stage in ['development','evaluate']:
        frame=pd.read_csv(a.run/stage/'metrics.csv');frame['stage']=stage;frames.append(frame)
        for method in cfg['methods']:
            with np.load(a.run/stage/f'{method}.npz') as z:n=len(z['commands']);sec=z['batch_seconds']
            timing.append(dict(stage=stage,method=method,cases=n,batch_median_ms=float(np.median(sec)*1000),batch_p95_ms=float(np.quantile(sec,.95)*1000),
                batch_max_ms=float(sec.max()*1000),amortized_median_ms_per_case=float(np.median(sec)*1000/n)))
    rows=pd.concat(frames,ignore_index=True);rows.to_csv(out/'all_episode_metrics.csv',index=False)
    summary=rows.groupby(['stage','method'],as_index=False)[METRICS+['settled','solver_nonfinite','fallbacks','current_acceleration_estimate_rmse','current_velocity_estimate_rmse','tail_tau_error_s','tail_gain_error','id_update_fraction','observer_use_fraction']].mean()
    violations=pd.read_csv(out/'raw_sampling_metrics.csv').groupby(['stage','method'],as_index=False)[['rail_violation_100hz','angle_violation_100hz','speed_violation_100hz']].sum()
    summary=summary.merge(violations,on=['stage','method']);summary.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(timing).to_csv(out/'runtime_batches.csv',index=False)
    ev=rows[rows.stage=='evaluate'];reference=ev[ev.method=='nominal'].set_index('case');comparisons=[];bootstrap=[];rng=np.random.default_rng(1107112026)
    for method in cfg['methods']:
        candidate=ev[ev.method==method].set_index('case');assert candidate.index.equals(reference.index)
        for metric in METRICS:
            delta=100*(candidate[metric].mean()/reference[metric].mean()-1)
            comparisons.append(dict(method=method,reference='nominal',metric=metric,mean=candidate[metric].mean(),reference_mean=reference[metric].mean(),
                change_percent=delta,paired_case_wins=int((candidate[metric]<reference[metric]).sum()),cases=len(candidate)))
            pairs=pd.DataFrame({'context':candidate.context,'candidate':candidate[metric],'reference':reference[metric]}).groupby('context').mean()
            ids=rng.integers(0,len(pairs),(2000,len(pairs)));effect=100*(pairs.candidate.to_numpy()[ids].mean(1)/pairs.reference.to_numpy()[ids].mean(1)-1)
            bootstrap.append(dict(method=method,metric=metric,contexts=len(pairs),change_percent=delta,descriptive_low=np.quantile(effect,.025),descriptive_high=np.quantile(effect,.975)))
    comp=pd.DataFrame(comparisons);comp.to_csv(out/'paired_comparisons.csv',index=False);pd.DataFrame(bootstrap).to_csv(out/'context_bootstrap.csv',index=False)
    pivot=summary[summary.stage=='evaluate'].set_index('method');r=pivot.loc['nominal'];gates={}
    for method in cfg['methods']:
        if method=='nominal':continue
        c=pivot.loc[method];wins=int(comp[(comp.method==method)&(comp.metric=='realized_cost')].paired_case_wins.iloc[0])
        gate=dict(cost_improved=bool(c.realized_cost<r.realized_cost),sway_improved=bool(c.sway_ise_deg2_s<r.sway_ise_deg2_s),
            tracking_improved=bool(c.tracking_ise_m2_s<r.tracking_ise_m2_s),final_error_within_1percent=bool(c.final_position_error_m<=r.final_position_error_m*1.01),
            no_additional_raw_violations=bool(all(c[k]<=r[k] for k in ['rail_violation_100hz','angle_violation_100hz','speed_violation_100hz'])),majority_cost_wins=wins>48)
        gate['passed']=all(gate.values());gates[method]=gate
    (out/'acceptance.json').write_text(json.dumps(gates,indent=2)+'\n')
    for factors in [['method','length'],['method','omega_ratio'],['method','tau'],['method','force_law'],['method','noisy','pulse']]:
        ev.groupby(factors,as_index=False)[METRICS].mean().to_csv(out/('groups_'+'_'.join(factors[1:])+'.csv'),index=False)
    methods=cfg['methods'];fig,axes=plt.subplots(1,3,figsize=(19,5.7),layout='constrained');colors=['#7b8790','#237c91','#267865','#668d3c','#ba7731','#9c577c','#bc5046','#4f569c','#272f3f']
    for ax,metric,label in zip(axes,['realized_cost','tracking_ise_m2_s','sway_ise_deg2_s'],['Realized task cost','Tracking ISE, m²·s','Sway ISE, deg²·s']):
        ax.bar(methods,[pivot.loc[m,metric] for m in methods],color=colors);ax.set_title(label);ax.tick_params(axis='x',rotation=50,labelsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Actuator ablations and online identification: 96 paired cases per controller');fig.savefig(out/'control_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(4,3,figsize=(15,12),layout='constrained');traces={m:np.load(a.run/'evaluate'/f'{m}.npz') for m in methods};grid=np.arange(121)*.05
    for row,case in enumerate([0,31,64,95]):
        for m,color in zip(['nominal','force_only','force_actuator','adaptive_params','adaptive_force_observer'],[colors[j] for j in [0,1,3,5,8]]):
            z=traces[m];axes[row,0].plot(grid,z['states'][case,:,0],label=m,color=color,lw=1.3);axes[row,1].plot(grid,np.rad2deg(z['states'][case,:,2]),color=color,lw=1.3)
            axes[row,2].step(grid[:-1],z['commands'][case],where='post',color=color,lw=1.1)
        axes[row,0].axhline(.3,color='black',ls=':',lw=1);axes[row,0].set_ylabel(f'Case{case}\nx, m');axes[row,1].set_ylabel('theta, deg');axes[row,2].set_ylabel('command, m/s²')
        for ax in axes[row]:ax.set_xlabel('time, s');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.savefig(out/'fixed_examples.png',dpi=150);plt.close(fig)
    for z in traces.values():z.close()
    lat=pd.read_csv(out/'runtime_batch1.csv');latency=lat.groupby('method').elapsed_ms.agg(['median','max']).reset_index()
    latency['fraction_over_50ms']=lat.groupby('method').over_50ms.mean().to_numpy();latency.to_csv(out/'runtime_batch1_summary.csv',index=False)
    view=['method','realized_cost','tracking_ise_m2_s','sway_ise_deg2_s','tail_rms_deg','final_position_error_m','settled','rail_violation_100hz','angle_violation_100hz','speed_violation_100hz']
    effects=comp.pivot(index='method',columns='metric',values='change_percent').reset_index()
    win_table=comp[comp.metric=='realized_cost'][['method','change_percent','paired_case_wins','cases']]
    ablations=[]
    for method,ref in [('force_actuator','force_only'),('force_actuator','actuator_only'),('joint_previous','force_actuator'),('adaptive_force','adaptive_params'),('adaptive_observer','adaptive_params'),('adaptive_force_observer','adaptive_force'),('adaptive_force_observer','adaptive_observer')]:
        row=dict(method=method,reference=ref)
        for metric in ['realized_cost','tracking_ise_m2_s','sway_ise_deg2_s','final_position_error_m']:
            row[metric+'_change_percent']=100*(pivot.loc[method,metric]/pivot.loc[ref,metric]-1)
        ablations.append(row)
    ablations=pd.DataFrame(ablations);ablations.to_csv(out/'ablation_effects.csv',index=False)
    doc=Path('docs/ADAPTIVE_ACTUATOR_2026_11_07_RU.md')
    diagnostics=summary[summary.stage=='evaluate'][['method','current_velocity_estimate_rmse','current_acceleration_estimate_rmse','tail_tau_error_s','tail_gain_error','id_update_fraction','observer_use_fraction']]
    diagnostics.to_csv(out/'identification_summary.csv',index=False)
    text=f'''# Online actuator identification и разделение uncertainty, 7 ноября2026

Завершены54development и864evaluation эпизода:96на каждый из9controller.
Новый протокол/код зафиксированы до запуска: `{p['git_head']}`.
Нейронные веса неизменны; online tau/gain updates являются заранее заданной
частью adaptive controller, а не подбором offline hyperparameters по test.
После development настройки не менялись. Все8альтернатив публикуются.

## Результаты на новом наборе

{table(pivot.reset_index()[view])}

Процентные изменения относительно nominal physical:

{table(effects)}

Парные выигрыши стоимости:

{table(win_table)}

Фиксированные gates: `{json.dumps(gates,ensure_ascii=False)}`.
Требуются одновременно lower mean cost/sway/tracking, final error<=1.01×nominal,
raw100Hz violations не выше и cost wins>48/96. Меньшее среднее не заменяет
большинство пар. Это finite-family criteria, не hardware или stability guarantee.

## Что отделено в этом опыте

Static: nominal; force_only с2законами forcing; actuator_only с3парами(tau,gain);
force_actuator с6сценариями; joint_previous с прежними18сценариями (ещё3частоты).
Поэтому force_only vs nominal и actuator_only vs nominal разделяют факторы,
которые в предыдущем joint изменялись одновременно. Joint_previous против
force_actuator отдельно показывает вклад frequency bank на этом новом grid.

Adaptive: adaptive_params с online lag/gain и старым trolley state observer;
adaptive_force добавляет2force laws. Adaptive_observer и adaptive_force_observer
дополнительно используют fitted endpoint v/a. Angular observer q/p остаётся общим.
Эти policy-ablations имеют одинаковые plant/noise cases, но разные trajectories;
качество идентификации endogenous к сигналам своего controller.

Процентные изменения между конкретными ablations:

{table(ablations)}

## Новый информационный контракт и оцениватель

Всем доступен буфер2s measured x и собственных уже применённых acceleration
commands на50ms grid. Angular history по-прежнему41samples100Hz (.4s). Static
методы не оценивают actuator по длинному буферу. Это явное расширение памяти,
поэтому опыт не является чистой заменой архитектуры старого41sample forecaster.
True v/a, параметры plant и будущие измерения контроллер не получает.

Profiled model: first-order lag tau и scalar gain K, initial x/v/a nuisance.
Для8tau grid values unit-gain forced response точно интегрируется. Initial
state basis[1,t,tau*(t-tau*(1-exp(-t/tau)))] профилируется, gain оценивается ridgeLS
с prior1/clips[.5,1.6]. Minimum21positions, informationRMS>=.0004m, fitRMSE<=.001m,
update раз в.2s, EMA.25; иначе prior/последняя оценка. Deadzone не fitted.

Endpoint observer при filtered tau/K профилирует initial state на past window
и вычисляет current v/a. Применяется после valid fit при текущем RMSE<=.001m;
clips v±.6,a±1.5, measured current x сохраняется. Только trolley initial state
меняется; angular observer остаётся прежним. Всё фиксировано внутри решения MPC.

Диагностика по истинному plant ПОСЛЕ решения (не вход controller):

{table(diagnostics)}

Tau/gain errors у adaptive методов описывают effective fit; из-за unmodelled
deadzone он не обязан совпадать с истинными параметрами. У static банков
id_tau/id_gain — nominal point prior, не единственный параметр всего банка.
Information threshold относится к residualized forced response и не гарантирует
идентифицируемость tau. При почти постоянной команде tau может смешиваться
с nuisance initial state и определяться prior. Отдельного fixed-parameter
2s-observer baseline здесь нет: влияние длинной памяти/сглаживания не полностью
отделено от fitted dynamics.

Update/use fractions показывают, сколько раз qualification реально пропустила
обновление или новый observer; low-information окна не объявляются успешным fit.

Связь с первичными работами: [Åström/Wittenmark1973](https://portal.research.lu.se/en/publications/on-self-tuning-regulators-2/)
совмещают LS estimator и self-tuning regulator. Наш finite-window profiled fit
и MPC не повторяют их алгоритм/теоремы. Связанный1972technical report отличается
от journal1973. [Gevers2005](https://perso.uclouvain.be/michel.gevers/PublisMig/P78.pdf)
рассматривает роль цели управления и информативности модели. Наша heuristic
qualification не доказывает persistent excitation, consistency или stability.

## Область проверки

L=.25/.62/1.07 × omega ratio.87/1.13 × tau.06/.19 × gain.78/1.22 =24contexts,
каждый exact/noisy×pulse absent/present. Balanced nuisance: damping.02/.08,
deadzone0/.02,effective/geometric forcing, initialangle3/5deg. Pulse.16rad/s²
на[2.45,2.85)s, новый seed1207112026. Это новый grid прежнего simulator family,
не independent hardware. Nuisance назначения не независимый full factorial.
Development L=.34/.54/.74×angle2/6deg — qualification без tuning.

Paired metrics имеют равные веса96episodes; context bootstrap resamples24contexts
целиком, не correlated ticks. Все негативные cases/violations сохранены. ISE
на50ms grid, peak/violations также на100Hz. Contact/state clipping отсутствуют.
Native xr features0, NeuralActuator bypass: acceleration-command ablation.
Оригинальные real-forecast gates не отменяются этим синтетическим опытом.

## Runtime и проверка

Batch time, включая history assembly/online update/observer/solver:

{table(pd.DataFrame(timing))}

Isolated warm batch1 full decision,15измерений на метод, milliseconds:

{table(latency)}

Для каждого повтора восстанавливается одно и то же pre-tick estimator state.
Его восстановление и чтение файлов не timed; текущий online update timed.
Выбраны ticks0/40/100 (update ticks), а не только дешёвые промежуточные тики.
Amortization — throughput, не latency. Simulation офлайн; wall-clock delay
сверх committed tick не действует на plant. Это не WCET/scheduling guarantee.

164unit tests прошли. Все {v['episodes']}episodes повторно интегрированы,
предикторные планы {v['decisions']}решений replayed. Online identifier state
восстановлен последовательно из measured position/applied commands, а не взят
из сохранённых estimates. SHA, bounds, pipeline, shared noise/disturbances,
warm-cost monotonicity, halfstepRK и {v['original_files']['checked_files']}original
files проверены. Full replay не перезапускал optimizer. Первоначальный unit-test ожидал новый
observer сразу после первого EMA update, до fit qualification; тест исправлен
до preregistration, финальный лог adaptive_actuator_tests_final.log содержит164PASS.
Все fitted traces
и raw100Hz trajectories опубликованы; neural weights не переобучались.

## Что ещё необходимо проверить

Структура истинного actuator в этом simulator близка к модели идентификатора:
первый порядок с неизвестными числами и дополнительной deadzone. Поэтому
масштаб выигрыша не следует переносить на другой привод. Нужен новый frozen
контроллерный тест с delay/двумя lag/изменением gain и friction, а также
fixed-parameter observer с той же2s памятью. Последний отделит пользу длинной
истории от пользы online parameter adaptation. Native-command опыт требует
предварительного согласования xr, actuator и sample timing. Эти дополнительные
испытания ещё не выполнены; существующие neural weights/default не меняются.
'''
    doc.write_text(text)
    report=dict(provenance_sha256=sha256(a.run/'provenance.json'),report_source_sha256=sha256(Path(__file__)),document=str(doc),document_sha256=sha256(doc),
        verification_sha256=sha256(out/'verification.json'),aggregation_helper_sha256=sha256(Path('scripts/report_angular_control.py')),runtime_batch1_sha256=sha256(out/'runtime_batch1.csv'),acceptance=gates,
        evaluation_episodes=864,development_episodes=54,online_actuator_identification=True,neural_training=False,native_controller_evaluated=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(summary.to_string(index=False));print('GATES',gates)

if __name__=='__main__':main()
