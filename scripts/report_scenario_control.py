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
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,default=Path('artifacts/runs/scenario_control_20261107'))
    ap.add_argument('--output',type=Path,default=Path('results/scenario_control_20261107'));a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
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
    summary=rows.groupby(['stage','method'],as_index=False)[METRICS+['settled','solver_nonfinite','fallbacks']].mean()
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
    methods=cfg['methods'];fig,axes=plt.subplots(1,3,figsize=(16,4.8),layout='constrained');colors=['#7b8790','#237c91','#267865','#668d3c','#ba7731','#9c577c']
    for ax,metric,label in zip(axes,['realized_cost','tracking_ise_m2_s','sway_ise_deg2_s'],['Realized task cost','Tracking ISE, m²·s','Sway ISE, deg²·s']):
        ax.bar(methods,[pivot.loc[m,metric] for m in methods],color=colors);ax.set_title(label);ax.tick_params(axis='x',rotation=35,labelsize=9);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('New finite-scenario / frozen-residual test: 96 paired cases per method');fig.savefig(out/'control_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(4,3,figsize=(15,12),layout='constrained');traces={m:np.load(a.run/'evaluate'/f'{m}.npz') for m in methods};grid=np.arange(121)*.05
    for row,case in enumerate([0,31,64,95]):
        for m,color in zip(methods,colors):
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
    doc=Path('docs/SCENARIO_CONTROL_2026_11_07_RU.md')
    text=f'''# Неопределённость модели и physical Jacobian: 7 ноября 2026

Завершены36development и576evaluation эпизодов:96на каждый из6методов.
Веса не переобучались. Новый grid отличается от предыдущего control-опыта;
настройки зафиксированы до запуска в `{p['git_head']}` и после development
не менялись. Все пять альтернатив публикуются, а не выбираются по evaluation.

## Основные результаты на новом наборе

{table(pivot.reset_index()[view])}

Процентные изменения относительно nominal physical (меньше ошибки/стоимость лучше):

{table(effects)}

Парные выигрыши по реализованной стоимости:

{table(win_table)}

Меньшая средняя стоимость может отражать улучшение нескольких тяжёлых случаев.
Поэтому среднее не заменяет отдельную проверку большинства парных выигрышей.

Фиксированные практические gates: `{json.dumps(gates,ensure_ascii=False)}`.
Требовалось одновременно улучшить средние cost/sway/tracking, удержать final
error в1%, не увеличить raw100Hz violations и выиграть cost в большинстве96cases.
Это критерии этой конечной синтетической задачи, не статистический сертификат.

## Какие методы проверены

Nominal physical — общий reference. Frequency усредняет cost по частотам
.85/1/1.15 от номинальной. Joint добавляет3пары(tau,gain) и2закона forcing,
всего18сценариев. Joint одновременно меняет actuator и forcing uncertainty;
их отдельные вклады этот набор ablations не разделяет. Spread дополнительно
штрафует .25 variance угла сценариев.
Finite scenario mean не является minimax, а spread — не chance constraint.
Сетка не калибрована как распределение; часть истинных lag лежит вне неё.

Hybrid вычисляет bounded offset clip(theta_adapter(warm)−theta_physical(warm),
±.015rad) один раз на тик. Внутри solver он фиксирован, поэтому command Jacobian
задаёт физика. Joint_hybrid применяет тот же offset к18сценариям. Это новые
controller predictors, а не переименование исходного end-to-end forecaster.
Старые forecast/load gates и default portable bundle от этого не меняются.

У всех одинаковые measured history41×100Hz, causal observer, backward v/a fits,
committed command delay .05s, horizon30,6блоков, input box±.8, trust region±.2,
200FISTAsteps и fractions0/.25/.5/1. Frozen-offset features аффинны; Jacobian
восстановлен unit central probes, без small-epsilon cancellation. Nominal
контроль тоже использует этот solver, поэтому сравнивать его абсолютные числа
с прошлой сеткой96cases как прямой прогресс нельзя.

Никаких true v/rate/a/future measurements в controller. Native xr features0,
NeuralActuator bypass. Это отдельная acceleration-command angular ablation,
не полноценный native-command пилот и не hardware validation.

## Основание из исследований

[Cautious MPC Hewing et al.](https://arxiv.org/abs/1705.10702) учитывает модельную
неопределённость при планировании. Здесь нет GP/вероятностного распространения
или гарантий этой работы; использована только общая идея учитывать uncertainty.
[Mayne et al.](https://www.sciencedirect.com/science/article/abs/pii/S0005109899002149)
рассматривают условия свойств constrained MPC. Наша конечная сетка и box limit
не устанавливают выполнение условий устойчивости/инвариантности. Подробности
и ограничения были записаны в SCENARIO_CONTROL_DESIGN_20261107.md до запуска.

## Набор условий и статистические пределы

24контекста L=.29/.64/1.11 × frequency ratio .86/1.14 × tau .075/.205 × gain
.85/1.15. Parity nuisance: damping .015/.09, deadzone0/.025, effective/geometric
forcing, initialangle2.5/5.5deg. Это сбалансированные назначения, не независимый
full factorial nuisance. Каждый контекст exact/noisy×pulse absent/present.
Pulse .17rad/s² на[2.65,3)s, новый seed1007112026. Шумы и disturbances одинаковы
между controller. .205s lag выходит за проектную сетку .07/.1/.18.
Development:6условий на метод, L=.36/.56/.76×angle2/6deg, без tuning.

Mean metrics — равные веса96эпизодов; group tables и все отдельные эпизоды
сохранены. context_bootstrap.csv resamples24plant contexts целиком, включая
4sensor/disturbance cases; интервалы описывают эту выбранную семью симулятора,
не новую популяцию устройств. Все rail/angle/speed violations сохранены.
ISE на50ms grid, отдельные пики/нарушения на100Hz. Contact/clipping состояния нет.

## Время и воспроизведение

Batch runtime:

{table(pd.DataFrame(timing))}

Изолированное warm batch1 полное решение,15измерений/метод, milliseconds:

{table(latency)}

Амортизация batch времени — throughput, не single-controller latency.
Ни один замер не является WCET/scheduling гарантией. Simulation офлайн: кроме
объявленного committed тика wall-clock solver delay на plant не действует.
Batch1/batch96 округление сохранено в runtime_batch1.csv.

157unit tests пройдены. Повторно интегрированы {v['episodes']}эпизодов,
воспроизведены выбранные forecasts/costs для {v['decisions']}решений с точной
реконструкцией warm plan и frozen hybrid offset. Проверены все inputs/source SHA,
paired noise/disturbances, pipeline, bounds, warm-cost monotonicity, causal
estimates, половинный RK4 шаг и {v['original_files']['checked_files']}original files.
Полный replay НЕ перезапускал optimizer. Все таблицы и trajectories сохранены,
первичные результаты/неудачи не удаляются. Existing saved weights неизменны.
'''
    doc.write_text(text)
    report=dict(provenance_sha256=sha256(a.run/'provenance.json'),report_source_sha256=sha256(Path(__file__)),document=str(doc),document_sha256=sha256(doc),
        verification_sha256=sha256(out/'verification.json'),aggregation_helper_sha256=sha256(Path('scripts/report_angular_control.py')),runtime_batch1_sha256=sha256(out/'runtime_batch1.csv'),acceptance=gates,
        evaluation_episodes=576,development_episodes=36,neural_training=False,native_controller_evaluated=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(summary.to_string(index=False));print('GATES',gates)

if __name__=='__main__':main()
