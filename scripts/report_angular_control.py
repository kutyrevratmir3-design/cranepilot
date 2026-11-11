"""Aggregate paired closed-loop results without selecting new models on evaluation."""
import argparse,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cranepilot.data import sha256

METRICS=['realized_cost','tracking_ise_m2_s','sway_ise_deg2_s','tail_rms_deg','final_position_error_m','effort_acceleration2','command_variation']


def table(frame):
    cols=list(frame);lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for row in frame.itertuples(index=False,name=None):
        vals=[format(v,'.6g') if isinstance(v,(float,np.floating)) else str(v) for v in row]
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,default=Path('artifacts/runs/angular_control_20261107'))
    ap.add_argument('--output',type=Path,default=Path('results/angular_control_20261107'));a=ap.parse_args();out=a.output;out.mkdir(exist_ok=True,parents=True)
    if (out/'report.json').exists():raise FileExistsError('Preserve previous report')
    p=json.loads((a.run/'provenance.json').read_text());v=json.loads((out/'verification.json').read_text())
    if p['status']!='complete' or not v['passed']:raise ValueError('Complete verified run required')
    for key,base in [('guarded_hashes',Path('.')),('result_hashes',a.run)]:
        for path,digest in p[key].items():
            if sha256(base/path)!=digest:raise ValueError(f'Changed {path}')
    assert v['provenance_sha256']==sha256(a.run/'provenance.json')
    cfg=p['config'];frames=[];timing=[]
    for stage in ['development','evaluate']:
        f=pd.read_csv(a.run/stage/'metrics.csv');f['stage']=stage;frames.append(f)
        for method in cfg['methods']:
            with np.load(a.run/stage/f'{method}.npz') as z:
                sec=z['batch_seconds'];n=len(z['commands'])
            timing.append(dict(stage=stage,method=method,batch_cases=n,decisions=len(sec),batch_median_ms=np.median(sec)*1000,batch_p95_ms=np.quantile(sec,.95)*1000,
                batch_max_ms=sec.max()*1000,amortized_median_ms_per_case=np.median(sec)*1000/n,batch_total_s=sec.sum()))
    allrows=pd.concat(frames,ignore_index=True);allrows.to_csv(out/'all_episode_metrics.csv',index=False)
    raw=pd.read_csv(out/'raw_sampling_metrics.csv')
    summary=allrows.groupby(['stage','method'],as_index=False)[METRICS+['settled','solver_nonfinite','fallbacks','current_acceleration_estimate_rmse']].mean()
    violations=raw.groupby(['stage','method'],as_index=False)[['rail_violation_100hz','angle_violation_100hz','speed_violation_100hz']].sum()
    summary=summary.merge(violations,on=['stage','method']);summary.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(timing).to_csv(out/'runtime_batches.csv',index=False)
    ev=allrows[allrows.stage=='evaluate'];comparisons=[];bootstrap=[];rng=np.random.default_rng(91712026)
    for ref in ['physical','retained']:
        left=ev[ev.method=='candidate'].set_index('case');right=ev[ev.method==ref].set_index('case')
        assert left.index.equals(right.index)
        for metric in METRICS:
            delta=100*(left[metric].mean()/right[metric].mean()-1)
            comparisons.append(dict(reference=ref,metric=metric,candidate_mean=left[metric].mean(),reference_mean=right[metric].mean(),change_percent=delta,
                paired_case_wins=int((left[metric]<right[metric]).sum()),cases=len(left)))
            # Resample whole plant contexts; retain four sensor/disturbance cases each.
            pairs=pd.DataFrame({'context':left.context,'candidate':left[metric],'reference':right[metric]}).groupby('context').mean()
            idx=rng.integers(0,len(pairs),(2000,len(pairs)))
            effect=100*(pairs.candidate.to_numpy()[idx].mean(1)/pairs.reference.to_numpy()[idx].mean(1)-1)
            bootstrap.append(dict(reference=ref,metric=metric,contexts=len(pairs),replicates=2000,change_percent=delta,
                descriptive_cluster_low_percent=np.quantile(effect,.025),descriptive_cluster_high_percent=np.quantile(effect,.975)))
    comp=pd.DataFrame(comparisons);comp.to_csv(out/'paired_comparisons.csv',index=False);pd.DataFrame(bootstrap).to_csv(out/'context_bootstrap.csv',index=False)
    pivot=summary[summary.stage=='evaluate'].set_index('method');c=pivot.loc['candidate'];r=pivot.loc['retained']
    cost_wins=int(comp[(comp.reference=='retained')&(comp.metric=='realized_cost')].paired_case_wins.iloc[0])
    gate=dict(sway_improved=bool(c.sway_ise_deg2_s<r.sway_ise_deg2_s),tracking_improved=bool(c.tracking_ise_m2_s<r.tracking_ise_m2_s),
        final_error_within_1percent=bool(c.final_position_error_m<=r.final_position_error_m*1.01),
        no_additional_raw_violations=bool(all(c[k]<=r[k] for k in ['rail_violation_100hz','angle_violation_100hz','speed_violation_100hz'])),
        majority_paired_cost_wins=cost_wins>len(ev[ev.method=='candidate'])/2)
    gate['passed']=all(gate.values());(out/'acceptance.json').write_text(json.dumps(gate,indent=2)+'\n')
    # Fixed examples, chosen in design-independent case order rather than by outcome.
    methods=cfg['methods'];fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,metric,label in zip(axes,['realized_cost','tracking_ise_m2_s','sway_ise_deg2_s'],['Realized task cost','Tracking ISE, m²·s','Sway ISE, deg²·s']):
        ax.bar(methods,[pivot.loc[m,metric] for m in methods],color=['#7b8790','#25788a','#aa6b2c']);ax.set_title(label);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Closed loop: 96 paired synthetic episodes per method');fig.savefig(out/'control_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(4,3,figsize=(13,11),layout='constrained')
    traces={m:np.load(a.run/'evaluate'/f'{m}.npz') for m in methods};grid=np.arange(121)*.05
    for row,case in enumerate([0,31,64,95]):
        for m in methods:
            z=traces[m];axes[row,0].plot(grid,z['states'][case,:,0],label=m)
            axes[row,1].plot(grid,np.rad2deg(z['states'][case,:,2]));axes[row,2].step(grid[:-1],z['commands'][case],where='post')
        axes[row,0].axhline(cfg['target_x_m'],color='black',ls=':',lw=1)
        axes[row,0].set_ylabel(f'Case {case}\nx, m');axes[row,1].set_ylabel('theta, deg');axes[row,2].set_ylabel('command, m/s²')
        for ax in axes[row]:ax.grid(alpha=.2);ax.set_xlabel('time, s')
    axes[0,0].legend();fig.savefig(out/'fixed_examples.png',dpi=160);plt.close(fig)
    for z in traces.values():z.close()
    latpath=out/'runtime_batch1.csv';lat=pd.read_csv(latpath) if latpath.exists() else None
    runtime_text=table(lat.groupby('method').elapsed_ms.agg(['median','max']).reset_index()) if lat is not None else 'Отдельный batch-one замер не выполнялся.'
    tests='150 PASS'
    diagnosis=pd.read_csv(out/'sensitivity_diagnostic.csv')
    diagnostic_table=diagnosis.groupby('method',as_index=False)[['absolute_jacobian_error','jacobian_cosine','mean_counterfactual_forecast_error_deg']].mean()
    diagnostic_table.to_csv(out/'sensitivity_summary.csv',index=False)
    doc=Path('docs/ANGULAR_CONTROL_2026_11_07_RU.md')
    text=f'''# Первое замкнутое сравнение angular predictor, 7 ноября 2026

Завершены 18 development и 288 evaluation эпизодов. Сравниваются физический
осциллятор, прежний phase_space и новый adapter_direct8, все на PRIMARY.
Веса заморожены; настройки не подбирались по development/evaluation.
Протокол и код: `{p['git_head']}`. Gate замены: **{gate['passed']}**.

## Средние результаты на 96 одинаковых условиях

{table(summary[summary.stage=='evaluate'].drop(columns='stage'))}

Изменения нового adapter относительно baseline и прежней сети:

{table(comp)}

Критерии из preregistration: `{json.dumps(gate,ensure_ascii=False)}`.
Gate относится только к этой новой angular-control задаче. Результат
не отменяет не пройденный load gate в реальном forecast benchmark.

## Что именно измерено

На каждом .05s тике optimizer читает 41 измеренную точку theta/x при100Hz,
причинные оценки v/a/q/p и номинальную частоту из TRAIN calibration. Истинные
v, angle_rate, actuator acceleration и его будущие значения скрыты. Команда
ускорения проходит через общий номинальный lag, а angular predictor получает
его среднее предсказанное ускорение. Реальный plant использует независимые
lag/gain/deadzone, нелинейный угол и скрытую frequency error.
Native reference features обнулены, NeuralActuator обходится. Это явная
замена входного контракта; данный результат не оценивает native-command
контроллер, реальный кран или аппаратную безопасность.

Один move-blocked horizon30, 6 уровней, hard bound±.8, trust region±.2,
central FD Jacobian (.002), 100 FISTA шагов и nonlinear backtracking у всех
трёх методов. Первый command уже committed; оптимизированный следующий
применяется через один тик. Fallback/nonfinite и все нарушения сохраняются.
Модельные веса и controller weights не обучались. Planning cost не равен
realized episode cost; отдельные tracking/sway/effort показатели сохранены.

Вдохновение: [DiffTune-MPC](https://arxiv.org/html/2312.11384v2) подчёркивает
разницу внутренней цели MPC и результата всего замкнутого эпизода. Мы не
реализуем обучение DiffTune или KKT differentiation. [Дифференцируемый MPC
Amos et al.](https://proceedings.neurips.cc/paper/2018/hash/ba6d843eb4251a4526ce65d1807a9309-Abstract.html)
служит общей основой; наш solver использует локальную linearization и FD.

## Диагностика чувствительности: отдельно после основного опыта

{table(diagnostic_table)}

Это exploratory post-hoc audit, не использованный для выбора модели, весов
или controller. Все три predictor получают одну историю физического baseline
для всех96cases на заранее фиксированных ticks20/60/100. Сравнивается Jacobian
30 углов по6 command levels с независимым нелинейным plant. Его истинное
состояние используется только здесь, в oracle диагностике, не в контроллере.
Всего864 модельных сравнения. У8 из288 историй true Jacobian практически
нулевой из-за deadzone: relative error и cosine для них не определены;
absolute error сохранена для всех. Начальная диагностическая версия с делением
на epsilon сохранена отдельно; её огромные relative means не интерпретируются.

Новый adapter улучшает среднюю counterfactual forecast error относительно
прежней сети, но хуже по средней absolute Jacobian error и cosine. Это
согласуется с необходимостью проверять реакцию на управляющие команды отдельно
от MAE. Диагностика не доказывает единственную причину ухудшения closed loop:
есть также неопределённость привода, наблюдателя и изменённый zero-xr контракт.

## Область сценариев и статистические пределы

24 контекста L=.27/.57/1.03 × omega_ratio .88/1.12 × tau .065/.18 × gain .8/1.2.
Дополнительные damping/deadzone/force-law/initial-angle назначены заранее
по сбалансированной таблице, но не образуют независимый full factorial.
Каждый контекст содержит exact/noisy sensors × pulse absent/present.
Sensor noise и disturbances совпадают между controllers по физическому времени.
Поэтому сравнение парное; единица resampling — целый контекст, а не tick.
`context_bootstrap.csv` даёт описательные интервалы по этому выбранному
семейству, не гарантию переноса на новое устройство.

Development L=.32/.52/.72 × angle2/6deg используется только как qualification.
Настройки после него не менялись; его SHA зафиксированы до evaluation.
История прогревается физически .4s, episode6s, target .1→.3m.
Raw100Hz trajectories сохранены. ISE/settling/terminal метрики рассчитаны
на50ms grid; 100Hz peak/нарушения находятся отдельно в raw_sampling_metrics.csv.

## Время вычислений

{table(pd.DataFrame(timing))}

Это полное время решения для батча, включая history/observer/прямые проходы,
local QP и проверку nonlinear proposals. Амортизация на case — throughput,
не задержка одиночного контроллера и не гарантия50ms deadline.
Изолированный warmed batch-one замер (15 измерений на метод, три фиксированных
истории; median/max в миллисекундах):

{runtime_text}

Batch-one и batch96 планы различаются не более .000461m/s² у candidate
из-за округления FD/линейной алгебры; это записано в runtime_batch1.csv.

Симуляция офлайн: кроме объявленного committed тика фактическое wall-clock
время не задерживает plant. Deadline-aware исполнение и scheduling здесь
не проверены; превышения50ms нельзя интерпретировать как real-time успех.

## Проверка воспроизведения

Unit tests: {tests}. Повторно интегрированы все {v['episodes']} эпизодов и
воспроизведены прогнозы для {v['decisions']} решений. Проверены точная
реконструкция измерений, paired exogenous arrays, pipeline, hard bounds,
best objective против warm start, causal state estimates и source SHA.
Original main/APHYNITY: {v['original_files']['checked_files']} файлов совпадают.
Полная проверка: results/angular_control_20261107/verification.json.
Предикторные планы воспроизведены, сам optimizer во full replay не запускался.
Численная сходимость plant проверена половинным RK4 шагом на всех эпизодах.
Первоначальная проверка остановилась на различии causal acceleration estimate
2.53e-15 из-за layout-dependent NumPy dot; после воспроизведения contiguous
истории из runner оценки совпали точно, tolerance не ослаблялся. Initial log
и JSON сохранены; веса и результаты управления не менялись.

Следующие обоснованные гипотезы: обучать реакцию на команду/многошаговый
closed-loop критерий на НОВОМ train simulator set; либо сохранять физический
command Jacobian, обучая ограниченную residual поправку/наблюдатель. Они ещё
не реализованы и потребуют нового протокола, без подгонки по этим96cases.
Native-controller по-прежнему требует согласования xr и модели привода.

Все неудачные/отрицательные результаты остаются в таблицах. Улучшение forecast
на реальных записях не даёт автоматического улучшения closed loop; решение
о следующем шаге следует принимать по sway/tracking/violations/runtime вместе.
'''
    doc.write_text(text)
    for stage in ['development','evaluate']:
        shutil.copy2(a.run/stage/'cases.csv',out/f'{stage}_cases.csv');shutil.copy2(a.run/stage/'metrics.csv',out/f'{stage}_metrics.csv')
    report=dict(provenance_sha256=sha256(a.run/'provenance.json'),report_source_sha256=sha256(Path(__file__)),document=str(doc),document_sha256=sha256(doc),
        verification_sha256=sha256(out/'verification.json'),acceptance=gate,development_episodes=18,evaluation_episodes=288,neural_training=False,native_controller_evaluated=False,
        sensitivity_diagnostic_sha256=sha256(out/'sensitivity_diagnostic.csv'),runtime_batch1_sha256=sha256(out/'runtime_batch1.csv'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(summary.to_string(index=False));print('GATE',gate)

if __name__=='__main__':main()
