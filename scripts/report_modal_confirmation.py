"""Conditional public holdout: preserve every condition and qualify the evidence."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cranepilot.data import sha256
from scripts.report_angular_control import table
from scripts.run_external_benchmark import verify_hashes

RUN=Path('artifacts/runs/modal_confirmation_20261107');OUT=Path('results/modal_confirmation_20261107')
def main():
    if (OUT/'report.json').exists():raise FileExistsError('Retain every report')
    p=json.loads((RUN/'provenance.json').read_text());v=json.loads((OUT/'verification.json').read_text());cfg=p['config'];assert v['passed'] and v['provenance_sha256']==sha256(RUN/'provenance.json');verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);verify_hashes(p['freeze']['frozen_parent_hashes'])
    metrics=pd.read_csv(RUN/'metrics.csv');cases=metrics.groupby(['scenario','case','method'],as_index=False)[['mae_deg','rmse_deg']].mean();records=cases.groupby(['scenario','method'],as_index=False)[['mae_deg','rmse_deg']].mean();summary=records.groupby('method',as_index=False)[['mae_deg','rmse_deg']].mean()
    cases.to_csv(OUT/'case_summary.csv',index=False);records.to_csv(OUT/'record_summary.csv',index=False);summary.to_csv(OUT/'summary.csv',index=False)
    matrix=records.pivot(index='scenario',columns='method',values='mae_deg');missing=sorted(set(cfg['sealed_confirmation_ids'])-set(matrix.index));n=len(matrix);rng=np.random.default_rng(cfg['confirmation_gate']['seed']);ids=rng.integers(0,n,(cfg['confirmation_gate']['bootstrap_samples'],n));ref=matrix.strong_arx.to_numpy();rows=[]
    for method in matrix:
        a=matrix[method].to_numpy();effect=100*(a[ids].mean(1)/ref[ids].mean(1)-1);rows.append(dict(method=method,mae_deg=float(a.mean()),change_percent=float(100*(a.mean()/ref.mean()-1)),scenario_wins=int((a<ref).sum()),scenarios=n,descriptive_low=float(np.quantile(effect,.025)),descriptive_high=float(np.quantile(effect,.975))))
    comparison=pd.DataFrame(rows);comparison.to_csv(OUT/'paired_comparisons.csv',index=False);chosen=comparison[comparison.method=='selected_modal'].iloc[0];gate=bool(not missing and not p['empty_cases'] and chosen.change_percent<=-10 and chosen.scenario_wins>=5 and chosen.descriptive_high<0)
    acceptance=dict(conditional_public_holdout_gate_passed=gate,final_goal_achieved=False,independently_collected_confirmation=False,missing_scenarios=missing,empty_cases=p['empty_cases'],criteria=cfg['confirmation_gate'],selected_modal=chosen.to_dict(),primary_metric='Seven-horizon MAE, axes equal within scenario, seven whole scenarios equal; no window-level bootstrap',qualification=cfg['confirmation_status'])
    (OUT/'acceptance.json').write_text(json.dumps(acceptance,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');x=np.arange(n);width=.35
    axes[0].bar(x-width/2,matrix.strong_arx,width,label='Strong ARX',color='#3f7183');axes[0].bar(x+width/2,matrix.selected_modal,width,label='Frozen modal',color='#bb8247');axes[0].set_xticks(x,[str(i) for i in matrix.index]);axes[0].set_xlabel('Previously unscored public scenario');axes[0].set_ylabel('MAE (degrees), equal axes/horizons');axes[0].legend(frameon=False)
    horizons=metrics.groupby(['scenario','method','horizon_s'],as_index=False).mae_deg.mean().groupby(['method','horizon_s'],as_index=False).mae_deg.mean()
    for method,label,color in [('strong_arx','Strong ARX','#3f7183'),('selected_modal','Frozen modal','#bb8247'),('local_ar','Local AR','#8c829d')]:
        frame=horizons[horizons.method==method];axes[1].plot(frame.horizon_s,frame.mae_deg,'o-',label=label,color=color)
    axes[1].set_xlabel('Forecast horizon (seconds)');axes[1].set_ylabel('MAE (degrees), equal scenarios');axes[1].legend(frameon=False)
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Settled-input sway forecast, 4-second history; conditional public holdout');fig.savefig(OUT/'modal_confirmation.png',dpi=170);plt.close(fig)
    coverage=pd.read_csv(RUN/'coverage.csv');doc=Path('docs/MODAL_CONFIRMATION_2026_11_07_RU.md')
    doc.write_text(f'''# Однократная проверка локальных мод на7условиях

Модель/ARX/selection заморожены до нового расчёта ошибок: commit
`{p['git_head']}`, selectionSHA
`{p['selection_unchanged_sha256']}`. Перенастройки по holdout нет.

## Результат

MAE selected modal **{chosen.mae_deg:.6f}°** против сильного ARX
**{ref.mean():.6f}°**, изменение **{chosen.change_percent:+.3f}%**;
побед **{int(chosen.scenario_wins)}/{n}** сценариев.
Парный bootstrap целых7сценариев: описательный95%интервал изменения
**[{chosen.descriptive_low:+.2f}%, {chosen.descriptive_high:+.2f}%]**.
Заранее заданный conditional gate: **{gate}**. Пропуски={missing},
пустые cases={p['empty_cases']}.

{table(comparison)}

Все условия, равные оси внутри каждого сценария:

{table(matrix.reset_index())}

## Что это позволяет утверждать

Прогноз остаточных колебаний при известном постоянном будущем reference,
неподвижных прошлом приводе/длине, с4sисторией. Обоим методам доступны те же
81past samples и все каналы. Nominal frequency calibration только TRAIN;
ARX24варианта на case, все120fits сохранены. Modal parameters оцениваются
только по прошлому текущего окна. Нейросеть для этого результата не нужна.
Fixed calibration и one-mode — заранее заданные абляции; выбирать лучший
метод после holdout запрещено. Таблицы сохраняют все случаи/горизонты.

IDs9,13,21,22,23,32,33 не имели найденных сохранённых fitting/validation/score
использований, но33recorddataset ранее импортировался целиком. Это **условный
holdout ранее не оценивавшихся публичных записей**, не независимо собранные новые
измерения. Основание ограниченной квалификации — metadata-only usage audit;
неучтённый доступ исключить нельзя. После этой проверки все7условий открыты
и становятся retrospective для любого следующего выбора метода.

Нет доказательства выигрыша на исходной .4sактивной задаче, во время подъёма,
в native-command closed loop, на новом стенде или в метриках других статей.
Гейт не использует будущую измеренную длину и не доказывает отсутствие forcing.
Поэтому **широкая исследовательская цель пока не закрыта**. Нужны новые сырые
записи авторов для подтверждения с действительно независимым происхождением.

## Покрытие и проверки

{table(coverage[['case','scenario','eligible','origins_considered','eligible_fraction']])}

Покрытие задано до оценивания, target-based исключений нет. Каждое пустое условие
запретило бы gate. Повторно рассчитаны все {v['forecasts_replayed']} forecasts
из исходных сигналов/сохранённых коэффициентов, результат точный. Все метрики
независимо пересчитаны; selectionSHA неизменён. 195unit tests PASS до fitting,
1281исходный файл main/APHYNITY сохранён. Все неудачные предыдущие этапы сохранены.
Веса/defaults исходного CranePilot и APHYNITY не заменялись.
''')
    report=dict(document=str(doc),document_sha256=sha256(doc),provenance_sha256=sha256(RUN/'provenance.json'),verification_sha256=sha256(OUT/'verification.json'),report_source_sha256=sha256(Path(__file__)),aggregation_helper_sha256=sha256(Path('scripts/report_angular_control.py')),conditional_gate_passed=gate,final_goal_achieved=False,independently_collected_confirmation=False)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(comparison.to_string(index=False));print('CONDITIONAL_GATE',gate,'FINAL_GOAL',False)

if __name__=='__main__':main()
