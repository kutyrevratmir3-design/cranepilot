"""Scope qualification with explicit old roles and equal-scenario regime summaries."""
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

RUN=Path('artifacts/runs/active_qualification_20261107');OUT=Path('results/active_qualification_20261107')
def main():
    if (OUT/'report.json').exists():raise FileExistsError('Retain earlier report')
    p=json.loads((RUN/'provenance.json').read_text());v=json.loads((OUT/'verification.json').read_text());assert v['passed'] and v['provenance_sha256']==sha256(RUN/'provenance.json');verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);cfg=p['config'];metrics=pd.read_csv(RUN/'metrics.csv');coverage=pd.read_csv(RUN/'coverage.csv');valid=metrics[metrics.windows>0];cases=valid.groupby(['role','group','case','scenario','method'],as_index=False).agg(mae_deg=('mae_deg','mean'),rmse_deg=('rmse_deg','mean'),windows=('windows','first'));records=cases.groupby(['role','group','scenario','method'],as_index=False)[['mae_deg','rmse_deg']].mean();cases.to_csv(OUT/'case_summary.csv',index=False);records.to_csv(OUT/'record_summary.csv',index=False);summary=records.groupby(['role','group','method'],as_index=False).agg(mae_deg=('mae_deg','mean'),scenarios=('scenario','nunique'));summary.to_csv(OUT/'summary.csv',index=False)
    primary=records[records.role==cfg['primary_role']];comparisons=[]
    for group,frame in primary.groupby('group'):
        m=frame.pivot(index='scenario',columns='method',values='mae_deg');ref=m.strong_arx
        for method in m:
            a=m[method];comparisons.append(dict(group=group,method=method,mae_deg=float(a.mean()),reference_mae_deg=float(ref.mean()),change_percent=float(100*(a.mean()/ref.mean()-1)),scenario_wins=int((a<ref).sum()),scenarios=len(m)))
    compare=pd.DataFrame(comparisons);compare.to_csv(OUT/'old_evaluation_comparisons.csv',index=False);cov=coverage.groupby(['role','group','scenario'],as_index=False)[['windows','total_windows']].mean();cov['fraction']=cov.windows/cov.total_windows;cov.to_csv(OUT/'coverage_summary.csv',index=False);primary_cov=cov[cov.role==cfg['primary_role']];primary_cov.to_csv(OUT/'old_evaluation_coverage.csv',index=False)
    regime_order=cfg['regime_names'];plot=summary[(summary.role==cfg['primary_role'])&summary['group'].isin(regime_order)].pivot(index='group',columns='method',values='mae_deg').reindex(regime_order);fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');x=np.arange(len(plot));width=.25
    for j,(method,label,color) in enumerate([('strong_arx','Frozen ARX','#3f7183'),('frozen_modal','Frozen modal','#b8a18b'),('frozen_blend','Frozen blend','#bb8247')]):axes[0].bar(x+(j-1)*width,plot[method],width,label=label,color=color)
    axes[0].set_xticks(x,['Low past\nmotion','Drive\nactive','Length\nactive','Drive +\nlength active']);axes[0].set_ylabel('MAE (degrees), equal scenarios/axes/horizons');axes[0].set_title('Old evaluation records; differing regime coverage');axes[0].legend(frameon=False)
    old=primary_cov[primary_cov['group']=='old_numerical_scope'].set_index('scenario');axes[1].bar([str(s) for s in old.index],old.fraction,color='#648b6d');axes[1].set_ylim(0,1);axes[1].set_xlabel('Previously opened evaluation scenario');axes[1].set_ylabel('Fraction of post-4s origins');axes[1].set_title('Previous numerical scope coverage')
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Frozen-kernel qualification outside the declared gate; no fitting');fig.savefig(OUT/'qualification.png',dpi=170);plt.close(fig)
    compact=compare[compare.method.isin(['strong_arx','frozen_modal','frozen_blend'])];doc=Path('docs/ACTIVE_QUALIFICATION_2026_11_07_RU.md')
    doc.write_text(f'''# Проверка границ: движение привода и подъём троса

Протокол/код до оценки: `{p['git_head']}`. Обучение, выбор модели и изменение
публичногоAPIgate не выполнялись. Все33записи уже открыты: это ретроспективная
квалификация существующихkernels, не новая независимая проверка.

## Результаты на прежних7оценочных записях

{table(compact)}

Основные4режима вычислены исключительно по прошлым401отсчётам4s:
low_past_motion; drive_active(диапазонx>2mmилиJ>2mrad); length_active(L>2mm);
оба движения. Это метки наблюдённой истории, не доказательство отсутствия
forcing. Ни угол будущей ошибки, ни будущая фактическаяL/позиция не выбирают
группу и не исключают окна. Каждая группа имеет своё число сценариев;
еёсреднее не является оценкой тойже популяции, что другаягруппа.

Покрытие по каждому сценарию:

{table(primary_cov)}

old_numerical_scope восстанавливает прежние численные условия(включая
timestamp≥25s и постоянную будущую записанную reference). Это дополнительная
описательная метка, не вход прогноза и не доказательство заранее известного
feedbackoutput. Все новые прогнозы работают только по истории; plannedfuture
reference или actualfutureL в них не передавались.

## Что именно проверено

53сценарий/ось сочетания, {p['packets']}пакетов, каждый из них оценён всеми5
методами без target/amplitude-based исключений. Отдельно сохраненыTRAIN,VAL,
old_evaluation; TRAIN/VALне выдаются за heldout. Семь горизонтов .10–1.50s.
Среднее сначала по горизонтам, затем по осям внутри сценария, затем равный вес
целых сценариев. Empty regimes остаются в coverage/metrics сwindows=0иNaN,
не становятся нулевойошибкой. Overlappingwindowsне считаются независимыми trials.

Изначальный4sпрофиль требует записанной истории. Первые4sкаждой записи здесь
неоцениваемы; нулеваяпредыстория невыдумывалась. Для будущей записи полезно
сохранить хотябы4sдо начала движения. Это отдельное ограничение от ошибки MAE.

Важное программное свойство: modal/softblend и сохранённыйsimpleARXв этом
опыте используют прошлыеуглы/L, без futurecommandplan. Они не определяют
реакцию на новую оптимизируемую команду и сами по себе не квалифицируютMPC.
ОсновнойneuralCranePilotимеет другой входнойконтракт; его оценки с.4sисторией
нельзя напрямую подменить этими4sцифрами. Семантика native xr/Jr и неизвестная
будущая фактическаяL остаются существенными для активного управления.

## Проверки и следующий шаг

210unit tests PASS перед запуском. Из исходных сохранённых сигналов и frozen
весов повторены {v['forecast_comparisons']} forecasts, все regime/coverage/head
metrics независимо пересчитаны. Все1281оригинальные main/APHYNITY файла целы.
Public APIgateнеослаблен: outsideoriginaltaskон по-прежнему возвращаетNaN.
Negative out-of-scope scores сохранены. Архив/таблицы вresults/active_qualification_20261107.

Эта проверка уточняет границы применимости и направления дальнейшей разработки.
Она не закрывает исходную цель: нужны подходящие новые физические записи и
заранее зафиксированная независимая оценка. Новыеhoist/actuator методы требуют
отдельного протокола; нельзя передавать им измеренную будущуюдлину или ускорение.
''')
    result=dict(document=str(doc),document_sha256=sha256(doc),provenance_sha256=sha256(RUN/'provenance.json'),verification_sha256=sha256(OUT/'verification.json'),report_source_sha256=sha256(Path(__file__)),fitting_performed=False,scope_expansion_performed=False,api_gate_changed=False,independent_confirmation=False,final_goal_achieved=False)
    (OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(compact.to_string(index=False));print('INDEPENDENT_CONFIRMATION',False,'API_GATE_CHANGED',False)
if __name__=='__main__':main()
