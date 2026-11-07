"""Frozen router results, all regime losses, and explicitly synthetic stress sensitivity."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cranepilot.data import sha256
from scripts.run_regime_fusion import RUN,OUT
from scripts.report_drive_conditioned import balanced_cases
from scripts.report_angular_control import table
from scripts.run_external_benchmark import verify_hashes

def main():
    if (OUT/'report.json').exists():raise FileExistsError('Preserve previous report')
    p=json.loads((RUN/'development.json').read_text());q=json.loads((RUN/'retrospective/provenance.json').read_text());s=json.loads((RUN/'selection.json').read_text());v=json.loads((OUT/'verification.json').read_text());assert v['passed'];verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);verify_hashes(q['result_hashes'],RUN/'retrospective');m=pd.read_csv(RUN/'retrospective/metrics.csv');valid=m[m.windows>0];axes=valid.groupby(['condition','group','case','scenario','method'],as_index=False).agg(mae_deg=('mae_deg','mean'),windows=('windows','first'));records=axes.groupby(['condition','group','scenario','method'],as_index=False).mae_deg.mean();summary=records.groupby(['condition','group','method'],as_index=False).agg(mae_deg=('mae_deg','mean'),scenarios=('scenario','nunique'));axes.to_csv(OUT/'axis_record_summary.csv',index=False);records.to_csv(OUT/'record_summary.csv',index=False);summary.to_csv(OUT/'summary.csv',index=False);balanced=[];case_rows=[]
    for condition,frame in m.groupby('condition'):
        cases,_=balanced_cases(frame);cases['condition']=condition;case_rows.append(cases);b=cases.groupby('method').mae_deg.mean()
        for method,value in b.items():balanced.append(dict(condition=condition,method=method,balanced_mae_deg=float(value)))
    balanced=pd.DataFrame(balanced);balanced.to_csv(OUT/'balanced_summary.csv',index=False);pd.concat(case_rows,ignore_index=True).to_csv(OUT/'balanced_case_summary.csv',index=False);clean=summary[summary.condition=='clean'];pivot=clean.pivot(index='group',columns='method',values='mae_deg');selected=s['selected'];keys=['legacy','active','hard_context','selected','frozen_arx','active_arx'];primary=clean[clean.method.isin(keys)];primary.to_csv(OUT/'clean_comparison.csv',index=False);stress=summary[(summary['group']=='all')&summary.method.isin(keys)];stress.to_csv(OUT/'stress_comparison.csv',index=False);val=pd.DataFrame([dict(method=key,balanced_mae_deg=value,quiet_feasible=s['quiet_feasible'].get(key),selected=key==selected) for key,value in s['validation_means'].items()]);val.to_csv(OUT/'validation_summary.csv',index=False)
    detailed=valid.pivot(index=['condition','group','case','scenario','horizon_s'],columns='method',values='mae_deg').reset_index();detailed['change_vs_legacy_deg']=detailed['selected']-detailed['legacy'];detailed.to_csv(OUT/'all_paired_changes.csv',index=False);losses=detailed[(detailed.condition=='clean')&(detailed['group']=='all')].sort_values('change_vs_legacy_deg',ascending=False);losses.to_csv(OUT/'head_losses_and_gains.csv',index=False);caseclean=axes[(axes.condition=='clean')&(axes['group']=='all')].pivot(index=['case','scenario'],columns='method',values='mae_deg');caseclean['change_vs_legacy_deg']=caseclean['selected']-caseclean['legacy'];caseclean.to_csv(OUT/'case_record_changes.csv');scenario=records[(records.condition=='clean')&(records['group']=='all')].pivot(index='scenario',columns='method',values='mae_deg');wins=int((scenario['selected']<scenario['legacy']).sum());change=100*(pivot.loc['all','selected']/pivot.loc['all','legacy']-1);hard_better=bool(pivot.loc['all','hard_context']<pivot.loc['all','selected']);regimes=p['qualification']['regime_names'];plot=pivot.reindex(regimes);conditions=q['conditions'];st=stress.pivot(index='condition',columns='method',values='mae_deg').reindex(conditions);fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained');width=.2
    for j,(key,label,color) in enumerate([('legacy','Legacy blend','#3f7183'),('active','Active blend','#b8a18b'),('hard_context','Hard context','#648b6d'),('selected','VAL-selected convex fit','#bb8247')]):
        for ax,data in [(axs[0],plot),(axs[1],st)]:ax.bar(np.arange(len(data))+(j-1.5)*width,data[key],width,label=label,color=color)
    axs[0].set_xticks(np.arange(4),['Low past\nmotion','Drive\nactive','Length\nactive','Drive +\nlength']);axs[0].set_title('Old records, clean history');axs[1].set_xticks(np.arange(4),['Clean','Angle jitter\n±q/2','Angle jitter\n±q','Angle lag\n50ms']);axs[1].set_title('Synthetic history perturbations; all origins')
    for ax in axs:ax.set_ylabel('MAE (degrees), equal scenarios/axes/horizons');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axs[0].legend(frameon=False);fig.suptitle('Retrospective routing qualification; no new independent measurements');fig.savefig(OUT/'regime_fusion.png',dpi=170);plt.close(fig)
    b=balanced[balanced.condition=='clean'].set_index('method').balanced_mae_deg;latency=pd.DataFrame(v['single_packet_timing']);latency.to_csv(OUT/'cpu_timing.csv',index=False);bundle=Path('artifacts/regime_bundle_20261107');count=sum(f.is_file() for f in bundle.rglob('*'));size=sum(f.stat().st_size for f in bundle.rglob('*') if f.is_file());diagnostic='Простой hard-context оказался лучше выбранного convex-fit на общем старом MAE. Выбор по oldevaluation не менялся; преимущество усложнения не подтверждено.' if hard_better else 'На общем старом MAE выбранный convex-fit лучше простого hard-context; это ретроспективное сравнение, а не независимое доказательство пользы усложнения.';doc=Path('docs/REGIME_FUSION_2026_11_07_RU.md')
    doc.write_text(f'''# Причинный выбор режима и смешивание прогнозов

Протокол/численный код до выбора: `{p['git_head']}`. Selection и data-free bundle
заморожены и закоммичены до нового replay. Все33записи уже многократно открыты;
здесь нет новой независимой проверки, аппаратного контроля или SOTA-результата.

## Метод

Четыре frozen-прогноза: прежнийARX, active-trainedARX, free-modal и forced-modal.
Новых TRAINfits не было. Из первых3sдоступной4sистории модели предсказывают
последнюю уже наблюдённую1s; шесть ошибок задают выпуклые веса с ridge к
контекстному prior. Никаких целей после текущего момента, будущих измеренных
позиций/L, будущих commands или scenario-ID в весах нет. Не используется
recurrent memory за пределами техже4s. Веса не считаются вероятностью верности.

Quiet определяется только по диапазонам прошлых401отсчётов x/J и прежнему
порогу2mm/2mrad. Там выдаётся прежний прогноз бит в бит. Это свойство конструкции,
не новая эмпирическая гарантия точности или устойчивости замкнутого контура.
В остальных окнах сравниваются fixed routing, inverse-error и constrained fit.
Никакая длина или угол ошибки не выбирают данные для исключения.
Legacy в таблицах означает ранее оценённый history-only kernel, включая
исследовательские origins вне прежнего publicAPIgate. Прежний API с knownplan
и ограничениями не ослаблен; новый predictor имеет отдельный контракт.

## Выбор по validation

Один вариант из11: **{selected}**. Legacy/change по balancedVAL:
{s['validation_legacy_change_percent']:.2f}%, casewins {s['legacy_case_wins']}/5.
Registered developmentgate: **{s['development_gate_passed']}**.
Quiet feasibility проверена на каждом VALокне, а не только на среднем.

{table(val)}

## Старые7оценочных сценариев

Общий MAE: {pivot.loc['all','selected']:.6f}° vs legacy
{pivot.loc['all','legacy']:.6f}° ({change:.2f}%), scenario wins {wins}/7.
Агрегация: равные горизонты, оси внутри сценария, затем равные целые сценарии;
каждый режим имеет своё покрытие. Overlapping windows не являются trials.

{table(primary)}

Balanced-score отдельно даёт {b['selected']:.6f}° vslegacy {b['legacy']:.6f}°:
quiet/drive поровну внутриaxis/record, равныеrecords внутриcase, пятьcases.
Эту величину нельзя напрямую сравнивать с общим MAE выше.

{diagnostic}

Сохранены все per-head/per-record gains и losses, а не только выигрышные:

{table(caseclean[['legacy','active','hard_context','selected','change_vs_legacy_deg']].reset_index())}

## Синтетическая чувствительность

{table(stress)}

Два равномерных jitterуровня ±q/2,±q добавлены только к истории угла всех
экспертов; q — исходный encoderquantum. Отдельно угол запаздывает на50ms.
Будущие цели — прежние encoderизмерения; метки режимов из прежних support/L
историй. Это пакетные synthetic perturbations, не новые physicalrecords,
не независимые повторы и не воспроизведение реального sensor-noise процесса.
Past-error truth тоже берётся из предоставленной noisy/lagged истории.
Quiet equality означает равенство legacy на тойже возмущённой истории,
а не равенство чистому legacyforecast. Ни одна стресс-оценка не меняла выбор.

## Проверка и переносимость

226unit tests PASS перед выбором; simplex-face solver независимо сверён с
SciPy constrained optimization. Development replay28125forecastcomparisons;
full raw/API replay {v['forecast_comparisons']} сравнений на VAL/oldevaluation,
включая все три stressconditions. Все метрики и selection пересчитаны;
quietconstraint сохранён, все1281main/APHYNITYоригинала целы.

Bundle `artifacts/regime_bundle_20261107`: {count}files,{size}bytes, безrawdata
и ancestralruns. `FrozenRegimePilot.forecast(history,family=...,axis=...)`
принимает только(N,401)past100Hznewestfirst: угол/L/x/J/native-reference
истории в соответствии с family. Futureplanаргумента нет. Loader проверяет
хеши всех весов и численных исходников. Portable replay блокировал np.load
внеbundle; исходные данные нужны только verifierдля получения history/targets.

{table(latency)}

Это observedsingle-packet CPUlatency на данномMac, не гарантия real-time.
15fixedzero-inputprobes: максимальный bias {v['maximum_zero_bias_deg']:.6f}°,
точно как уlegacyв тихом режиме. Exactequilibrium не заявляется.

Прежние bundles/API/neuraldefault и APHYNITY не изменены. Новый API — отдельный
экспериментальный history-onlypredictor. Будущее движение support предположено,
nativeactuator xr/Jr→фактическое движение не идентифицирован; качество прогноза
не доказывает counterfactual MPC или hardwareclosed-loopperformance.
Дальнейшая решающая проверка требует новых авторских/стендовыхrawtrials и
заранее зафиксированного протокола до просмотра их результатов.
''');report=dict(document=str(doc),document_sha256=sha256(doc),source_sha256=sha256(Path(__file__)),verification_sha256=sha256(OUT/'verification.json'),selected=selected,validation_change_percent=s['validation_legacy_change_percent'],old_clean_mae_deg=float(pivot.loc['all','selected']),old_legacy_mae_deg=float(pivot.loc['all','legacy']),old_change_percent=float(change),old_scenario_wins=wins,hard_context_better_on_old_all_mae=hard_better,bundle_files=count,bundle_bytes=size,independent_confirmation=False,final_goal_achieved=False);(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));print(primary.to_string(index=False));print(stress.to_string(index=False))
if __name__=='__main__':main()
