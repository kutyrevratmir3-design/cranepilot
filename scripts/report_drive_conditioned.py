"""Report registered selection and late attribution controls without reselection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cranepilot.data import sha256
from scripts.run_drive_conditioned import RUN,OUT
from scripts.report_angular_control import table
from scripts.run_external_benchmark import verify_hashes

def balanced_cases(frame):
    # Horizons equal; quiet/active binary groups pool their constituent regimes
    # by observed window counts, then receive equal weight within each record.
    regimes={'low_past_motion':0,'length_active':0,'drive_active':1,'drive_and_length_active':1};part=frame[frame['group'].isin(regimes)].copy();part['drive']=part['group'].map(regimes);part=part[part.windows>0];part['weighted_error']=part.mae_deg*part.windows;binary=part.groupby(['case','scenario','method','drive'],as_index=False)[['weighted_error','windows']].sum();binary['mae_deg']=binary.weighted_error/binary.windows;records=binary.groupby(['case','scenario','method'],as_index=False).mae_deg.mean();return records.groupby(['case','method'],as_index=False).mae_deg.mean(),records

def main():
    if (OUT/'report.json').exists():raise FileExistsError('Retain earlier report')
    dev=json.loads((RUN/'development.json').read_text());ret=json.loads((RUN/'retrospective/provenance.json').read_text());attrib=json.loads((RUN/'attribution/provenance.json').read_text());v=json.loads((OUT/'verification.json').read_text());selection=json.loads((RUN/'selection.json').read_text());assert v['passed'];verify_hashes(dev['guarded_hashes']);verify_hashes(dev['result_hashes'],RUN);verify_hashes(ret['result_hashes'],RUN/'retrospective');verify_hashes(attrib['source_hashes']);verify_hashes(attrib['result_hashes'],RUN/'attribution');assert not attrib['used_for_selection'];frames=[pd.read_csv(RUN/'validation_metrics.csv'),pd.read_csv(RUN/'retrospective/metrics.csv'),pd.read_csv(RUN/'attribution/metrics.csv')];metrics=pd.concat(frames,ignore_index=True);metrics.to_csv(OUT/'all_metrics.csv',index=False);valid=metrics[metrics.windows>0];axes=valid.groupby(['role','group','case','scenario','method'],as_index=False).agg(mae_deg=('mae_deg','mean'),windows=('windows','first'));records=axes.groupby(['role','group','scenario','method'],as_index=False).mae_deg.mean();summary=records.groupby(['role','group','method'],as_index=False).agg(mae_deg=('mae_deg','mean'),scenarios=('scenario','nunique'));axes.to_csv(OUT/'axis_record_summary.csv',index=False);records.to_csv(OUT/'record_summary.csv',index=False);summary.to_csv(OUT/'regime_summary.csv',index=False)
    old=metrics[metrics.role=='old_evaluation'];cases,balanced_records=balanced_cases(old);cases.to_csv(OUT/'balanced_case_summary.csv',index=False);balanced_records.to_csv(OUT/'balanced_axis_record_summary.csv',index=False);balanced=cases.groupby('method').mae_deg.mean();balanced.to_csv(OUT/'balanced_summary.csv');selected=selection['selected'];ratio=float(balanced['selected']/balanced['strong_arx']);p=cases.pivot(index='case',columns='method',values='mae_deg');old_wins=int((p['selected']<p.strong_arx).sum());validation=pd.DataFrame([dict(method=key,balanced_mae_deg=value,quiet_mae_deg=selection['validation_quiet_means'][key],selected=key==selected) for key,value in selection['validation_means'].items()]);validation.to_csv(OUT/'validation_summary.csv',index=False);ref_rows=pd.DataFrame([dict(case=name,reference=c['reference'],model=c['long_model'],**c['reference_validation_scores']) for name,c in selection['cases'].items()]);ref_rows.to_csv(OUT/'reference_choices.csv',index=False)
    keys=['strong_arx','old_blend','selected','matched_free_blend','matched_local_zero_blend'];primary=summary[(summary.role=='old_evaluation')&summary.method.isin(keys)];primary.to_csv(OUT/'primary_regime_comparison.csv',index=False);regimes=dev['qualification']['regime_names'];plot=primary[primary['group'].isin(regimes)].pivot(index='group',columns='method',values='mae_deg').reindex(regimes);fig,ax=plt.subplots(figsize=(10,5),layout='constrained');x=np.arange(4);width=.18
    for j,(method,label,color) in enumerate([('strong_arx','Matched ARX','#3f7183'),('old_blend','Old blend','#b8a18b'),('selected','VAL-selected method','#bb8247'),('matched_free_blend','Late free-blend control','#648b6d')]):ax.bar(x+(j-1.5)*width,plot[method],width,label=label,color=color)
    ax.set_xticks(x,['Low past motion','Drive active','Length active','Drive + length']);ax.set_ylabel('MAE (degrees), equal scenarios/axes/horizons');ax.set_title('Previously opened records: retrospective development');ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.savefig(OUT/'drive_conditioned.png',dpi=170);plt.close(fig)
    coefficients=[]
    for name in selection['cases']:
        d=json.loads((RUN/name/'prior_diagnostics.json').read_text());coefficients.append(dict(case=name,max_absolute_basis_gain=max(abs(x) for x in d['physical_basis_gain']),condition_median=d['online_system_condition_median'],condition_max=d['online_system_condition_max']))
    diagnostics=pd.DataFrame(coefficients);diagnostics.to_csv(OUT/'conditioning_summary.csv',index=False);doc=Path('docs/DRIVE_CONDITIONED_2026_11_07_RU.md');better_control=balanced[['matched_free_blend','matched_local_zero_blend']].min()<=balanced['selected'];attribution_text='Простой контроль без forced-поправки не хуже выбранного метода: отдельный выигрыш от input-aware physics не подтверждён.' if better_control else 'Выбранный метод лучше обоих контролей без forced-поправки на этом ретроспективном balanced-score; независимую пользу ещё предстоит проверить.'
    doc.write_text(f'''# Движение привода: физическая поправка и усиленный ARX

Протокол и численные исходники до fitting: `{dev['git_head']}`.
Все33старые записи уже открыты; это ретроспективная разработка.
Будущие измеренные положение/длина/ускорение не передавались в прогноз.

## Что сделано

Физический локальный oscillator учитывает измеренную прошлую x/J через
точно интегрированный forced-response basis. Он оценивает текущее состояние
колебаний с TRAIN-приором coupling; два варианта онлайн(fixed/adapt) и два
предположения о будущем support(hold/ограниченная линейная экстраполяция).
Всего5обученных priors и4physical/4blend варианта. Это частичная локальная
модель, не полный MHE/MPC и не идентифицированный native-actuator.

Новая сильная база:120ARXgridfits на всех post4sTRAINокнах с теми же доступными
сигналами, плюс5short3sfits. Per-case VAL выбирает IO/angle-LPV,
direct/modal-residual, ridge, затем3s/4s/старыйARXfallback. Улучшение базы
само по себе не доказывает пользу физической поправки.

{table(ref_rows)}

## Зафиксированный выбор по validation

Один общий метод из11: **{selected}**. BalancedMAE меняется на
{selection['validation_change_percent']:.2f}% против matchedARX;
выигрыш {selection['validation_case_wins']}/5cases.
Заранее заданный developmentgate(≥10%gain,≥4/5wins,quiet≤1.05reference):
**{selection['development_gate_passed']}**. Никакого перевыбора по oldevaluation.

{table(validation)}

Balanced-score: равный вес пятиcases; внутриcase тихие и drive-active origins
имеют равный вес. Все7горизонтов. В fitting каждыйTRAINrecord имеет равный
вес и равные присутствующие driveгруппы. VALrecords4/17/28 прежние; новые
hoist-VALусловия не выдуманы. Длина меняется в частиTRAIN/oldevaluation.

## Ретроспективный replay прежних7сценариев

BalancedMAE selected={balanced['selected']:.6f}°, matchedARX={balanced['strong_arx']:.6f}°,
изменение={100*(ratio-1):.2f}%; casewins={old_wins}/5.
Здесь balanced-среднее сначала по presentquiet/drive внутриaxis/record,
потом поровнуrecords внутриcase и пятьcases. Это не та же агрегация, что
следующая таблица: в ней равный вес сценариев/осей/горизонтов отдельно по
regime. Две таблицы не сравнивать как одинаковый MAE.

{table(pd.DataFrame([dict(method=k,balanced_mae_deg=float(balanced[k])) for k in keys]))}

{table(primary)}

Все consideredorigins≥4s сохранены, stride.25s, история4s, горизонты.10–1.50s.
Regimes только из прошлых401отсчётов, без target-based отбора. Пересекающиеся
окна не являются независимыми испытаниями. old_numerical_scope с future
recordedreference — только описательная историческая метка, не вход прогноза.
Малые числа сценариев в hoistгруппах не доказывают общую hoist-robustness.

## Проверка вклада физики

После первого trolleyVALoutput дополнительно зафиксированы два диагностических
контроля с тем же новымARX и тем же past-only softgate: free-modal blend и
локальный modal-refit blend с нулевыми forcedgains. Они не участвовали в выборе
11методов и developmentgate. Это поздняя ablation, её статус явно сохранён.
{attribution_text}

Нельзя приписывать всю разницу с old_blend физике: ARXпереобучен на активных
TRAINокнах и заново выбран поVAL. Сохранены все физические варианты, все125
ARXвесов/оценок, controls и проигрыши.

## Численная проверка и границы

218testsPASS доfitting, включая независимую matrix-exponential проверку
first-order-hold. Повторены все125TRAINARXfits и5physicalpriors; selection,
прогнозы и всеregime/headmetrics пересчитаны. {v['forecast_comparisons']}
forecastcomparisons. Все1281оригинальные main/APHYNITYфайла целы.

{table(diagnostics)}

Диагностика condition относится к регуляризованной малой системе онлайнfit,
не является доказательством физической идентифицируемости. Веса basis могут
поглощать ошибки геометрии и датчиков. Изменение Lза горизонт и native
xr/Jr→реализованное движение здесь не идентифицированы. Гипотетический
support-path interface проверен синтетически, не заменяет actuator дляMPC.
Новый метод не менял прежние frozenbundles, publicAPIgate или neuraldefault.

Результат расширяет проверку до активных участков, но исходную независимую
цель не закрывает. Нужны новые сырые записи авторов/стенда с единицами,
временем, командами и минимум4sпредыстории. Публикационные/SOTA/аппаратные
выводы до такого теста преждевременны.
''');report=dict(document=str(doc),document_sha256=sha256(doc),source_sha256=sha256(Path(__file__)),verification_sha256=sha256(OUT/'verification.json'),selected=selected,validation_gate=selection['development_gate_passed'],retrospective_balanced_selected_mae_deg=float(balanced['selected']),retrospective_balanced_arx_mae_deg=float(balanced['strong_arx']),retrospective_change_percent=100*(ratio-1),retrospective_case_wins=old_wins,late_unforced_control_not_worse=bool(better_control),independent_confirmation=False,final_goal_achieved=False);(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));print(primary.to_string(index=False))

if __name__=='__main__':main()
