"""Summarize every frozen candidate and all old-record losses without reselection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cranepilot.data import sha256
from scripts.run_dated_forecast import RUN,OUT
from scripts.run_external_benchmark import write,verify_hashes
from scripts.report_drive_conditioned import balanced_cases
from scripts.report_angular_control import table

def main():
    if (OUT/'report.json').exists():raise FileExistsError('Retain previous report')
    p=json.loads((RUN/'development.json').read_text());s=json.loads((RUN/'selection.json').read_text());q=json.loads((RUN/'retrospective/provenance.json').read_text());v=json.loads((OUT/'verification.json').read_text());dv=json.loads((OUT/'development_verification.json').read_text());assert v['passed'] and dv['passed'];verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);verify_hashes(q['result_hashes'],RUN/'retrospective');m=pd.read_csv(RUN/'retrospective/metrics.csv');valid=m[m.windows>0];axes=valid.groupby(['age_ms','noise_quantum','group','case','scenario','method'],as_index=False).agg(mae_deg=('mae_deg','mean'),windows=('windows','first'));records=axes.groupby(['age_ms','noise_quantum','group','scenario','method'],as_index=False).mae_deg.mean();summary=records.groupby(['age_ms','noise_quantum','group','method'],as_index=False).agg(mae_deg=('mae_deg','mean'),scenarios=('scenario','nunique'));axes.to_csv(OUT/'axis_record_summary.csv',index=False);records.to_csv(OUT/'record_summary.csv',index=False);summary.to_csv(OUT/'summary.csv',index=False);curve=summary[(summary['group']=='all')&(summary.noise_quantum==0)].pivot(index='age_ms',columns='method',values='mae_deg').sort_index();curve.to_csv(OUT/'clean_age_curve.csv');balanced=[]
    for (age,noise),frame in m.groupby(['age_ms','noise_quantum']):
        c,_=balanced_cases(frame)
        for method,value in c.groupby('method').mae_deg.mean().items():balanced.append(dict(age_ms=int(age),noise_quantum=float(noise),method=method,balanced_mae_deg=float(value)))
    balanced=pd.DataFrame(balanced);balanced.to_csv(OUT/'balanced_summary.csv',index=False);balanced_clean=balanced[balanced.noise_quantum==0].pivot(index='age_ms',columns='method',values='balanced_mae_deg').sort_index();val=pd.DataFrame([dict(method=k,balanced_mae_deg=value,selected=k==s['selected']) for k,value in s['validation_means'].items()]).sort_values('balanced_mae_deg');val.to_csv(OUT/'validation_summary.csv',index=False);refs=pd.DataFrame([dict(case=name,age_ms=int(age)*10,reference=c['key']) for name,case in s['cases'].items() for age,c in case['reference'].items()]);refs.to_csv(OUT/'reference_choices.csv',index=False);pairs=records[records['group']=='all'].pivot(index=['age_ms','noise_quantum','scenario'],columns='method',values='mae_deg').reset_index();pairs['delta_parent_deg']=pairs.selected-pairs.parent_quad;pairs['delta_arx_deg']=pairs.selected-pairs.matched_arx;pairs['delta_bare_modal_deg']=pairs.selected-pairs.bare_dated_modal;pairs.to_csv(OUT/'paired_scenario_changes.csv',index=False);fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for j,noise in enumerate([0,1]):
        frame=summary[(summary['group']=='all')&(summary.noise_quantum==noise)].pivot(index='age_ms',columns='method',values='mae_deg').sort_index()
        for method,label in [('parent_quad','Previous quadratic adapter'),('selected','Frozen VAL selection'),('bank_simplex_1','Direct bank alone (simplex 1)'),('bare_dated_modal','Bare dated modal (4s)'),('direct_variable_length','Variable-length propagation (2.5s)'),('matched_arx','Age-conditioned ARX pool')]:axs[j].plot(frame.index,frame[method],marker='o',markersize=4,label=label)
        axs[j].set_xlabel('Known acquisition age of angle (ms)');axs[j].set_ylabel('MAE (degrees), equal old scenarios/axes/horizons');axs[j].set_title('Clean history' if noise==0 else 'Synthetic history jitter ±encoder quantum');axs[j].grid(alpha=.2)
    axs[0].legend(frameon=False,fontsize=8);fig.suptitle('Retrospective sensitivity; 30/70/250ms absent from new TRAIN/VAL');fig.savefig(OUT/'dated_forecast.png',dpi=170);plt.close(fig);headline=curve[['parent_quad','selected','bank_simplex_1','bare_dated_modal','direct_variable_length','matched_arx','augmented_arx']].reset_index();timing=pd.DataFrame(v['cpu_timing']);timing.to_csv(OUT/'cpu_timing.csv',index=False);zero=pd.DataFrame(v['zero_probes']);zero.to_csv(OUT/'zero_probes.csv',index=False);oldrank=curve.drop(columns='selected').mean().sort_values().rename('old_clean_mean_mae_deg').reset_index();oldrank.to_csv(OUT/'all_candidate_old_ranking.csv',index=False);cfg=p['config'];g=cfg['development_gate'];chosen=s['selected'];means=s['validation_means'];failed=[]
    if means[chosen]>g['maximum_parent_ratio']*means['parent_quad']:failed.append('меньше 3% выигрыша у прежнего адаптера')
    if means[chosen]>g['maximum_arx_ratio']*means['matched_arx']:failed.append('меньше 5% выигрыша у условного ARX')
    if s['arx_case_wins']<g['minimum_arx_case_wins']:failed.append('меньше 4/5 выигранных случаев у ARX')
    if any(value[chosen]>g['maximum_each_age_parent_ratio']*value['parent_quad'] for value in s['per_age'].values()):failed.append('ухудшение более 5% относительно прежнего адаптера хотя бы при одном VAL age')
    lines=[]
    for age in [50,100,150,200,250]:
        row=curve.loc[age];loss=pairs[(pairs.age_ms==age)&(pairs.noise_quantum==0)&(pairs.delta_parent_deg>0)].scenario.astype(int).tolist();lines.append(f"- {age} мс: {row.selected:.6f}° против прежних {row.parent_quad:.6f}° ({100*(row.selected/row.parent_quad-1):+.2f}%); проигранные прежнему методу сценарии: {loss}.")
    bundle=Path('artifacts/dated_bundle_20261108');doc=Path('docs/DATED_FORECAST_2026_11_08_RU.md');doc.write_text(f'''# Прямой прогноз от датированного измерения

Выбран по validation **{chosen}**. Критерий допуска: **{s['development_gate_passed']}**;
выигрыш у условного ARX в {s['arx_case_wins']}/5 случаях. Невыполненные условия:
{'; '.join(failed) if failed else 'нет'}.

## Что проверено

Протокол и численный код зафиксированы до расчётов: `{p['git_head']}`.
Измеренный угол заканчивается на t−age; история длины, положения и исходных
референсов — на t. Известный целый age от 0 до 250 мс, 100 Гц, 401 отсчёт
на канал. Первые прогнозы в 4.25 с, шаг 1 с: везде настоящая предыстория,
без дополнения края. Цели через 0.10/0.20/0.35/0.50/0.75/1.00/1.50 с после t.

21 вариант: прямые модальные модели, перенос через уже измеренное движение
привода, удержание или линейное продолжение будущего положения, модель
переменной длины, причинная смесь шести экспертов и фиксированные смеси с
прежним пилотом. Будущее положение/длина — явные допущения, не измеренные
входы. Переменная длина меняет жёсткость и добавляет 2 Ldot/L к затуханию;
это локальная малая модель, не доказательство динамики реального 3D стенда.
Обоснование и первичные источники в DATED_FORECAST_DESIGN_20261108.md.

Весам доступны шесть прогнозных ошибок, полученных из пакета на одну секунду
раньше. Все шесть целей уже получены к t, последняя — последний доступный угол.
Спрогнозированный текущий угол не становится эталоном. Тот старый пакет
использует собственное продолжение привода, а не реально увиденную позже
траекторию. При age=0 все кандидаты точно сохраняют прежний прогноз;
незащищённый bare_dated_modal приведён отдельно.

## Соперник и выбор

120 новых ARX обучены на тех же TRAIN записях, новых age 0/20/50/100/150/200 мс
и двух уровнях шума: чисто или uniform ±encoder quantum. 2 набора признаков ×
2 формы × 6 ridge × 5 случаев. Масштабирование только по TRAIN. Кроме них,
в пуле все 120 прежних age-augmented ARX и оба сохранённых ARX под всеми 8
адаптерами. Для каждого из пяти случаев и шести известных TRAIN age выбрано
лучшее по VAL с усреднением двух шумов — 30 решений. На прочих age ближайший
TRAIN age; при равенстве меньший. Все методы получают одинаковые доступные
истории и age. Плотность исходных TRAIN окон у старого банка и новых ARX
различается; сравниваются полные pipelines, не изолированная архитектура.

Общий кандидат выбран по равному весу пяти случаев, age и шума, с балансом
quiet/drive. Полная таблица, включая отрицательные варианты:

{table(val)}

Все веса/выборы зафиксированы до нового replay. Ничего не перевыбрано по его
ошибкам. development_gate_passed — внутренний критерий, не независимое
подтверждение и не гарантия для реального контроллера.

## Уже открытые записи

Все 33 исходные записи использовались раньше; новые записи от авторов ещё
не получены. Это ретроспективное испытание синтетической задержки/шума.
30/70/250 мс не участвовали в новой сетке TRAIN/VAL. Угол задержан искусственно,
записанные команды/положение остаются прежними, замкнутый контур не пересчитан.
Шум добавлен только к полученной истории; цель — прежний encoder, а не
независимый высокоточный физический эталон. Непрерывный сетевой процесс,
потери пакетов и неизвестная меняющаяся задержка здесь не проверены.

{table(headline)}

Здесь равные веса семи сценариев, осей внутри сценария и семи горизонтов;
это иная агрегация, чем balanced VAL score. Ни одно окно не исключалось по
ошибке или амплитуде. Перекрывающиеся окна не являются независимыми trials.

{chr(10).join(lines)}

При 200 и 250 мс один прямой банк `bank_simplex_1` точнее выбранной смеси:
{curve.loc[200,'bank_simplex_1']:.6f}° и {curve.loc[250,'bank_simplex_1']:.6f}°
соответственно. Выбор не менялся по этим числам. Член переменной длины меняет
общую чистую MAE относительно свободной модели 2.5 с максимум на
{np.abs(curve.direct_variable_length-curve.direct_free2p5).max():.6f}°; это не
основание приписывать ему основной выигрыш смеси.

Для сопоставления с критерием выбора ниже также приведена balanced MAE:
равные пять случаев, записи внутри случая и quiet/drive при их наличии.

{table(balanced_clean[['parent_quad','selected','bank_simplex_1','matched_arx']].reset_index())}

Все индивидуальные проигрыши, режимы, шумы и кандидаты сохранены в CSV/NPZ.
График включает простой dated modal и переменную длину даже при их выигрыше
над выбранным методом. Нельзя выдавать лучшего после replay за заранее
выбранную модель или сравнивать эти MAE с другими статьями без общего теста.

## Воспроизводимость и отдельный API

В трёх VAL сценариях длина постоянна: ветка переменной длины там точно
совпадает со свободной моделью 2.5 с. Поэтому VAL не подтверждает отдельный
выигрыш от члена 2 Ldot/L. Покрытие изменения длины и различающиеся прогнозы
на старых записях посчитаны отдельно в length_coverage.csv/json после freeze,
без изменения выбора.

243 существующих и новых unit tests прошли до обучения. Переменная длина
проверена независимым solve_ivp, постоянная длина — аналитическим решением.
Повторно обучены все 120 новых ARX, воспроизведены все оценки VAL, 30 решений
ARX, общий выбор, gate и все метрики. Полный raw/API replay:
{v['forecast_comparisons']} сравнений, максимальная разница
{v['maximum_forecast_error_rad']:.3g} рад. {len(v['clock_poison_probes'])} проверок
с заменой неполученных углов/будущих каналов не изменили входы и прогноз.
Причинность программы не подтверждает фактическую доступность каналов на стенде.
1281 исходный main/APHYNITY файл проверен по хешам; прежние профили сохранены.

Отдельный `artifacts/dated_bundle_20261108`: {sum(f.is_file() for f in bundle.rglob('*'))}
файлов, {sum(f.stat().st_size for f in bundle.rglob('*') if f.is_file())} байт.
`FrozenDatedPilot.forecast(..., angle_age_samples=..., include_reference=True)`.
401 отсчёт newest-first; age обязателен из реальных метаданных, не оценён по
будущей ошибке. Проверка API запрещала np.load вне этого bundle: данные и
ранние runs для inference не нужны, нужен репозиторий и его зависимости.
Новый профиль экспериментальный; прежний default не заменён. Отдельно
исправлена воспроизводимость старых тестов: точный сохранённый JSON параметров
добавлен в Git без изменения его содержимого. На чистом git archive прошли
243 теста, затем прошёл Linux GitHub CI (run 42490990869, commit d93121d).
Подробности в CI_FIXTURE_20261108.md; это исправление комплектации, не обучение.

{table(timing)}

CPU — измерение на данном Mac, три повтора на случай; весь банк считается
и при выдаче только выбранного метода. Гарантии real time нет. 20 zero-input
проб при L=.4 м: max bias {zero.maximum_bias_deg.max():.6f}°, не exact zero.
Для независимого вывода нужны новые сырые записи, описание единиц/фильтров,
времена acquisition/arrival и заранее замороженный протокол проверки.
''');write(OUT/'report.json',dict(selected=chosen,development_gate_passed=s['development_gate_passed'],gate_failures=failed,validation_means=means,old_clean_curve=curve.to_dict(),verification_sha256=sha256(OUT/'verification.json'),development_verification_sha256=sha256(OUT/'development_verification.json'),document_sha256=sha256(doc),independent_confirmation=False,final_goal_achieved=False));print(headline.to_string(index=False));print('SELECTED',chosen,'GATE',s['development_gate_passed'])

if __name__=='__main__':main()
