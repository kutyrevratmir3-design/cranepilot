"""Report the stronger reference without changing the frozen candidate."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cranepilot.data import sha256
from scripts.dense_reference import RUN,OUT,setup
from scripts.report_drive_conditioned import balanced_cases
from scripts.report_angular_control import table
from scripts.run_external_benchmark import write,verify_hashes

def main():
    if (OUT/'report.json').exists():raise FileExistsError('Preserve previous report')
    cfg,grids,primary,task=setup();p=json.loads((RUN/'development.json').read_text());q=json.loads((RUN/'retrospective/provenance.json').read_text());v=json.loads((OUT/'verification.json').read_text());dv=json.loads((OUT/'development_verification.json').read_text());assert v['passed'] and dv['passed'];verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);verify_hashes(q['result_hashes'],RUN/'retrospective');s=json.loads((RUN/'selection.json').read_text());m=pd.read_csv(RUN/'retrospective/metrics.csv');valid=m[m.windows>0];axis=valid.groupby(['age_ms','noise_quantum','group','case','scenario','method'],as_index=False).mae_deg.mean();records=axis.groupby(['age_ms','noise_quantum','group','scenario','method'],as_index=False).mae_deg.mean();summary=records.groupby(['age_ms','noise_quantum','group','method'],as_index=False).mae_deg.mean();axis.to_csv(OUT/'axis_summary.csv',index=False);records.to_csv(OUT/'record_summary.csv',index=False);summary.to_csv(OUT/'summary.csv',index=False);curve=summary[(summary['group']=='all')&(summary.noise_quantum==0)].pivot(index='age_ms',columns='method',values='mae_deg').sort_index();curve.to_csv(OUT/'clean_age_curve.csv');balanced=[]
    for role,frame in [('validation',pd.read_csv(RUN/'validation_metrics.csv')),('old_evaluation',m)]:
        for (age,noise),part in frame.groupby(['age_ms','noise_quantum']):
            c,_=balanced_cases(part)
            for method,value in c.groupby('method').mae_deg.mean().items():balanced.append(dict(role=role,age_ms=int(age),noise_quantum=float(noise),method=method,mae_deg=float(value)))
    b=pd.DataFrame(balanced);b.to_csv(OUT/'balanced_summary.csv',index=False);val=b[b.role=='validation'].groupby('method').mae_deg.mean();old=b[(b.role=='old_evaluation')&(b.noise_quantum==0)].pivot(index='age_ms',columns='method',values='mae_deg').sort_index();choices=pd.DataFrame([dict(case=name,age_ms=int(age)*10,key=c['key'],validation_mae_deg=c['validation_scores'][c['key']]) for name,case in s['cases'].items() for age,c in case['reference'].items()]);choices.to_csv(OUT/'reference_choices.csv',index=False);paired=records[records['group']=='all'].pivot(index=['age_ms','noise_quantum','scenario'],columns='method',values='mae_deg').reset_index();paired['selected_minus_dense_arx_deg']=paired.selected-paired.dense_matched_arx;paired.to_csv(OUT/'paired_old_scenarios.csv',index=False);fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,noise in zip(axes,[0,1]):
        f=summary[(summary['group']=='all')&(summary.noise_quantum==noise)].pivot(index='age_ms',columns='method',values='mae_deg').sort_index()
        for method,label in [('parent_quad','Previous quadratic adapter'),('selected','Unchanged primary mixture'),('bank_simplex_1','Direct bank alone'),('matched_arx','Previous conditional ARX'),('dense_matched_arx','Full dense reference pool')]:ax.plot(f.index,f[method],marker='o',markersize=4,label=label)
        ax.set_xlabel('Known angle acquisition age (ms)');ax.set_ylabel('MAE (degrees), equal old scenarios/axes/horizons');ax.set_title('Clean history' if noise==0 else 'Synthetic jitter ±encoder quantum');ax.grid(alpha=.2)
    axes[0].legend(frameon=False,fontsize=8);fig.suptitle('Reference challenge only: primary frozen before expanding the ARX pool');fig.savefig(OUT/'dense_reference.png',dpi=170);plt.close(fig);support=json.loads((RUN/'train_support.json').read_text());support_frame=pd.DataFrame(support['records']);support_summary=support_frame.groupby('case',as_index=False).agg(records=('scenario','nunique'),original_windows=('original_windows','sum'),new_dense_base_windows=('augmented_base_windows','sum'));support_summary.to_csv(OUT/'train_support_summary.csv',index=False);bundle=Path('artifacts/dense_reference_bundle_20261108');doc=Path('docs/DENSE_REFERENCE_2026_11_08_RU.md');headline=curve[['selected','matched_arx','dense_matched_arx','parent_quad','bank_simplex_1']].reset_index();doc.write_text(f'''# Полный плотный ARX-контроль без перевыбора CranePilot

Протокол/код до fitting: `{p['git_head']}`. Выбранный ранее `mix_bank_10`
не менялся. Все старые прогнозы и отрицательные результаты сохранены.
Все 33 записи уже были открыты: это исправление и усиление сравнения,
**не независимая проверка**. Новых подходящих физических логов не получено.

## Какой пробел устранён

В прежнем сравнении был доступ к тем же TRAIN-записям, но не к полному
плотному ARX-пулу. Для jib theta оба названных эксперта падали к старому
quiet-trained ARX; для trolley использовалась выбранная короткая модель.
Не все плотные 4-секундные модели предлагались как самостоятельные соперники.
Поэтому прежние числа нельзя называть доказательством равной поддержки всех
supervised TRAIN-пакетов во всех случаях. Старые отчёты оставлены историческими.

Теперь предлагаются все 24 исходных плотных 4-секундных ARX на случай под
каждым из 8 адаптеров, сохранённые 3-секундные/legacy модели, все прежние
age-augmented модели и 120 новых плотных ARX. Всего 304 варианта на случай;
часть численно совпадает. Никакой тестовый score не выбирал соперника.

{table(support_summary)}

Первичный физический prior и исходные плотные ARX обучались на одном `tr`
со стартом 4 с и шагом 25 raw-отсчётов. Новая augmentation использует
4.25 с/шаг25 для настоящей предыстории при задержке до 250 мс: её origins
равны исходным без первого. Исходные fits, включая окно в 4 с, остаются в пуле.
Роли записей, каналы, исходные истории и целевые timestamps проверены ledger.
Доступ к сырым TRAIN-данным и плотная поддержка сопоставлены; формы признаков,
окна 3/4 с, физические objectives и augmentation различаются. Это не заявление,
что каждый learner получил одинаковые преобразованные примеры.

## Замороженный соперник

120 новых fits: angle/io LPV × direct/dated-modal-residual × 6 ridge × 5 cases.
Те же TRAIN ages 0/20/50/100/150/200 мс и шум 0/±q. VAL/replay остались на
прежних origins 4.25 с/шаг1с, без изменения популяции оценки. Для каждого
case/knownage лучшая из 304 моделей выбрана по среднему двух VAL-шумов —
30 решений. Для нового age ближайший TRAIN age, при равенстве меньший.

VAL balanced MAE, равные случаи/age/шум: primary {val['selected']:.6f}°,
прежний ARX {val['previous_matched_arx']:.6f}°, новый {val['dense_matched_arx']:.6f}°.
Снижение primary относительно нового ARX:
{100*(1-val['selected']/val['dense_matched_arx']):.2f}%. Это внутренний VAL-result.
Сам primary и его старый gate заново не выбирались и не переписывались.

{table(choices)}

## Старый retrospective replay

{table(headline)}

Таблица: равные 7 сценариев, оси внутри сценария и 7 горизонтов. Поддержка
при одинаковых age/noise/origins проверена; это не новый физический набор.
Будущие измеренные команды/положение не поступали в прогноз. Synthetic lag
сохраняет записанное управление и не пересчитывает замкнутый контур.

Balanced-представление, равный вес пяти cases и quiet/drive. Здесь quiet
означает отсутствие прошлого движения опоры и включает length_active;
изменение длины отдельно не балансируется. У jib22/23 drive содержит всего
2 из 94 окон, у simult33 — 3, но получает половину веса. Поэтому эта оценка
не заменяет таблицу с равными физическими сценариями выше.

{table(old[['selected','matched_arx','dense_matched_arx']].reset_index())}

Расширенный пул улучшил VAL, но его VAL-выбор хуже прежнего ARX в общем
старом среднем для age 50–250 мс. Это усиление полноты сравнения, а не
универсальное улучшение точности ARX. Оба соперника остаются в таблицах;
они не заменялись минимумом по test score. Primary выигрывает у нового ARX
7/7 старых сценариев до 200 мс, но лишь 5/7 при 250 мс: проиграны 9 и 21.
Прямой банк при больших age местами точнее выбранной смеси; эта потеря тоже
сохранена. Все 26 методов, режимы, шумы и проигранные сценарии доступны в CSV.

## Воспроизводимость и применение

297 unit tests прошли перед этим replay. Повторены все 120 новых TRAIN fits,
все 18240 VAL option scores и все 30 выборов. Точная поддержка плотных TRAIN
origins/targets подтверждена. Source/raw/portable replay: {v['forecast_vectors_checked']}
векторов, max error {v['maximum_error_rad']:.3g} рад. Исходные main/APHYNITY
файлы и прежние bundles не изменялись. Полный отчёт проверки содержит
покрытие/clock probes и хеши; сам зелёный тест не доказывает независимость.

Отдельный comparator-only bundle: {sum(f.is_file() for f in bundle.rglob('*'))} файлов,
{sum(f.stat().st_size for f in bundle.rglob('*') if f.is_file())} байт. Он требует
уже сохранённый `dated_bundle_20261108` и основной исходный код, но не dataset
или старые run-каталоги для inference. API:

```python
from cranepilot.matched_dated_pilot import FrozenDenseReference
reference = FrozenDenseReference()
result = reference.forecast(history, family="trolley", axis="theta", angle_age_samples=0)
```

Параметр age обязателен. Контракт историй/единиц/горизонтов тот же, что у
замороженного primary. Нативные xr/Jr не переименовываются в силу, ускорение
или напряжение без доказательства. В новом независимом тесте будут зафиксированы
оба bundle, все 30 решений, TRAIN evidence и metadata routing до новых ошибок.

Prospective intake/scoring: `FRESH_CONFIRMATION_20261108.md`. Реестр ранее
виденных сигналов и проверка metadata не создают новых trials. Основная цель
остаётся открытой до настоящих новых подходящих записей и полного аудита.
''');write(OUT/'report.json',dict(validation_means=val.to_dict(),clean_old_curve=curve.to_dict(),primary_unchanged=True,primary_manifest_sha256=p['primary_manifest_sha256'],verification_sha256=sha256(OUT/'verification.json'),document_sha256=sha256(doc),independent_confirmation=False,final_goal_achieved=False));print(headline.to_string(index=False))

if __name__=='__main__':main()
