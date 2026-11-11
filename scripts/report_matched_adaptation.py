"""All same-data outcomes and explicit lack of independent confirmation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cranepilot.data import sha256
from scripts.report_angular_control import table

OUT=Path('results/matched_adaptation_20261107');RUN=Path('artifacts/runs/matched_adaptation_20261107')

def main():
    out=OUT;dest=out/'report.json'
    if dest.exists():raise FileExistsError('Retain earlier report')
    p=json.loads((RUN/'provenance.json').read_text());v=json.loads((out/'verification.json').read_text());cfg=p['config'];s=json.loads((RUN/'selection.json').read_text());assert v['passed'] and v['provenance_sha256']==sha256(RUN/'provenance.json')
    from scripts.run_external_benchmark import verify_hashes
    verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN)
    metrics=pd.concat([pd.read_csv(RUN/g/'metrics.csv') for g in ['validation','retrospective']],ignore_index=True);metrics.to_csv(out/'all_metrics.csv',index=False)
    records=metrics.groupby(['group','record','method'],as_index=False)[['mae_deg','rmse_deg']].mean();records.to_csv(out/'record_summary.csv',index=False);summary=records.groupby(['group','method'],as_index=False)[['mae_deg','rmse_deg']].mean();summary.to_csv(out/'summary.csv',index=False)
    comparisons=[]
    for group in ['validation','retrospective']:
        m=records[records.group==group].pivot(index='record',columns='method',values='mae_deg');r=m.strong_arx.to_numpy();rng=np.random.default_rng(7112028);ids=rng.integers(0,4,(2000,4))
        for method in cfg['methods']:
            a=m[method].to_numpy();effect=100*(a[ids].mean(1)/r[ids].mean(1)-1)
            comparisons.append(dict(group=group,method=method,change_percent=100*(a.mean()/r.mean()-1),record_wins=int((a<r).sum()),records=4,descriptive_low=np.quantile(effect,.025),descriptive_high=np.quantile(effect,.975)))
    comparison=pd.DataFrame(comparisons);comparison.to_csv(out/'paired_comparisons.csv',index=False)
    chosen=s['candidate'];val=s['validation_scores'];gain=100*(val[chosen]/val['strong_arx']-1);gate=bool(gain<=-10 and comparison[(comparison.group=='validation')&(comparison.method==chosen)].record_wins.iloc[0]>=3)
    (out/'acceptance.json').write_text(json.dumps(dict(development_gate_passed=gate,criterion='validation MAE <=.90 strongARX and >=3/4 record wins',validation_selected=chosen,validation_change_percent=gain,independent_confirmation=False,final_goal_achieved=False),indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(13,5.4),layout='constrained')
    for ax,group in zip(axes,['validation','retrospective']):
        frame=summary[summary.group==group].set_index('method').loc[cfg['methods']];ax.bar(cfg['methods'],frame.mae_deg,color=['#3f7b87','#648d65','#bc8b48','#706897','#ab6577']);ax.set_title(group);ax.set_ylabel('MAE (deg), equal records/horizons');ax.tick_params(axis='x',rotation=40,labelsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Equal training access: development and retrospective scores, no fresh confirmation');fig.savefig(out/'matched_comparison.png',dpi=170);plt.close(fig)
    doc=Path('docs/MATCHED_ADAPTATION_2026_11_07_RU.md');fits=pd.read_csv(RUN/'ridge_validation.csv');neural=pd.DataFrame(s['neural'])
    text=f'''# Равный доступ к обучению: разработка против усиленного ARX

Протокол и код до fitting: `{p['git_head']}`. TRAIN1–12, VALIDATION13–16
внешнего rotary-camera набора. Все модели получили одинаковые обучающие records
и доступные causal history/known-command signals. TEST17–20 уже использовался:
новая оценка на нём **ретроспективная**, НЕ независимое подтверждение.

## Итог

Выбран только по validation: **{chosen}**, изменение MAE против сильного ARX
**{gain:+.3f}%**. Development gate(≥10% reduction и≥3/4recordwins): **{gate}**.
**Итоговая цель ещё не достигнута: свежего подтверждающего набора нет.**
Ни один опубликованный default/bundle или APHYNITY не заменён.

{table(summary)}

Парные comparisons по целым records, не overlapping windows:

{table(comparison)}

Bootstrap4recordclusters,2000draws,seed7112028,общие resamples для методов;
описательные интервалы не устраняют reuse и малый размер данных.

## Что именно проверено

StrongARX выбирает direct или geometric residual и ridge среди6значений
1e−10..1; все12fit сохранены. Поэтому улучшение не обеспечивается намеренно
узким comparator grid. PhysicspriorARX использует ту же линейную feature class
поверх damped oscillator, fitted на тех же12TRAINrecords в предыдущем этапе.
StructuredNARX добавляет5групп ограниченных нелинейных history/command terms,
без новых measured signals и arbitrary polynomial search. Future command features
обрываются на соответствующем horizon. Все24ridge outcomes:

{table(fits.drop(columns='file'))}

Прежняя retained neural architecture теперь дообучена на тех же12records;
3source seeds,100epochs max,AdamWlr.0003,record-balanced seven-horizon MSE.
Checkpoint только по validation, epoch−1 допустим. Equilibrium variant вычитает
отклик этой же сети на zero packet с той же omega. Это ограничение zero response,
не passivity/stability theorem. Все6neuralfits:

{table(neural)}

Изменены только новые model copies. Virtual reference=2*knownvelocity/.2,
metadata/source-clock frequency и forcing convention используют TRAIN-only
PEM omega/gain. X и prescribed acceleration масштабированы gain*g/(L*omega²).
Это явный адаптер угловой подмодели; native actuator bypass остаётся. Angular
history9native samples каузально удерживаются до41entries. Никакой будущей
измеренной траектории, test mean или future timestamp в input нет.

## Данные авторов на том же стенде

По предложению пользователя проверены работы Kulash Talapiden и Ahmad Bala
Alhassan. [Talapiden et al.2026](https://journals.sagepub.com/doi/10.1177/10775463251411778)
прямо указывают выдачу данных по запросу corresponding author.
[Публичный набор](https://data.mendeley.com/datasets/x69hyyv3zw/2) — прежние33cases,
не свежий test. [Alhassan et al.2024](https://doi.org/10.1111/exsy.13484) также
указывают on-request data; принадлежность к тому же физическому экземпляру
не подтверждена. Для ISA2026 raw-file link не проверен. Полный source audit
и конкретный перечень сигналов: SAME_RIG_DATA_ACCESS_20261107.md.
Авторские письма/посты не отправлялись. Ссылки на результаты в PDF и статья
с теми же авторами сами по себе не предоставляют новые численные записи.

## Проверки и дальнейшая работа

184unit tests PASS до fitting. Все {v['forecasts']} forecasts повторно рассчитаны
из raw source+saved checkpoints, metrics независимо пересчитаны, validation
selection воспроизведён. Все {v['original_files']['checked_files']} original
main/APHYNITY files сохранены. Source/parent/checkpoint/resultSHA verified.
Прогнозы игнорируют poisoned targets/futuretimestamps. Изменений старых runs нет.

Новый метод оценивается как разработка. Даже хороший retrospective score не
закрывает задачу независимого преимущества. Для подтверждения нужны новые raw
recordings, замороженный метод/бейслайн и заранее заданные metrics/conditions.
Не понижать comparator или критерии ради формального успеха. Если кандидат
не выигрывает, продолжать с конкретной диагностикой и сохранить отрицательный
результат; не объявлять нейросеть обязательной частью пилота.
'''
    doc.write_text(text);report=dict(document=str(doc),document_sha256=sha256(doc),provenance_sha256=sha256(RUN/'provenance.json'),verification_sha256=sha256(out/'verification.json'),report_source_sha256=sha256(Path(__file__)),aggregation_helper_sha256=sha256(Path('scripts/report_angular_control.py')),independent_confirmation=False,final_goal_achieved=False,development_gate_passed=gate,selected=chosen)
    dest.write_text(json.dumps(report,indent=2)+'\n');print(summary.to_string(index=False));print('VALIDATION_SELECTED',chosen,'CHANGE%',gain,'DEVELOPMENT_GATE',gate,'FINAL_GOAL',False)

if __name__=='__main__':main()
