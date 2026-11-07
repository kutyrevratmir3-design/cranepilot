"""Do not base the scientific conclusion on a nuisance-sensitive reference."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from scripts.report_angular_control import table
from scripts.run_external_benchmark import verify_hashes

RUN=Path('artifacts/runs/modal_reference_challenge_20261107');OUT=Path('results/modal_reference_challenge_20261107')
def main():
    if (OUT/'report.json').exists():raise FileExistsError('Retain previous report')
    p=json.loads((RUN/'development.json').read_text());q=json.loads((RUN/'retrospective/provenance.json').read_text());v=json.loads((OUT/'verification.json').read_text());assert v['passed'] and v['development_sha256']==sha256(RUN/'development.json') and v['retrospective_provenance_sha256']==sha256(RUN/'retrospective/provenance.json');verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);verify_hashes(q['result_hashes'],RUN/'retrospective')
    m=pd.read_csv(RUN/'retrospective/metrics.csv');records=m.groupby(['scenario','method'],as_index=False).mae_deg.mean();records.to_csv(OUT/'record_summary.csv',index=False);matrix=records.pivot(index='scenario',columns='method',values='mae_deg');summary=records.groupby('method',as_index=False).mae_deg.mean();summary.to_csv(OUT/'summary.csv',index=False)
    a=matrix.selected_modal.to_numpy();b=matrix.challenged_arx.to_numpy();ids=np.random.default_rng(7112030).integers(0,len(b),(10000,len(b)));effect=100*(a[ids].mean(1)/b[ids].mean(1)-1);lo,hi=np.quantile(effect,[.025,.975]);change=100*(a.mean()/b.mean()-1);wins=int((a<b).sum());conditional=bool(change<=-10 and wins>=5 and hi<0)
    result=dict(modal_mae_deg=float(a.mean()),challenged_arx_mae_deg=float(b.mean()),change_percent=float(change),scenario_wins=wins,scenarios=7,descriptive_low=float(lo),descriptive_high=float(hi),numerical_gate_against_challenged_reference=conditional,independent_confirmation=False,final_goal_achieved=False,post_hoc_design=True)
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');choices=json.loads((RUN/'selection.json').read_text());choice_table=pd.DataFrame([dict(case=k,**v) for k,v in choices.items()]).drop(columns=['file','checkpoint_sha256'],errors='ignore')
    doc=Path('docs/MODAL_REFERENCE_CHALLENGE_2026_11_07_RU.md')
    doc.write_text(f'''# Проверка силы ARX: преимущество меньше, уверенности пока мало

Дополнительный протокол зафиксирован **после** просмотра7holdout результатов;
он ретроспективный. Модальная модель и её прогнозы не изменялись. Сравнение
с первоначальным full-input ARX не удалено, но его большой провал переноса
нельзя использовать как единственное основание сильного научного вывода.

## Итог против улучшенного сравнения

Modal MAE **{a.mean():.6f}°**, strengthened simple ARX **{b.mean():.6f}°**:
**{change:+.3f}%**, побед **{wins}/7**. Whole-scenario paired bootstrap95%
описательный интервал **[{lo:+.2f}%, {hi:+.2f}%]**. Он включает0:
надёжное статистическое преимущество ещё не установлено. Против этого
reference численный gate с требованием upper<0: **{conditional}**.
**Исследовательская цель не достигнута.** Нужны новые независимые записи.

{table(summary)}

{table(matrix.reset_index())}

Особенно важен провал scenario21: modal хуже новогоARX примерно вдвое.
Усреднение не должно скрывать этот случай. Изменение−88.8% против первого
ARX верно для заранее замороженного сравнения, но консервативный рабочий
вывод теперь использует−46.9% и широкий интервал против более сильной базы.

## Что сделано для честного сравнения

Все240новых fits сохранены. ARX получил4простых representations: только углы,
углы×частота, разности относительно текущего угла, частотные interactions
разностей. Direct и harmonicresidual,6ridge. Нет абсолютных координат/команд,
которые могли выучить неустойчивые корреляции. Первый frozenARX оставался
допустимым fallback. Выбор только по прежним validation, не по7records.
Все12сценарий/ось predictions проверены повторно, метрики пересчитаны; старые
modal/ARX/localAR прогнозы не изменялись. 1281оригинальный файл сохранён.

{table(choice_table)}

Наблюдаемое уменьшение ошибки не доказывает новую архитектуру, SOTA,
управление на оборудовании или превосходство на исходной .4sактивной задаче.
Это отдельный4s settled-input forecast. Отложенные7публичных записей теперь
полностью открыты и не могут снова считаться свежими. Новые данные авторов
нужно проверять после отдельного заранее замороженного протокола.
''')
    (OUT/'report.json').write_text(json.dumps(dict(document=str(doc),document_sha256=sha256(doc),development_sha256=sha256(RUN/'development.json'),retrospective_sha256=sha256(RUN/'retrospective/provenance.json'),verification_sha256=sha256(OUT/'verification.json'),report_source_sha256=sha256(Path(__file__)),comparison=result),indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
