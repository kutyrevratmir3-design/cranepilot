"""Development-only modal report before the seven conditional records are scored."""
import json
from pathlib import Path
import pandas as pd
from cranepilot.data import sha256
from scripts.report_angular_control import table
from scripts.run_external_benchmark import verify_hashes

RUN=Path('artifacts/runs/modal_history_20261107');OUT=Path('results/modal_history_20261107')
def main():
    if (OUT/'report.json').exists():raise FileExistsError('Retain previous report')
    p=json.loads((RUN/'provenance.json').read_text());s=json.loads((RUN/'selection.json').read_text());v=json.loads((OUT/'verification.json').read_text());assert v['passed'] and v['provenance_sha256']==sha256(RUN/'provenance.json')
    verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN)
    summary=pd.DataFrame([dict(case=name,**c['validation_scores']) for name,c in s['cases'].items()]);summary.to_csv(OUT/'summary.csv',index=False)
    doc=Path('docs/MODAL_DEVELOPMENT_2026_11_07_RU.md')
    doc.write_text(f'''# Локальная оценка мод: development

Commit до fitting: `{p['git_head']}`. Отдельная задача: прогноз остаточных
колебаний с4sкаузальной историей при неподвижных прошлом приводе/длине и
постоянном известном reference. Горизонты .10–1.50s. ARX получает ту же историю
и все те же доступные каналы; train-only calibration общая. Это не исходная
.4sактивная задача и не аппаратное управление.

Средняя MAE выбранной modal модели **{s['validation_mean_selected_deg']:.6f}°**,
усиленного ARX **{s['validation_mean_arx_deg']:.6f}°**:
**{s['change_percent']:+.3f}%**, побед{s['case_wins']}/5.
Development gate: **{s['development_gate_passed']}**. На simult/beta модель хуже;
это не универсальная победа. Только3знакомых validationrecords, оси зависимы.
Пять выбора по validation увеличивают оптимизм; независимости здесь нет.

{table(summary)}

На каждый case24ARXfits(direct/residual/LPV,6ridge),10localARsettings,
3modalварианта. Все120обученных ARX сохранены и повторно проверены.
Двухмодовая модель выбрана во всех5cases. Частота/затухание/состояние оцениваются
только по последним4s; неизвестные будущие измерения и targets игнорируются.
Отдельно сохранены все абляции, coverage, diagnostics и отрицательные исходы.

195unit tests PASS до fitting; точный replay {v['forecasts_replayed']} forecasts,
все120ridgefits/metrics/selection проверены. Все1281исходных main/APHYNITY файлов
сохранены. До завершения этой разработки IDs9,13,21,22,23,32,33 не читались
runner/verification. Metadata audit не нашёл старых сохранённых scores для них,
но полный набор ранее импортировался. Это возможный условный holdout, не новые
независимо собранные измерения. Общая цель пока не достигнута.
''')
    (OUT/'report.json').write_text(json.dumps(dict(document=str(doc),document_sha256=sha256(doc),provenance_sha256=sha256(RUN/'provenance.json'),verification_sha256=sha256(OUT/'verification.json'),source_sha256=sha256(Path(__file__)),development_gate_passed=s['development_gate_passed'],confirmation_records_read=False,final_goal_achieved=False),indent=2)+'\n')
    print(summary.to_string(index=False))
if __name__=='__main__':main()
