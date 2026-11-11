"""Publish all shared-response development outcomes, including the failed gain gate."""
import json
from pathlib import Path
import pandas as pd
from cranepilot.data import sha256
from scripts.report_angular_control import table
from scripts.run_external_benchmark import verify_hashes

RUN=Path('artifacts/runs/shared_response_20261107');OUT=Path('results/shared_response_20261107')
def main():
    if (OUT/'report.json').exists():raise FileExistsError('Retain prior report')
    p=json.loads((RUN/'provenance.json').read_text());s=json.loads((RUN/'selection.json').read_text());v=json.loads((OUT/'verification.json').read_text())
    assert v['passed'] and v['provenance_sha256']==sha256(RUN/'provenance.json')
    verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN)
    m=pd.read_csv(RUN/'validation_metrics.csv');records=m.groupby(['record','method'],as_index=False).mae_deg.mean();summary=records.groupby('method',as_index=False).mae_deg.mean();records.to_csv(OUT/'record_summary.csv',index=False);summary.to_csv(OUT/'summary.csv',index=False)
    matrix=records.pivot(index='record',columns='method',values='mae_deg');wins=int((matrix.selected<matrix.strong_arx).sum());gate=s['change_percent']<=-10 and wins>=3
    doc=Path('docs/SHARED_RESPONSE_2026_11_07_RU.md')
    doc.write_text(f'''# Общая импульсная характеристика: разработка

Протокол до fitting: `{p['git_head']}`. Те же 12 TRAIN и 4 многократно
использованных VALIDATION записи внешнего rotary-camera крана. Старый TEST
17–20 не читался. Независимого подтверждения нет.

Вместо отдельных коэффициентов для каждого горизонта проверены общие моды
состояния и одна causal FIR характеристика по известным будущим командам.
18 ridge fits, 90 validation comparisons с фиксированными blends против
усиленного ARX. Все исходы сохранены; это повышает риск переоптимизации validation.

Выбран {s['selected']}, blend={s['blend']}. Validation MAE
{s['validation_mae_deg']:.9f}° против ARX {s['reference_mae_deg']:.9f}°:
**{s['change_percent']:+.3f}%**, побед {wins}/4. Gate ≥10% и ≥3/4 побед: **{gate}**.
Цель доказанного преимущества ещё не достигнута. Defaults, bundle и APHYNITY
не изменены.

{table(summary)}

187 unit tests PASS до fitting. Перезапущены все {v['fits_replayed']} fits и
{v['comparisons_replayed']} validation comparisons, прогнозы и метрики совпали.
1281 исходных файла сохранены. Ни будущие измеренные углы, ни test centering,
ни будущие timestamps не используются. Это классическая структурированная
регрессия, не заявление новой теории управления.
''')
    r=dict(document=str(doc),document_sha256=sha256(doc),provenance_sha256=sha256(RUN/'provenance.json'),verification_sha256=sha256(OUT/'verification.json'),report_source_sha256=sha256(Path(__file__)),independent_confirmation=False,final_goal_achieved=False,development_gate_passed=bool(gate),record_wins=wins,selection=s)
    (OUT/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(summary.to_string(index=False));print('GATE',gate)
if __name__=='__main__':main()
