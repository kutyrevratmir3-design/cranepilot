"""Selected global method and all outcomes, without claiming a fresh success."""
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

RUN=Path('artifacts/runs/modal_backtest_gate_20261107');OUT=Path('results/modal_backtest_gate_20261107')
def main():
    if (OUT/'report.json').exists():raise FileExistsError('Retain prior report')
    p=json.loads((RUN/'development.json').read_text());q=json.loads((RUN/'retrospective/provenance.json').read_text());s=json.loads((RUN/'selection.json').read_text());v=json.loads((OUT/'verification.json').read_text());assert v['passed'] and v['development_sha256']==sha256(RUN/'development.json') and v['retrospective_sha256']==sha256(RUN/'retrospective/provenance.json');verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);verify_hashes(q['result_hashes'],RUN/'retrospective')
    metrics=pd.read_csv(RUN/'retrospective/metrics.csv');cases=metrics.groupby(['case','scenario','method'],as_index=False).mae_deg.mean();records=cases.groupby(['scenario','method'],as_index=False).mae_deg.mean();cases.to_csv(OUT/'case_summary.csv',index=False);records.to_csv(OUT/'record_summary.csv',index=False);matrix=records.pivot(index='scenario',columns='method',values='mae_deg');summary=records.groupby('method',as_index=False).mae_deg.mean();summary.to_csv(OUT/'summary.csv',index=False);ids=np.random.default_rng(7112030).integers(0,7,(10000,7));ref=matrix.reference.to_numpy();comparisons=[]
    for method in matrix:
        a=matrix[method].to_numpy();effect=100*(a[ids].mean(1)/ref[ids].mean(1)-1);lo,hi=np.quantile(effect,[.025,.975]);comparisons.append(dict(method=method,mae_deg=float(a.mean()),change_percent=float(100*(a.mean()/ref.mean()-1)),scenario_wins=int((a<ref).sum()),descriptive_low=float(lo),descriptive_high=float(hi)))
    comparison=pd.DataFrame(comparisons);comparison.to_csv(OUT/'paired_comparisons.csv',index=False);chosen=comparison[comparison.method=='selected'].iloc[0];gate=bool(chosen.change_percent<=-10 and chosen.scenario_wins>=5 and chosen.descriptive_high<0);weight_rows=[]
    for file in sorted((RUN/'retrospective').glob('*.npz')):
        with np.load(file) as f:
            w=f['weight_'+s['selected']];weight_rows.append(dict(file=file.name,mean_modal_weight=float(w.mean()),p05=float(np.quantile(w,.05)),median=float(np.median(w)),p95=float(np.quantile(w,.95))))
    weights=pd.DataFrame(weight_rows);weights.to_csv(OUT/'weights.csv',index=False);acceptance=dict(selected=s['selected'],development_gate_passed=s['development_gate_passed'],retrospective_numerical_gate=gate,independent_confirmation=False,final_goal_achieved=False,comparison=chosen.to_dict(),scope='Reliability development on fully opened public records; no fresh confirmation')
    (OUT/'acceptance.json').write_text(json.dumps(acceptance,indent=2)+'\n');fig,ax=plt.subplots(figsize=(10,5),layout='constrained');x=np.arange(7);width=.25
    for j,(method,label,color) in enumerate([('reference','Stronger ARX','#3f7183'),('modal','Frozen modal','#b8a18b'),('selected','Past-error blend','#bb8247')]):ax.bar(x+(j-1)*width,matrix[method],width,label=label,color=color)
    ax.set_xticks(x,[str(i) for i in matrix.index]);ax.set_xlabel('Previously opened public scenario');ax.set_ylabel('MAE (degrees), equal axes/horizons');ax.set_title('Causal reliability selection: retrospective development');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(frameon=False);fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
    doc=Path('docs/MODAL_BACKTEST_GATE_2026_11_07_RU.md');dev=pd.DataFrame([dict(case=k,**value) for k,value in s['case_scores'].items()])
    doc.write_text(f'''# Выбор прогноза по последней наблюдённой секунде

Протокол до fitting: `{p['git_head']}`. Все33старых сценария уже были открыты;
этот этап **ретроспективная разработка**, не независимое подтверждение.
Старый portable bundle/defaults/APHYNITY не заменены.

По VALIDATION глобально выбран **{s['selected']}**, изменение против более
сильного ARX **{s['validation_change_percent']:+.3f}%**, побед
**{s['validation_case_wins']}/5**, gate={s['development_gate_passed']}.
Reference дополнительно выбирает3sили4s историю по validation; он усилен.
Новых3sARXfits только5, прежние representation/form/alpha унаследованы.

На старых7условиях selectedMAE **{chosen.mae_deg:.6f}°** против ARX
**{ref.mean():.6f}°**, **{chosen.change_percent:+.3f}%**, побед
**{int(chosen.scenario_wins)}/7**. Whole-scenario descriptive95%interval
**[{chosen.descriptive_low:+.2f}%,{chosen.descriptive_high:+.2f}%]**.
Numerical retrospective gate={gate}. **Общая цель всё ещё не достигнута:
новых независимых записей не получено.** Статистическое условие на многократно
открытых данных не превращает их в новый test.

{table(comparison)}

{table(matrix.reset_index())}

## Почему выбор каузальный

В текущей4sистории первые3s(от−4до−1s) используются для прогноза последней
уже наблюдённой секунды. Проверяются шесть прошлых точек, включая текущий угол.
Частота prefix оценивается только изprefix, nominal length берётся в−1s.
Средние ошибки этихpastforecasts em,er задают вес modal=er²/(em²+er²+quantum²/12).
Это эвристический вес, не вероятность/гарантия. Реальный прогноз после0s
смешивает полный4smodal и выбранный ARX. Будущие после0s targets/length/position
не участвуют в выборе. Новый future gate не введён.

Шесть операторов заранее заданы; один общий выбран по прежнейvalidation.
Все варианты сохранены; выбирать лучший по7records после их открытия нельзя.
Прежний источник ошибки21 не объясняется крупным будущим подъёмом длины:
см. MODAL_FAILURE_DIAGNOSIS_20261107_RU.md. Гипотеза надёжности проверена
на уровне прогноза, не теоремой устойчивости переключаемого контроллера.

Все validation исходы:

{table(dev)}

Веса выбранного алгоритма по cases, computed изpastonly ошибок:

{table(weights)}

## Проверки и границы

204tests PASS перед fitting. Полныйraw-to-forecast replay:
{v['forecast_comparisons']} сравнений, всеhead metrics пересчитаны, глобальный
выбор и3s/4sreference выбор воспроизведены. Future target poisoning не меняет
backtest/evidence/weights/forecasts. Все1281оригинальные main/APHYNITY файла
сохранены. Новые открытые leads не дали подходящих рядов: IRV содержит
таблицы freq/zeta/peakacceleration, Haarhoff репозиторий описывает симуляции;
подробности DATA_ACCESS_FOLLOWUP_20261107.md. Агент авторам не писал.

Это та же отдельная4s residual-sway задача, не исходный .4sactiveforecast,
hardware или SOTA. Для полной цели нужен frozen test подходящих новыхraw
recordings. Пользователь может запросить их у Talapiden/Alhassan. До получения
данных сравнение настарых33условиях нельзя выдавать за независимое.
''')
    report=dict(document=str(doc),document_sha256=sha256(doc),development_sha256=sha256(RUN/'development.json'),retrospective_sha256=sha256(RUN/'retrospective/provenance.json'),verification_sha256=sha256(OUT/'verification.json'),report_source_sha256=sha256(Path(__file__)),selection=s['selected'],acceptance=acceptance)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(comparison.to_string(index=False));print('SELECTED',s['selected'],'RETROSPECTIVE_GATE',gate,'FINAL_GOAL',False)
if __name__=='__main__':main()
