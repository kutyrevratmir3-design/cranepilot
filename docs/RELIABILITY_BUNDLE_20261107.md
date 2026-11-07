# FrozenReliabilityPilot: отдельный каузальный кандидат

Комплект `artifacts/reliability_bundle_20261107`:19файлов,182278байт.
Содержит неизменный предыдущий modal bundle, пять новых3sARX checkpoints,
заранее выбранный общийsoft-оператор, правила задачи и source/fileSHA.
Inference не требует raw data, training или старого дерева runs; нужны исходники
основного репозитория и егоPython зависимости. APHYNITY не используется.

```python
from cranepilot.reliability_pilot import FrozenReliabilityPilot
pilot = FrozenReliabilityPilot()
result = pilot.forecast(
    history, future_reference,
    family="trolley", axis="theta", known_plan=True,
)
```

Контракт входов тот же, что в MODAL_BUNDLE_20261107.md: past(N,401), newest
first,100Hz; Lвметрах, углы врадианах, необходимые позиции/native references.
Future reference(N,151), current first, действительно известный заранее план,
не будущиеoutputsfeedbackконтроллера. Родительскийvalidator проверяет shapes,
finitevalues, положительнуюL, current reference и settled-input eligibility.
Вне разрешённой задачи прогноз/весNaN. При воспроизведении оценки origins≥25s.

Выходы: `forecast_rad`, `reference_arx_rad`, `modal_forecast_rad`(N,7),
`modal_weight`(N,), `backtest_mae_rad`(modal/arx), `eligible`, `horizons_s`.
Веса вычисляются по прогнозу последней наблюдённой1s из предыдущих3s; будущие
targets после текущего момента не используются. Это эвристика надёжности,
не вероятность или формальная гарантия устойчивости контроллера.

Сводка: validation−30.14%,5/5casewins. Старые7сценариев .189779°vs.478509°,
−60.34%,6/7wins, descriptive95%interval[−84.03%,−23.25%]. Этап спроектирован
после открытия этих данных: **это retrospective, независимая цель не закрыта**.
Scenario21 всё ещё проигран .403819°vs.375804°, хотя прежнийmodal давал .700020°.

Проверки:206unit tests PASS;80700raw-to-forecast comparisons и всеметрики
повторены точно. Portable API повторил9302окна во всех17validation/retrospective
case-records точно. Сохранены исходные1281файла main/APHYNITY и старыеbundles.
Перваяверсия verifier имела missingimport; исходник/trace сохранены в
results/modal_backtest_gate_20261107/verification_attempt_1. Исправлен только
verifier, без fitting/смены весов/прогнозов.

Observed median21–22ms/singlepacket наэтомMac; это не hard-real-time гарантия.
Отдельно проверен zero-angle/zero-reference input приL=.1,.4,.8m для5cases.
Чистыйmodal даёт точный0, смесь наследует небольшойinterceptbias ARX: максимум
**.035304°** на этих15фиксированных probes. Точное equilibrium constraint
для смеси не заявляется; thresholds/веса послеаудита не менялись.
Такой прогноз сам по себе не доказывает безопасность/улучшение closedloop.

Для новых данных сначала проверить units, clock, encoder resolution(в исходном
контракте2π/4096rad), тип/массу, platform и known-plan semantics. До их оценки
зафиксировать manifest SHA и протокол. Полный отчёт:
MODAL_BACKTEST_GATE_2026_11_07_RU.md; дополнительные источники данных:
DATA_ACCESS_FOLLOWUP_20261107.md. Авторам агент не писал.
