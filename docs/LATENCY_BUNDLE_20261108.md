# FrozenLatencyPilot: известный возраст измерения угла

Это отдельный экспериментальный профиль с bundle
`artifacts/latency_bundle_20261108`. Ему нужны исходники основного проекта и
его окружение. Для inference не нужны rawданные, trainingruns или исходный
архив эксперимента. Manifest фиксирует выбранный поVAL способ компенсации,
параметры сильного ARX-соперника, вложенный frozenregimebank и SHA256source/weights.

```python
from cranepilot.latency_pilot import FrozenLatencyPilot

pilot = FrozenLatencyPilot()
result = pilot.forecast(
    history,
    family="simult",
    axis="theta",
    angle_age_samples=5,  # известные50ms из acquisition/current timestamps
    include_reference=True,
)
forecast_rad = result["forecast_rad"]
matched_arx_rad = result["matched_arx_rad"]
```

Все historyмассивы имеют форму(N,401),100Hz,newestfirst. Списокканалов
тотже, что в REGIME_BUNDLE_20261107.md: выбранныйугол,L,x/J,xr/Jr в зависимости
отfamily. Углы/J вradians,x/Lвmetres,L>0; nativecommands в исходныхединицах.
Не назначать nativexr/Jr смысл ускорения/напряжения без metadata.

Ключевое различие часов: история выбранногоугла заканчивается на
current_time−angle_age_samples*.01s. Всеостальные каналы заканчиваются на
current_time. Возраст — известныйцелыйscalar0..15, одинаковыйдляbatch.
None/неизвестныйвозраст не поддерживается. Значение0 означает синхронныйугол,
а не автоматическое обнаружение задержки. История должна быть равномерной
по acquisitiontime; переменный пропуск/OOSM/дробныйage требует отдельного
заранее зафиксированного causaladapter. Простаяshapeпроверка не проверяетclock.

Пример50ms: самыйновыйугол соответствуетt−.05s, самыйстарыйt−4.05s;
самыеновые L/x/J/references соответствуютt,самыестарыеt−4s. Не надо молча
сдвигать supportвместе с углом или заменять отсутствующий старыйуголpadding.
На записи должна существовать такая предыстория. Futureизмерения не передаются.

Адаптер переставляет уже наблюдённыйугол на нужныеtimestamps и оценивает
только недостающие новейшиеотсчёты. Прифизическомспособе поддерживающее
движение междуsensorclock иcurrentclock уже наблюдено к моменту вызова;
это не oraclefutureinput. Выборспособа заморожен поVAL, а не по ошибкам
пользовательской новойзаписи. Приage0прогноз точно прежнийFrozenRegimePilot.

Результаты:

| Поле | Смысл |
|---|---|
| forecast_rad | прогноз(N,7) для .10,.20,.35,.50,.75,1,1.5s послеcurrenttime |
| head_estimate_rad | (N,age) оценённые новейшиеуглы, newestfirst; не наблюдения |
| backtest_valid | шесть validheadflags: приpositiveage последняяfalse |
| weights | выпуклые веса четырёхfrozenexperts(N,7,4) |
| matched_arx_rad | приinclude_reference=True, frozenpercaseVALreference |

Headestimates исключены изpasttruthпри перевзвешивании; их нельзя выдавать
за новые сенсорныеизмерения. Include_reference нужен длясравнения и может
стоить дополнительного inference, если у reference другойспособсинхронизации.

Нынешние оценки — syntheticknownage/jitter на ранее открытых физических
записях. Это не выявление фактического sensorlag стенда, не unseenphysicaltrial,
не гарантия real-time/controllerстабильности. Nativeactuator map и реакцияна
новыйоптимизируемый command не идентифицированы. ПредыдущиеAPI/default/bundles
и APHYNITY сохранены. Независимаяцель требует новых подходящихrawtrials,
метаданныхclock/units/encoderq иfreezeпротокола доихtargeterrors.

Все8адаптеров/120ARXfits, selection, контрольслабыхмест, replay и CPUtiming
сохранены вresults/latency_observer_20261108 и LATENCY_OBSERVER_2026_11_08_RU.md.
Дополнительный post-hoc age-conditionedARXchallenge из отчёта остаётся отдельным
соперником. APIполеmatched_arx_rad сохраняет первоначальный глобальныйVALreference;
опубликованный первичныйbundle не пересобирался после егооценок.
