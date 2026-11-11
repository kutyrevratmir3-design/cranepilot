# Отдельный профиль FrozenRegimePilot

Bundle: `artifacts/regime_bundle_20261107`,36файлов,406575байт. Он содержит
прежний frozen bank, выбранные long/shortARX и forced-priors пятиcases,
selection и SHA256manifest. Исходные данные и trainingruns для inference
не нужны. Нужны численные исходники основного репозитория и его окружение;
это не автономный wheel или замена основного neuralCranePilot.

```python
from cranepilot.regime_pilot import FrozenRegimePilot

pilot = FrozenRegimePilot()  # также можно передать путь к скопированному bundle
result = pilot.forecast(history, family="simult", axis="theta")
forecast_rad = result["forecast_rad"]
```

`history` — словарь NumPy-массивов одинаковой формы(N,401),100Hz,
newestfirst: первый отсчёт — текущий, последний — ровно4sназад.

| family | axis | Обязательные каналы истории |
|---|---|---|
| trolley | theta | theta,L,x,xr |
| jib | theta или beta | выбранный угол,L,J,Jr |
| simult | theta или beta | выбранный угол,L,x,xr,J,Jr |

Углы/J — радианы, x/L — метры, L>0. Native xr/Jr должны иметь теже единицы
и смысл, что исходные записи; они не объявлены ускорением или напряжением.
History может также содержать второй угол, но он не участвует в прогнозе
данной оси. Все переданные массивы должны быть конечными и согласованными.
Неверные shapes,unknownchannels,targets/futurefields отклоняются. Futureplan
не передаётся. API не восстанавливает отсутствующие временные метки и не
доказывает синхронизацию каналов по одной только форме массива.

Прогноз выдаётся для .10,.20,.35,.50,.75,1.00,1.50s. Другие результаты:

| Поле | Значение |
|---|---|
| legacy_forecast_rad | прежний history-only kernel на техже входах |
| active_forecast_rad | frozen drive-conditioned blend |
| weights | выпуклые веса(N,7,4) |
| context_ratio | max past x/J range / прежний порог2mmили2mrad |
| quiet_preserved | context_ratio≤1: точное сохранение legacy |
| expert_forecasts_rad | четыре прогноза(N,7,4) |
| prefix_errors_rad | шесть уже наблюдённых prefix-errors(N,6,4) |
| old_prior,active_prior | веса прежней/активной смеси |

Порядок экспертов: oldARX, activeARX, free-modal, forced-modal. Каждый пакет
обрабатывается независимо, без statecacheпредыдущих вызовов и дополнительных
измерений старше4s. Первые3sпредсказывают последнюю уже наблюдённую1s;
только эти past-errors корректируют веса. Данные после текущего момента
не поступают в predictor. Тихая ветка сохраняетlegacyточно даже при noise;
это не означает сохранение чистого прогноза при испорченной истории.

Прежний `FrozenReliabilityPilot` и его eligibility/known-plan контракт сохранены.
Legacy в новых сравнительных таблицах — тотже kernel в исследовательском
history-only режиме, включая окна вне прежнего gate. Новый API не заявляет
его прежнюю условную квалификацию на всех активных окнах.

Все оценки текущего профиля — ретроспективные. Это forecastкомпонент: будущая
L/траекторияsupport неизвестны, применяется ограниченное предположение о них.
Отсутствует идентифицированная native-actuator модель для оптимизации будущих
команд; выпуклость весов не доказывает устойчивость controller. Реальные
closed-loop/hardware и новые независимые записи ещё требуют проверки.

Loader проверяет хеши весов и численных sourcefiles. Не менять старые frozen
источники ради новых опытов: создать новую версию с отдельным manifest.
Replay/zero-bias/CPUlatency и synthetic stress sensitivity описаны в
`docs/REGIME_FUSION_2026_11_07_RU.md` и `results/regime_fusion_20261107`.
