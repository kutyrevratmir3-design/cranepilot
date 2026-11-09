# Полный плотный ARX-контроль без перевыбора CranePilot

Протокол/код до fitting: `40762792fab299a15925cbe515037a40856e16f2`. Выбранный ранее `mix_bank_10`
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

| case | records | original_windows | new_dense_base_windows |
| --- | --- | --- | --- |
| jib_beta | 6 | 2250 | 2244 |
| jib_theta | 6 | 2250 | 2244 |
| simult_beta | 7 | 2625 | 2618 |
| simult_theta | 7 | 2625 | 2618 |
| trolley_theta | 10 | 3750 | 3740 |

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

VAL balanced MAE, равные случаи/age/шум: primary 1.574710°,
прежний ARX 1.794505°, новый 1.757634°.
Снижение primary относительно нового ARX:
10.41%. Это внутренний VAL-result.
Сам primary и его старый gate заново не выбирались и не переписывались.

| case | age_ms | key | validation_mae_deg |
| --- | --- | --- | --- |
| trolley_theta | 0 | drive3/forced_3s_adapt | 0.510027 |
| trolley_theta | 20 | drive3/quadratic_0.25 | 0.565099 |
| trolley_theta | 50 | augmented/dense/arx_io_lpv_direct_1e-06.npz | 0.588609 |
| trolley_theta | 100 | drive3/forced_3s_fixed | 0.650467 |
| trolley_theta | 150 | drive3/forced_3s_fixed | 0.687147 |
| trolley_theta | 200 | drive3/forced_3s_fixed | 0.71244 |
| jib_theta | 0 | legacy3/forced_3s_adapt | 1.52244 |
| jib_theta | 20 | legacy3/quadratic_0.25 | 1.53118 |
| jib_theta | 50 | legacy3/quadratic_0.25 | 1.57866 |
| jib_theta | 100 | legacy4/forced_3s_fixed | 1.67282 |
| jib_theta | 150 | legacy3/forced_3s_adapt | 1.73423 |
| jib_theta | 200 | legacy3/forced_3s_fixed | 1.70975 |
| jib_beta | 0 | drive4/forced_3s_adapt/arx_angle_lpv_direct_1e-06.npz | 1.38881 |
| jib_beta | 20 | drive4/quadratic_0.25/arx_angle_lpv_direct_1e-08.npz | 1.31858 |
| jib_beta | 50 | drive4/quadratic_0.25/arx_angle_lpv_direct_1e-06.npz | 1.40175 |
| jib_beta | 100 | drive4/quadratic_0.25/arx_io_lpv_direct_0.01.npz | 1.68853 |
| jib_beta | 150 | augmented/dense/arx_io_lpv_direct_0.0001.npz | 1.7558 |
| jib_beta | 200 | augmented/dense/arx_io_lpv_direct_0.0001.npz | 1.76777 |
| simult_theta | 0 | augmented/latency/arx_angle_lpv_direct_1e-10.npz | 2.17359 |
| simult_theta | 20 | augmented/latency/arx_angle_lpv_direct_1e-10.npz | 2.25056 |
| simult_theta | 50 | augmented/dated/arx_angle_lpv_direct_1e-10.npz | 2.20032 |
| simult_theta | 100 | augmented/dated/arx_angle_lpv_direct_1e-10.npz | 2.4548 |
| simult_theta | 150 | augmented/dated/arx_angle_lpv_direct_1e-08.npz | 2.71018 |
| simult_theta | 200 | augmented/dated/arx_io_lpv_direct_0.01.npz | 2.88646 |
| simult_beta | 0 | drive4/forced_3s_adapt/arx_angle_lpv_direct_1e-10.npz | 2.29687 |
| simult_beta | 20 | augmented/dense/arx_angle_lpv_direct_1e-06.npz | 2.3368 |
| simult_beta | 50 | augmented/dense/arx_angle_lpv_direct_1e-08.npz | 2.41891 |
| simult_beta | 100 | augmented/dense/arx_angle_lpv_direct_1e-10.npz | 2.65437 |
| simult_beta | 150 | drive4/forced_3s_fixed/arx_angle_lpv_direct_0.0001.npz | 2.77208 |
| simult_beta | 200 | drive3/forced_3s_fixed | 2.78995 |

## Старый retrospective replay

| age_ms | selected | matched_arx | dense_matched_arx | parent_quad | bank_simplex_1 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.525065 | 1.0738 | 1.07091 | 0.525065 | 0.525065 |
| 20 | 0.531247 | 1.09615 | 1.0759 | 0.554539 | 0.606507 |
| 30 | 0.530262 | 1.11453 | 1.08383 | 0.560087 | 0.609737 |
| 50 | 0.535381 | 1.06068 | 1.14518 | 0.577427 | 0.617924 |
| 70 | 0.544667 | 1.09428 | 1.16983 | 0.605676 | 0.614783 |
| 100 | 0.565824 | 1.07673 | 1.18277 | 0.667135 | 0.626112 |
| 150 | 0.603989 | 1.07637 | 1.15927 | 0.846449 | 0.634828 |
| 200 | 0.668859 | 1.23214 | 1.30672 | 1.11036 | 0.664884 |
| 250 | 0.742453 | 1.24583 | 1.31597 | 1.47869 | 0.692257 |

Таблица: равные 7 сценариев, оси внутри сценария и 7 горизонтов. Поддержка
при одинаковых age/noise/origins проверена; это не новый физический набор.
Будущие измеренные команды/положение не поступали в прогноз. Synthetic lag
сохраняет записанное управление и не пересчитывает замкнутый контур.

Balanced-представление, равный вес пяти cases и quiet/drive. Здесь quiet
означает отсутствие прошлого движения опоры и включает length_active;
изменение длины отдельно не балансируется. У jib22/23 drive содержит всего
2 из 94 окон, у simult33 — 3, но получает половину веса. Поэтому эта оценка
не заменяет таблицу с равными физическими сценариями выше.

| age_ms | selected | matched_arx | dense_matched_arx |
| --- | --- | --- | --- |
| 0 | 1.07721 | 1.69402 | 1.64371 |
| 20 | 1.04012 | 1.71117 | 1.67019 |
| 30 | 1.01868 | 1.7351 | 1.67662 |
| 50 | 0.996905 | 1.66119 | 1.70881 |
| 70 | 0.992478 | 1.7057 | 1.7358 |
| 100 | 1.0181 | 1.70299 | 2.19601 |
| 150 | 1.06326 | 1.75118 | 2.68984 |
| 200 | 1.16738 | 2.67448 | 3.53308 |
| 250 | 1.32366 | 2.69682 | 3.57103 |

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
origins/targets подтверждена. Source/raw/portable replay: 544824
векторов, max error 0 рад. Исходные main/APHYNITY
файлы и прежние bundles не изменялись. Полный отчёт проверки содержит
покрытие/clock probes и хеши; сам зелёный тест не доказывает независимость.

Отдельный comparator-only bundle: 20 файлов,
1914777 байт. Он требует
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
