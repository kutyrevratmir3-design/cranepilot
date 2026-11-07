# Движение привода: физическая поправка и усиленный ARX

Протокол и численные исходники до fitting: `dbda863474491443538445da8e631ce760b1a0a4`.
Все33старые записи уже открыты; это ретроспективная разработка.
Будущие измеренные положение/длина/ускорение не передавались в прогноз.

## Что сделано

Физический локальный oscillator учитывает измеренную прошлую x/J через
точно интегрированный forced-response basis. Он оценивает текущее состояние
колебаний с TRAIN-приором coupling; два варианта онлайн(fixed/adapt) и два
предположения о будущем support(hold/ограниченная линейная экстраполяция).
Всего5обученных priors и4physical/4blend варианта. Это частичная локальная
модель, не полный MHE/MPC и не идентифицированный native-actuator.

Новая сильная база:120ARXgridfits на всех post4sTRAINокнах с теми же доступными
сигналами, плюс5short3sfits. Per-case VAL выбирает IO/angle-LPV,
direct/modal-residual, ridge, затем3s/4s/старыйARXfallback. Улучшение базы
само по себе не доказывает пользу физической поправки.

| case | reference | model | long | short | old |
| --- | --- | --- | --- | --- | --- |
| trolley_theta | short | arx_io_lpv_direct_1e-06.npz | 0.740791 | 0.64585 | 1.30343 |
| jib_theta | old | arx_io_lpv_direct_1e-06.npz | 1.61962 | 1.73925 | 1.40509 |
| jib_beta | long | arx_angle_lpv_direct_1e-08.npz | 1.41875 | 1.55252 | 1.68099 |
| simult_theta | long | arx_angle_lpv_direct_1e-10.npz | 2.20582 | 2.25409 | 2.59646 |
| simult_beta | long | arx_angle_lpv_direct_1e-10.npz | 2.26064 | 2.29794 | 2.45131 |

## Зафиксированный выбор по validation

Один общий метод из11: **blend_linear_adapt**. BalancedMAE меняется на
-11.99% против matchedARX;
выигрыш 5/5cases.
Заранее заданный developmentgate(≥10%gain,≥4/5wins,quiet≤1.05reference):
**True**. Никакого перевыбора по oldevaluation.

| method | balanced_mae_deg | quiet_mae_deg | selected |
| --- | --- | --- | --- |
| strong_arx | 1.58723 | 0.685534 | False |
| old_blend | 1.67093 | 0.340775 | False |
| free_modal | 2.06339 | 0.42997 | False |
| physical_hold_fixed | 2.04328 | 0.429921 | False |
| blend_hold_fixed | 1.421 | 0.396623 | False |
| physical_hold_adapt | 1.9963 | 0.429921 | False |
| blend_hold_adapt | 1.41028 | 0.396623 | False |
| physical_linear_fixed | 2.01913 | 0.429921 | False |
| blend_linear_fixed | 1.41574 | 0.396634 | False |
| physical_linear_adapt | 1.96296 | 0.429921 | False |
| blend_linear_adapt | 1.39697 | 0.396634 | True |

Balanced-score: равный вес пятиcases; внутриcase тихие и drive-active origins
имеют равный вес. Все7горизонтов. В fitting каждыйTRAINrecord имеет равный
вес и равные присутствующие driveгруппы. VALrecords4/17/28 прежние; новые
hoist-VALусловия не выдуманы. Длина меняется в частиTRAIN/oldevaluation.

## Ретроспективный replay прежних7сценариев

BalancedMAE selected=1.188271°, matchedARX=1.663313°,
изменение=-28.56%; casewins=5/5.
Здесь balanced-среднее сначала по presentquiet/drive внутриaxis/record,
потом поровнуrecords внутриcase и пятьcases. Это не та же агрегация, что
следующая таблица: в ней равный вес сценариев/осей/горизонтов отдельно по
regime. Две таблицы не сравнивать как одинаковый MAE.

| method | balanced_mae_deg |
| --- | --- |
| frozen_arx | 1.4144 |
| strong_arx | 1.66331 |
| old_blend | 1.2309 |
| selected | 1.18827 |
| matched_free_blend | 1.22854 |
| matched_local_zero_blend | 1.22856 |

| role | group | method | mae_deg | scenarios |
| --- | --- | --- | --- | --- |
| old_evaluation | all | frozen_arx | 0.8581 | 7 |
| old_evaluation | all | matched_free_blend | 0.555793 | 7 |
| old_evaluation | all | matched_local_zero_blend | 0.555833 | 7 |
| old_evaluation | all | old_blend | 0.545399 | 7 |
| old_evaluation | all | selected | 0.548902 | 7 |
| old_evaluation | all | strong_arx | 1.03408 | 7 |
| old_evaluation | drive_active | frozen_arx | 1.80549 | 6 |
| old_evaluation | drive_active | matched_free_blend | 1.73075 | 6 |
| old_evaluation | drive_active | matched_local_zero_blend | 1.73075 | 6 |
| old_evaluation | drive_active | old_blend | 1.72642 | 6 |
| old_evaluation | drive_active | selected | 1.65228 | 6 |
| old_evaluation | drive_active | strong_arx | 2.22905 | 6 |
| old_evaluation | drive_and_length_active | frozen_arx | 3.21505 | 5 |
| old_evaluation | drive_and_length_active | matched_free_blend | 2.44477 | 5 |
| old_evaluation | drive_and_length_active | matched_local_zero_blend | 2.44477 | 5 |
| old_evaluation | drive_and_length_active | old_blend | 3.15703 | 5 |
| old_evaluation | drive_and_length_active | selected | 2.28923 | 5 |
| old_evaluation | drive_and_length_active | strong_arx | 2.3029 | 5 |
| old_evaluation | length_active | frozen_arx | 0.894749 | 2 |
| old_evaluation | length_active | matched_free_blend | 0.0983299 | 2 |
| old_evaluation | length_active | matched_local_zero_blend | 0.0983299 | 2 |
| old_evaluation | length_active | old_blend | 0.0989405 | 2 |
| old_evaluation | length_active | selected | 0.0983313 | 2 |
| old_evaluation | length_active | strong_arx | 0.922915 | 2 |
| old_evaluation | low_past_motion | frozen_arx | 0.495183 | 7 |
| old_evaluation | low_past_motion | matched_free_blend | 0.24198 | 7 |
| old_evaluation | low_past_motion | matched_local_zero_blend | 0.24202 | 7 |
| old_evaluation | low_past_motion | old_blend | 0.193369 | 7 |
| old_evaluation | low_past_motion | selected | 0.241997 | 7 |
| old_evaluation | low_past_motion | strong_arx | 0.69987 | 7 |
| old_evaluation | old_numerical_scope | frozen_arx | 0.477167 | 7 |
| old_evaluation | old_numerical_scope | matched_free_blend | 0.237596 | 7 |
| old_evaluation | old_numerical_scope | matched_local_zero_blend | 0.237647 | 7 |
| old_evaluation | old_numerical_scope | old_blend | 0.188822 | 7 |
| old_evaluation | old_numerical_scope | selected | 0.237624 | 7 |
| old_evaluation | old_numerical_scope | strong_arx | 0.672236 | 7 |

Все consideredorigins≥4s сохранены, stride.25s, история4s, горизонты.10–1.50s.
Regimes только из прошлых401отсчётов, без target-based отбора. Пересекающиеся
окна не являются независимыми испытаниями. old_numerical_scope с future
recordedreference — только описательная историческая метка, не вход прогноза.
Малые числа сценариев в hoistгруппах не доказывают общую hoist-robustness.

## Проверка вклада физики

После первого trolleyVALoutput дополнительно зафиксированы два диагностических
контроля с тем же новымARX и тем же past-only softgate: free-modal blend и
локальный modal-refit blend с нулевыми forcedgains. Они не участвовали в выборе
11методов и developmentgate. Это поздняя ablation, её статус явно сохранён.
Выбранный метод лучше обоих контролей без forced-поправки на этом ретроспективном balanced-score; независимую пользу ещё предстоит проверить.

Нельзя приписывать всю разницу с old_blend физике: ARXпереобучен на активных
TRAINокнах и заново выбран поVAL. Сохранены все физические варианты, все125
ARXвесов/оценок, controls и проигрыши.

Для прозрачности frozen_arx — прежняя сильная база, перенесённая из уже
проверенного qualification без нового fitting. strong_arx — новый VAL-selected
active-trained reference. Его выбор наVAL не гарантирует лучший перенос;
в oldevaluation он местами хуже прежнего ARX. Обе базы показаны, результаты
не сводятся к проценту против только более удобного comparator.

## Численная проверка и границы

218testsPASS доfitting, включая независимую matrix-exponential проверку
first-order-hold. Повторены все125TRAINARXfits и5physicalpriors; selection,
прогнозы и всеregime/headmetrics пересчитаны. 87375
forecastcomparisons. Все1281оригинальные main/APHYNITYфайла целы.

| case | max_absolute_basis_gain | condition_median | condition_max |
| --- | --- | --- | --- |
| trolley_theta | 0.447581 | 1 | 1 |
| jib_theta | 0.0444001 | 1 | 8.51472 |
| jib_beta | 0.418581 | 1 | 5.13662 |
| simult_theta | 0.674631 | 1 | 166.384 |
| simult_beta | 2.3683 | 1 | 128.907 |

Диагностика condition относится к регуляризованной малой системе онлайнfit,
не является доказательством физической идентифицируемости. Веса basis могут
поглощать ошибки геометрии и датчиков. Изменение Lза горизонт и native
xr/Jr→реализованное движение здесь не идентифицированы. Гипотетический
support-path interface проверен синтетически, не заменяет actuator дляMPC.
Новый метод не менял прежние frozenbundles, publicAPIgate или neuraldefault.

Результат расширяет проверку до активных участков, но исходную независимую
цель не закрывает. Нужны новые сырые записи авторов/стенда с единицами,
временем, командами и минимум4sпредыстории. Публикационные/SOTA/аппаратные
выводы до такого теста преждевременны.
