# Проверка границ: движение привода и подъём троса

Протокол/код до оценки: `396cee43d5ff9884fdb582b188ed89de52662daf`. Обучение, выбор модели и изменение
публичногоAPIgate не выполнялись. Все33записи уже открыты: это ретроспективная
квалификация существующихkernels, не новая независимая проверка.

## Результаты на прежних7оценочных записях

| group | method | mae_deg | reference_mae_deg | change_percent | scenario_wins | scenarios |
| --- | --- | --- | --- | --- | --- | --- |
| all | frozen_blend | 0.545399 | 0.8581 | -36.4411 | 6 | 7 |
| all | frozen_modal | 0.731735 | 0.8581 | -14.7262 | 3 | 7 |
| all | strong_arx | 0.8581 | 0.8581 | 0 | 0 | 7 |
| drive_active | frozen_blend | 1.72642 | 1.80549 | -4.37963 | 3 | 6 |
| drive_active | frozen_modal | 2.35101 | 1.80549 | 30.2146 | 2 | 6 |
| drive_active | strong_arx | 1.80549 | 1.80549 | 0 | 0 | 6 |
| drive_and_length_active | frozen_blend | 3.15703 | 3.21505 | -1.80474 | 2 | 5 |
| drive_and_length_active | frozen_modal | 4.27254 | 3.21505 | 32.8917 | 0 | 5 |
| drive_and_length_active | strong_arx | 3.21505 | 3.21505 | 0 | 0 | 5 |
| length_active | frozen_blend | 0.0989405 | 0.894749 | -88.9421 | 2 | 2 |
| length_active | frozen_modal | 0.0906641 | 0.894749 | -89.8671 | 2 | 2 |
| length_active | strong_arx | 0.894749 | 0.894749 | 0 | 0 | 2 |
| low_past_motion | frozen_blend | 0.193369 | 0.495183 | -60.9499 | 6 | 7 |
| low_past_motion | frozen_modal | 0.265718 | 0.495183 | -46.3394 | 6 | 7 |
| low_past_motion | strong_arx | 0.495183 | 0.495183 | 0 | 0 | 7 |
| old_numerical_scope | frozen_blend | 0.188822 | 0.477167 | -60.4285 | 6 | 7 |
| old_numerical_scope | frozen_modal | 0.261154 | 0.477167 | -45.2699 | 6 | 7 |
| old_numerical_scope | strong_arx | 0.477167 | 0.477167 | 0 | 0 | 7 |

Основные4режима вычислены исключительно по прошлым401отсчётам4s:
low_past_motion; drive_active(диапазонx>2mmилиJ>2mrad); length_active(L>2mm);
оба движения. Это метки наблюдённой истории, не доказательство отсутствия
forcing. Ни угол будущей ошибки, ни будущая фактическаяL/позиция не выбирают
группу и не исключают окна. Каждая группа имеет своё число сценариев;
еёсреднее не является оценкой тойже популяции, что другаягруппа.

Покрытие по каждому сценарию:

| role | group | scenario | windows | total_windows | fraction |
| --- | --- | --- | --- | --- | --- |
| old_evaluation | all | 9 | 375 | 375 | 1 |
| old_evaluation | all | 13 | 375 | 375 | 1 |
| old_evaluation | all | 21 | 375 | 375 | 1 |
| old_evaluation | all | 22 | 375 | 375 | 1 |
| old_evaluation | all | 23 | 375 | 375 | 1 |
| old_evaluation | all | 32 | 375 | 375 | 1 |
| old_evaluation | all | 33 | 375 | 375 | 1 |
| old_evaluation | drive_active | 9 | 32 | 375 | 0.0853333 |
| old_evaluation | drive_active | 13 | 7 | 375 | 0.0186667 |
| old_evaluation | drive_active | 21 | 142 | 375 | 0.378667 |
| old_evaluation | drive_active | 22 | 8 | 375 | 0.0213333 |
| old_evaluation | drive_active | 23 | 8 | 375 | 0.0213333 |
| old_evaluation | drive_active | 32 | 124 | 375 | 0.330667 |
| old_evaluation | drive_active | 33 | 0 | 375 | 0 |
| old_evaluation | drive_and_length_active | 9 | 18 | 375 | 0.048 |
| old_evaluation | drive_and_length_active | 13 | 28 | 375 | 0.0746667 |
| old_evaluation | drive_and_length_active | 21 | 18 | 375 | 0.048 |
| old_evaluation | drive_and_length_active | 22 | 0 | 375 | 0 |
| old_evaluation | drive_and_length_active | 23 | 0 | 375 | 0 |
| old_evaluation | drive_and_length_active | 32 | 18 | 375 | 0.048 |
| old_evaluation | drive_and_length_active | 33 | 10 | 375 | 0.0266667 |
| old_evaluation | length_active | 9 | 0 | 375 | 0 |
| old_evaluation | length_active | 13 | 275 | 375 | 0.733333 |
| old_evaluation | length_active | 21 | 0 | 375 | 0 |
| old_evaluation | length_active | 22 | 0 | 375 | 0 |
| old_evaluation | length_active | 23 | 0 | 375 | 0 |
| old_evaluation | length_active | 32 | 0 | 375 | 0 |
| old_evaluation | length_active | 33 | 297 | 375 | 0.792 |
| old_evaluation | low_past_motion | 9 | 325 | 375 | 0.866667 |
| old_evaluation | low_past_motion | 13 | 65 | 375 | 0.173333 |
| old_evaluation | low_past_motion | 21 | 215 | 375 | 0.573333 |
| old_evaluation | low_past_motion | 22 | 367 | 375 | 0.978667 |
| old_evaluation | low_past_motion | 23 | 367 | 375 | 0.978667 |
| old_evaluation | low_past_motion | 32 | 233 | 375 | 0.621333 |
| old_evaluation | low_past_motion | 33 | 68 | 375 | 0.181333 |
| old_evaluation | old_numerical_scope | 9 | 291 | 375 | 0.776 |
| old_evaluation | old_numerical_scope | 13 | 65 | 375 | 0.173333 |
| old_evaluation | old_numerical_scope | 21 | 215 | 375 | 0.573333 |
| old_evaluation | old_numerical_scope | 22 | 291 | 375 | 0.776 |
| old_evaluation | old_numerical_scope | 23 | 291 | 375 | 0.776 |
| old_evaluation | old_numerical_scope | 32 | 233 | 375 | 0.621333 |
| old_evaluation | old_numerical_scope | 33 | 50 | 375 | 0.133333 |

old_numerical_scope восстанавливает прежние численные условия(включая
timestamp≥25s и постоянную будущую записанную reference). Это дополнительная
описательная метка, не вход прогноза и не доказательство заранее известного
feedbackoutput. Все новые прогнозы работают только по истории; plannedfuture
reference или actualfutureL в них не передавались.

## Что именно проверено

53сценарий/ось сочетания, 19875пакетов, каждый из них оценён всеми5
методами без target/amplitude-based исключений. Отдельно сохраненыTRAIN,VAL,
old_evaluation; TRAIN/VALне выдаются за heldout. Семь горизонтов .10–1.50s.
Среднее сначала по горизонтам, затем по осям внутри сценария, затем равный вес
целых сценариев. Empty regimes остаются в coverage/metrics сwindows=0иNaN,
не становятся нулевойошибкой. Overlappingwindowsне считаются независимыми trials.

Изначальный4sпрофиль требует записанной истории. Первые4sкаждой записи здесь
неоцениваемы; нулеваяпредыстория невыдумывалась. Для будущей записи полезно
сохранить хотябы4sдо начала движения. Это отдельное ограничение от ошибки MAE.

Важное программное свойство: modal/softblend и сохранённыйsimpleARXв этом
опыте используют прошлыеуглы/L, без futurecommandplan. Они не определяют
реакцию на новую оптимизируемую команду и сами по себе не квалифицируютMPC.
ОсновнойneuralCranePilotимеет другой входнойконтракт; его оценки с.4sисторией
нельзя напрямую подменить этими4sцифрами. Семантика native xr/Jr и неизвестная
будущая фактическаяL остаются существенными для активного управления.

## Проверки и следующий шаг

210unit tests PASS перед запуском. Из исходных сохранённых сигналов и frozen
весов повторены 99375 forecasts, все regime/coverage/head
metrics независимо пересчитаны. Все1281оригинальные main/APHYNITY файла целы.
Public APIgateнеослаблен: outsideoriginaltaskон по-прежнему возвращаетNaN.
Negative out-of-scope scores сохранены. Архив/таблицы вresults/active_qualification_20261107.

Эта проверка уточняет границы применимости и направления дальнейшей разработки.
Она не закрывает исходную цель: нужны подходящие новые физические записи и
заранее зафиксированная независимая оценка. Новыеhoist/actuator методы требуют
отдельного протокола; нельзя передавать им измеренную будущуюдлину или ускорение.
