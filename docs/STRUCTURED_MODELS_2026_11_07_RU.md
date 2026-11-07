# Структурированные модели CranePilot — 7 ноября 2026

Завершены 72 обучений: 8 новых вариантов × 3 разбиения × 3 seed. Выбор общей стратегии по validation — `previous_local`. Заранее заданные критерии принятия: **selected: не выполнен; common: выполнен**. Локальные решения: primary: `adapter_direct8`, short_transfer: `previous_local`, long_transfer: `latent_residual6`.

В 9 fit выбран checkpoint эпохи −1: 9 добавочных моделей точно сохраняют исходный soft, 0 самостоятельных моделей сохраняют собственную начальную динамику. Это не улучшение от обучения. Даже успешный gate строки `previous_local` означает сохранение ранее выбранных моделей, а не новое архитектурное достижение.

## Результат предварительного выбора

| fold | regime | mae_deg | change_vs_previous_local_percent | change_vs_soft_symmetry_percent |
| --- | --- | --- | --- | --- |
| long_transfer | active | 0.901386 | -3.26949 | 0.959211 |
| long_transfer | all | 0.501157 | -2.24003 | 0.0442741 |
| primary | active | 0.954652 | -7.92659 | -8.65388 |
| primary | all | 0.473621 | -3.46045 | -3.7746 |
| short_transfer | active | 1.64397 | 0 | -8.96192 |
| short_transfer | all | 0.383598 | 0 | -12.1468 |

Все реальные записи уже участвовали в исследовании. Новый этап фиксирует архитектуры и выбор checkpoints до evaluation/simulation, но не создаёт новый независимый test. Результат ретроспективный; нет аппаратного подтверждения и основания для заявления SOTA. Существующий LinearMPC не использует этот forecaster: sway, tracking, effort, ограничения и стоимость замкнутого управления здесь не измерены.

Аудит перехода от прогнозов к реальному оптимизатору описан отдельно в [CONTROL_INTEGRATION_AUDIT](CONTROL_INTEGRATION_AUDIT_20261107.md).

`previous_local` — победители предыдущего этапа: phase-space fine-tune в primary, LPV actor ridge .01 в short-transfer, incumbent в long-transfer. Это разное устройство моделей между folds. `selected` и `common` — псевдонимы замороженного выбора, а не дополнительные fit. Любой более удачный evaluation-only кандидат в таблицах ниже остаётся исследовательской гипотезой и не заменяет сохранённый выбор.

## Что проверяли

Четыре adapter: ширина8/16 с ограниченной силой через осциллятор; ширина8 с прямой поправкой угла; force8 с фазовым surrogate. База soft заморожена; новый выход инициализирован нулём, параметры базы сохранены в каждом checkpoint. Обучается321 или641 новых параметров. Для силы bound .05 rad и zeta .05; для прямого углового выхода bound .02 rad. Поэтому direct/force — архитектурная ablation с разным физическим смыслом и масштабом ограничения.

Четыре latent-модели: самостоятельные размерности6/12, добавочная размерности6 и её phase-space вариант. Глобальный генератор задан как кососимметричная часть минус положительная диагональ затухания; переход использует преобразование Cayley с шагом omega(L)×.05. Это обучаемая модель вход–выход, не доказанное Koopman-представление. Устойчивость относится только к автономному скрытому переходу. Она не доказывает устойчивость базовой сети, крана или контроллера и не ограничивает общее усиление форсирующих/выходных матриц.

Вдохновение и ограничения переноса разобраны в [обзоре первоисточников](STRUCTURED_MODELS_LITERATURE_20261107.md); исходный [протокол](STRUCTURED_MODELS_DESIGN_20261107.md) зафиксирован до обучения. Добавление отдельной сети к замороженной модели связано с [KNODE online, §3.2–4](https://proceedings.mlr.press/v205/jiahao23a/jiahao23a.pdf), но здесь используется offline TRAIN, без онлайн-очереди и управления роботом. Условия связи скрытой и физической устойчивости обсуждаются в [Fan et al., §4–5](https://arxiv.org/html/2401.08153v2); у нашего идентификационного блока эти условия не установлены.

Входы:41 наблюдение при100Гц, текущая длина и30 известных будущих команд при20Гц. Future measured theta/x используются только как метки. Актор и TRAIN-калибровка длины неизменны. Потеря — SmoothL1((prediction−target)/.1,beta=.1) на30 шагах; phase добавляет .25 ошибки нормированных приращений и .25 ошибки конечного угла. Это surrogate, не фактическая угловая скорость или замкнутый cost. Adapter имеет дополнительный штраф .01 квадрата собственного bounded head; latent — без него. Семейства отличаются архитектурой и регуляризацией.

Headline: evaluation-сценарии с длиной вне TRAIN-диапазона. Равные веса целых сценариев и горизонтов 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5с; interpolation и нагрузка отдельно. Перекрывающиеся окна не считаются независимыми повторениями.

## Все методы: экстраполяция, MAE в градусах

| model | long_transfer / active | long_transfer / all | primary / active | primary / all | short_transfer / active | short_transfer / all |
| --- | --- | --- | --- | --- | --- | --- |
| adapter_direct8 | 0.85034 | 0.504717 | 0.954652 | 0.473621 | 1.77908 | 0.43754 |
| adapter_force16 | 0.887992 | 0.498276 | 1.03696 | 0.493674 | 1.80485 | 0.43696 |
| adapter_force8 | 0.892012 | 0.499937 | 1.03626 | 0.491805 | 1.80506 | 0.436982 |
| adapter_force8_phase | 0.890566 | 0.500193 | 1.03746 | 0.494011 | 1.80333 | 0.437828 |
| common | 0.931853 | 0.512641 | 1.03684 | 0.490598 | 1.64397 | 0.383598 |
| incumbent | 0.931853 | 0.512641 | 1.06854 | 0.500498 | 1.81124 | 0.43213 |
| latent_residual6 | 0.901386 | 0.501157 | 0.990663 | 0.487593 | 1.78984 | 0.4287 |
| latent_residual6_phase | 0.824994 | 0.497374 | 1.02101 | 0.490018 | 1.7941 | 0.431342 |
| latent_standalone12 | 1.00016 | 0.525698 | 1.0565 | 0.539566 | 1.74516 | 0.441854 |
| latent_standalone6 | 1.00214 | 0.540607 | 1.09921 | 0.53817 | 1.7685 | 0.434851 |
| physical | 1.25281 | 0.656627 | 1.26491 | 0.554201 | 1.92444 | 0.462755 |
| previous_local | 0.931853 | 0.512641 | 1.03684 | 0.490598 | 1.64397 | 0.383598 |
| selected | 0.901386 | 0.501157 | 0.954652 | 0.473621 | 1.64397 | 0.383598 |
| soft_symmetry | 0.892822 | 0.500935 | 1.04509 | 0.492199 | 1.8058 | 0.436635 |

Тихий режим, те же extrapolation-сценарии:

| model | long_transfer | primary | short_transfer |
| --- | --- | --- | --- |
| adapter_direct8 | 0.49856 | 0.45052 | 0.382207 |
| adapter_force16 | 0.49202 | 0.467673 | 0.380657 |
| adapter_force8 | 0.49364 | 0.465736 | 0.380669 |
| adapter_force8_phase | 0.493906 | 0.467995 | 0.381644 |
| common | 0.506077 | 0.464426 | 0.33057 |
| incumbent | 0.506077 | 0.473318 | 0.375439 |
| latent_residual6 | 0.494637 | 0.463393 | 0.372601 |
| latent_residual6_phase | 0.492132 | 0.464549 | 0.375255 |
| latent_standalone12 | 0.518133 | 0.514807 | 0.387892 |
| latent_standalone6 | 0.532784 | 0.511457 | 0.379606 |
| physical | 0.647517 | 0.520709 | 0.402528 |
| previous_local | 0.506077 | 0.464426 | 0.33057 |
| selected | 0.494637 | 0.45052 | 0.33057 |
| soft_symmetry | 0.494582 | 0.465714 | 0.380266 |

## Нагрузка в основном разбиении

| model | active | all | quiet |
| --- | --- | --- | --- |
| adapter_direct8 | 1.05139 | 0.559697 | 0.549882 |
| adapter_force16 | 1.26811 | 0.553028 | 0.539413 |
| adapter_force8 | 1.27048 | 0.552275 | 0.538587 |
| adapter_force8_phase | 1.2683 | 0.550492 | 0.536816 |
| common | 1.27684 | 0.548625 | 0.534905 |
| incumbent | 1.31801 | 0.562599 | 0.548329 |
| latent_residual6 | 1.007 | 0.546738 | 0.537465 |
| latent_residual6_phase | 1.14994 | 0.550799 | 0.539164 |
| latent_standalone12 | 0.901735 | 0.547207 | 0.539697 |
| latent_standalone6 | 1.03 | 0.544748 | 0.534902 |
| physical | 1.55809 | 0.592218 | 0.574092 |
| previous_local | 1.27684 | 0.548625 | 0.534905 |
| selected | 1.05139 | 0.559697 | 0.549882 |
| soft_symmetry | 1.30433 | 0.553962 | 0.539809 |

## Выбранные прогнозы по отдельным extrapolation-записям

| fold | scenario | regime | mae_deg | previous_local_mae_deg | change_vs_previous_local_percent |
| --- | --- | --- | --- | --- | --- |
| long_transfer | 6 | active | 1.32307 | 1.39476 | -5.13938 |
| long_transfer | 6 | all | 0.420147 | 0.420506 | -0.0853499 |
| long_transfer | 7 | active | 0.734836 | 0.773068 | -4.94556 |
| long_transfer | 7 | all | 0.183494 | 0.20555 | -10.7305 |
| long_transfer | 8 | active | 0.646249 | 0.627735 | 2.9493 |
| long_transfer | 8 | all | 0.899831 | 0.911865 | -1.31978 |
| primary | 1 | active | 1.70079 | 1.77972 | -4.43492 |
| primary | 1 | all | 0.434945 | 0.448292 | -2.97733 |
| primary | 2 | active | 1.06903 | 1.20796 | -11.5011 |
| primary | 2 | all | 0.438252 | 0.456801 | -4.06071 |
| primary | 7 | active | 0.591805 | 0.633107 | -6.52366 |
| primary | 7 | all | 0.163849 | 0.17721 | -7.53993 |
| primary | 8 | active | 0.45698 | 0.526564 | -13.2147 |
| primary | 8 | all | 0.857438 | 0.880088 | -2.57357 |
| short_transfer | 1 | active | 1.98397 | 1.98397 | 0 |
| short_transfer | 1 | all | 0.359473 | 0.359473 | 0 |
| short_transfer | 2 | active | 1.29554 | 1.29554 | 0 |
| short_transfer | 2 | all | 0.356185 | 0.356185 | 0 |
| short_transfer | 3 | active | 1.65239 | 1.65239 | 0 |
| short_transfer | 3 | all | 0.435135 | 0.435135 | 0 |

Все методы и все режимы каждого сценария сохранены в `real_record_means.csv`; горизонты — `real_horizons.csv`, полные ячейки — `real_metrics_all.csv`. Графики `all_methods_extrapolation.png` и `selected_horizons.png` показывают общую MAE и зависимость от горизонта. Validation и interpolation присутствуют в отдельных summary-файлах и не смешиваются с headline.

## Предварительно заданные критерии принятия

| rule | resolved_model | scope | fold | regime | mae_deg | reference | reference_mae_deg | ratio | maximum_ratio | passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selected | adapter_direct8 | evaluation_extrapolation | primary | all | 0.473621 | previous_local | 0.490598 | 0.965395 | 1 | True |
| selected | adapter_direct8 | evaluation_extrapolation | primary | active | 0.954652 | previous_local | 1.03684 | 0.920734 | 1 | True |
| selected | previous_local | evaluation_extrapolation | short_transfer | all | 0.383598 | previous_local | 0.383598 | 1 | 1.01 | True |
| selected | previous_local | evaluation_extrapolation | short_transfer | active | 1.64397 | previous_local | 1.64397 | 1 | 1.01 | True |
| selected | latent_residual6 | evaluation_extrapolation | long_transfer | all | 0.501157 | previous_local | 0.512641 | 0.9776 | 1.01 | True |
| selected | latent_residual6 | evaluation_extrapolation | long_transfer | active | 0.901386 | previous_local | 0.931853 | 0.967305 | 1.01 | True |
| selected | adapter_direct8 | load_stress | primary | all | 0.559697 | previous_local | 0.548625 | 1.02018 | 1.01 | False |
| selected | adapter_direct8 | load_stress | primary | active | 1.05139 | previous_local | 1.27684 | 0.82343 | 1.01 | True |
| common | previous_local | evaluation_extrapolation | primary | all | 0.490598 | previous_local | 0.490598 | 1 | 1 | True |
| common | previous_local | evaluation_extrapolation | primary | active | 1.03684 | previous_local | 1.03684 | 1 | 1 | True |
| common | previous_local | evaluation_extrapolation | short_transfer | all | 0.383598 | previous_local | 0.383598 | 1 | 1.01 | True |
| common | previous_local | evaluation_extrapolation | short_transfer | active | 1.64397 | previous_local | 1.64397 | 1 | 1.01 | True |
| common | previous_local | evaluation_extrapolation | long_transfer | all | 0.512641 | previous_local | 0.512641 | 1 | 1.01 | True |
| common | previous_local | evaluation_extrapolation | long_transfer | active | 0.931853 | previous_local | 0.931853 | 1 | 1.01 | True |
| common | previous_local | load_stress | primary | all | 0.548625 | previous_local | 0.548625 | 1 | 1.01 | True |
| common | previous_local | load_stress | primary | active | 1.27684 | previous_local | 1.27684 | 1 | 1.01 | True |

И локальный, и общий выбор сравниваются с последним `previous_local`: primary all/active без ухудшения; short, long и primary load с допуском1%. Это практические пороги, не статистические границы. Они не используются для повторного выбора по evaluation. Сохранены все отрицательные результаты, неудачные обновления и fallback.

## Фактическое обучение

| model | fits | epoch_minus_one | unchanged_soft_fallbacks | selected_epoch_min | selected_epoch_median | selected_epoch_max | epochs_executed | trainable_min | trainable_max | frozen_parameters | optimizer_steps | backward_passes | fit_seconds | runner_elapsed_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adapter_direct8 | 9 | 1 | 1 | -1 | 96 | 99 | 739 | 321 | 321 | 8821 | 3695 | 3695 | 9.66894 | 10.7539 |
| adapter_force16 | 9 | 3 | 3 | -1 | 12 | 35 | 308 | 641 | 641 | 8821 | 1540 | 1540 | 9.70988 | 10.7094 |
| adapter_force8 | 9 | 4 | 4 | -1 | 0 | 51 | 315 | 321 | 321 | 8821 | 1575 | 1575 | 9.02555 | 11.2198 |
| adapter_force8_phase | 9 | 1 | 1 | -1 | 5 | 39 | 315 | 321 | 321 | 8821 | 1575 | 1575 | 10.4855 | 11.6247 |
| latent_residual6 | 9 | 0 | 0 | 10 | 61 | 99 | 696 | 4290 | 4290 | 8821 | 3480 | 3480 | 70.6181 | 71.2195 |
| latent_residual6_phase | 9 | 0 | 0 | 9 | 17 | 97 | 475 | 4290 | 4290 | 8821 | 2375 | 2375 | 48.8082 | 49.4078 |
| latent_standalone12 | 9 | 0 | 0 | 45 | 97 | 99 | 812 | 4620 | 4620 | 0 | 4060 | 4060 | 38.3296 | 38.4756 |
| latent_standalone6 | 9 | 0 | 0 | 47 | 98 | 99 | 812 | 4290 | 4290 | 0 | 4060 | 4060 | 21.593 | 21.7221 |

Всего 4472 эпох, 22360 optimizer updates, 22360 backward; суммарный fit 218.239с. AdamW lr=0.001, decay=0.0001, максимум100эпох, patience20, batch256, gradient clip2. Adapter кеширует замороженные признаки; latent обновляет динамику. Одинаковый предел эпох не означает равные FLOPs. `fit_seconds` включает внутреннюю подготовку и validation; `runner_elapsed_seconds` дополнительно содержит сохранение/перезагрузку и проверку.

Полные fit и selected epochs — `fits_flat.csv`; обучаемые/замороженные количества приведены отдельно. Для standalone эпоха−1 — собственная физически ориентированная инициализация, для residual — точный soft; эти fallback нельзя объединять в утверждение «сохранены прежние прогнозы».

## Разброс seed

| fold | model | seed_mae_mean_deg | seed_mae_std_deg | seed_mae_min_deg | seed_mae_max_deg | ensemble_mae_deg |
| --- | --- | --- | --- | --- | --- | --- |
| long_transfer | adapter_direct8 | 0.515061 | 0.0207959 | 0.492852 | 0.534073 | 0.504717 |
| long_transfer | adapter_force16 | 0.503681 | 0.0180373 | 0.491047 | 0.524338 | 0.498276 |
| long_transfer | adapter_force8 | 0.505484 | 0.0173734 | 0.490122 | 0.524338 | 0.499937 |
| long_transfer | adapter_force8_phase | 0.505496 | 0.0171564 | 0.490776 | 0.524338 | 0.500193 |
| long_transfer | latent_residual6 | 0.505208 | 0.00735988 | 0.497511 | 0.512176 | 0.501157 |
| long_transfer | latent_residual6_phase | 0.511214 | 0.0121021 | 0.49806 | 0.521875 | 0.497374 |
| long_transfer | latent_standalone12 | 0.526658 | 0.00268252 | 0.524955 | 0.52975 | 0.525698 |
| long_transfer | latent_standalone6 | 0.541839 | 0.00252863 | 0.538931 | 0.543523 | 0.540607 |
| primary | adapter_direct8 | 0.479929 | 0.0163188 | 0.464427 | 0.496957 | 0.473621 |
| primary | adapter_force16 | 0.49846 | 0.0173446 | 0.480613 | 0.515254 | 0.493674 |
| primary | adapter_force8 | 0.496661 | 0.01654 | 0.479742 | 0.512794 | 0.491805 |
| primary | adapter_force8_phase | 0.498593 | 0.0156206 | 0.483473 | 0.51467 | 0.494011 |
| primary | latent_residual6 | 0.493723 | 0.0170591 | 0.47603 | 0.510069 | 0.487593 |
| primary | latent_residual6_phase | 0.496341 | 0.0179791 | 0.476863 | 0.5123 | 0.490018 |
| primary | latent_standalone12 | 0.541473 | 0.00290438 | 0.539656 | 0.544823 | 0.539566 |
| primary | latent_standalone6 | 0.539254 | 0.0037714 | 0.534915 | 0.541747 | 0.53817 |
| short_transfer | adapter_direct8 | 0.443144 | 0.00666665 | 0.438085 | 0.450698 | 0.43754 |
| short_transfer | adapter_force16 | 0.441171 | 0.00269698 | 0.438079 | 0.443038 | 0.43696 |
| short_transfer | adapter_force8 | 0.441208 | 0.00263309 | 0.43819 | 0.443038 | 0.436982 |
| short_transfer | adapter_force8_phase | 0.442308 | 0.00388427 | 0.437823 | 0.444624 | 0.437828 |
| short_transfer | latent_residual6 | 0.434499 | 0.00479695 | 0.430815 | 0.439923 | 0.4287 |
| short_transfer | latent_residual6_phase | 0.436283 | 0.00832277 | 0.426804 | 0.442394 | 0.431342 |
| short_transfer | latent_standalone12 | 0.445746 | 0.00202771 | 0.443806 | 0.447852 | 0.441854 |
| short_transfer | latent_standalone6 | 0.436665 | 0.00445488 | 0.431776 | 0.440494 | 0.434851 |

Показаны mean/std/min/max MAE отдельных seed и отдельная MAE ансамбля. Последняя сначала усредняет прогнозы, поэтому не равна средней seed MAE. Для добавочных моделей seeds включают разные frozen bases; у standalone это инициализации новой модели. Три seed не дают доверительный интервал переноса на неизвестную длину. Контролям без отдельных сохранённых seed-массивов не приписывается нулевой разброс.

## Крупные ошибки, поздний горизонт и приращения

| scope | fold | model | regime | mean_packet_peak_deg | mean_of_record_packet_mae_p95_deg | tail_mae_deg | normalized_increment_mae_deg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| evaluation_extrapolation | long_transfer | common | all | 1.32584 | 1.87441 | 0.686145 | 0.968525 |
| evaluation_extrapolation | long_transfer | previous_local | all | 1.32584 | 1.87441 | 0.686145 | 0.968525 |
| evaluation_extrapolation | long_transfer | selected | all | 1.31694 | 1.85955 | 0.724846 | 0.986107 |
| evaluation_extrapolation | long_transfer | soft_symmetry | all | 1.29127 | 1.84484 | 0.626412 | 0.978246 |
| evaluation_extrapolation | primary | common | all | 1.29611 | 1.82885 | 0.661027 | 0.916922 |
| evaluation_extrapolation | primary | previous_local | all | 1.29611 | 1.82885 | 0.661027 | 0.916922 |
| evaluation_extrapolation | primary | selected | all | 1.25874 | 1.82068 | 0.644087 | 0.902231 |
| evaluation_extrapolation | primary | soft_symmetry | all | 1.30097 | 1.83485 | 0.666947 | 0.921198 |
| evaluation_extrapolation | short_transfer | common | all | 1.21985 | 1.31657 | 0.698498 | 0.806816 |
| evaluation_extrapolation | short_transfer | previous_local | all | 1.21985 | 1.31657 | 0.698498 | 0.806816 |
| evaluation_extrapolation | short_transfer | selected | all | 1.21985 | 1.31657 | 0.698498 | 0.806816 |
| evaluation_extrapolation | short_transfer | soft_symmetry | all | 1.27538 | 1.45104 | 0.703849 | 0.805125 |

`mean_of_record_packet_mae_p95_deg` — среднее отдельных p95 записей, **не pooled p95** окон. Peak — средний максимум абсолютной ошибки30 шагов пакета; tail — MAE последних10шагов; increment — ошибка приращения, делённая на omega×.05. Это диагностики прогноза, не нарушения физических ограничений. Все методы/режимы сохранены в `risk_summary.csv` и `risk_record_metrics.csv`.

## Новые точки симуляции

| model | long_transfer / exact | long_transfer / noisy_quantized | primary / exact | primary / noisy_quantized | short_transfer / exact | short_transfer / noisy_quantized |
| --- | --- | --- | --- | --- | --- | --- |
| adapter_direct8 | 1.3379 | 1.33772 | 1.28775 | 1.2879 | 1.22419 | 1.22372 |
| adapter_force16 | 1.29963 | 1.29934 | 1.25565 | 1.25555 | 1.22584 | 1.22555 |
| adapter_force8 | 1.30087 | 1.30059 | 1.25423 | 1.25414 | 1.2258 | 1.2255 |
| adapter_force8_phase | 1.30091 | 1.30062 | 1.25491 | 1.25481 | 1.226 | 1.22569 |
| common | 1.3792 | 1.37947 | 1.25773 | 1.25768 | 1.3107 | 1.3101 |
| incumbent | 1.3792 | 1.37947 | 1.25873 | 1.25883 | 1.23602 | 1.23587 |
| latent_residual6 | 1.33354 | 1.33313 | 1.24166 | 1.24169 | 1.21616 | 1.21565 |
| latent_residual6_phase | 1.29545 | 1.29494 | 1.25008 | 1.24996 | 1.2241 | 1.22389 |
| latent_standalone12 | 1.4463 | 1.44684 | 1.25848 | 1.25895 | 1.22272 | 1.22227 |
| latent_standalone6 | 1.43949 | 1.44018 | 1.24747 | 1.24771 | 1.22237 | 1.22177 |
| physical | 1.25203 | 1.25082 | 1.25314 | 1.25247 | 1.25536 | 1.25466 |
| previous_local | 1.3792 | 1.37947 | 1.25773 | 1.25768 | 1.3107 | 1.3101 |
| selected | 1.33354 | 1.33313 | 1.28775 | 1.2879 | 1.3107 | 1.3101 |
| soft_symmetry | 1.30208 | 1.30181 | 1.2499 | 1.24992 | 1.22584 | 1.22555 |

Число случаев по folds: {"long_transfer": 1152, "primary": 1152, "short_transfer": 1152}, всего3456. Все 12 методов и два псевдонима получают одно заданное ускорение; актор обойдён только для этой угловой проверки. `simulation_groups.csv` раздельно показывает force-law, drive, future-gain и measurement, включая p95/max ошибок случаев; дополнительные файлы разлагают частоту, затухание, длину и горизонт. Это новые точки прежнего семейства симулятора, без независимого аппаратного испытания.

## Replay, заморозка и сохранность

Полный replay из сохранённых весов: 678 сравнений (246 ансамблевых/контрольных и432 отдельных seed) на17496 пакетах из18 записей. Максимальное расхождение0°. Входы заново собраны только из истории и известного плана; метки и средние ансамблей проверены отдельно.

Существующие и новые unit tests: 142 PASS; лог `structured_models_tests.log`. Проверены72 сохранённых state_dict. Виды проверок: `{"direct_bounded_head": 9, "fixed_damped_oscillator": 27, "latent_cayley": 36}`. Learned Cayley, fixed damped filter и bounded direct head — разные проверки; у direct head нет рекуррентного оператора. Сводка аудита: `{"passed": true, "models_checked": 72, "warm_start_provenance_sha256": "baa7bf0c01df8d01505a32af63f04ce868013d3144fb7011a3dbfebcb65914ef", "initialization_rule": "torch.manual_seed(seed), then exact make_structured_model factory", "comparison": "torch.equal; drift in float64; buffers audited separately from parameter counts", "frozen_tensor_checks": 1332, "fallback_tensor_checks": 234, "buffer_checks": 144, "stability_checks": 72, "initial_checkpoint_selections": 9, "unchanged_models": 9, "latent_homogeneous_contraction_checks": 36, "max_total_l2_delta": 8.8254704922457}`. Полные матричные показатели и проверки замороженных bases/epoch−1 находятся в `replay_verification.json` и `parameter_and_stability_audit.csv`; численный аудит дополняет структурное доказательство и не является сертификатом управления. Исходный snapshot основной версии и APHYNITY: 1281 совпадающих файлов. Этапные SHA/научные проверки: `{"guarded_hashes": 117, "checkpoint_hashes": 72, "result_hashes": 108, "source_run_checkpoints": 37, "broad_run_checkpoints": 82, "parent_run_checkpoints": 90, "train_checks": 87, "evaluate_checks": 81}`.

Runner также проверил reload, родительские прогнозы, подмену targets, независимую сборку history-only, сходимость интегратора и инвариантность к скрытому будущему gain. Проверки не используют evaluation для градиентов или выбора модели.

## Задержка прогнозирования

| fold | model | measurements | median_ms | p95_ms | max_ms | fraction_over_50ms |
| --- | --- | --- | --- | --- | --- | --- |
| long_transfer | adapter_direct8 | 30 | 17.5092 | 20.894 | 21.598 | 0 |
| long_transfer | adapter_force16 | 30 | 19.1151 | 23.4248 | 32.8492 | 0 |
| long_transfer | adapter_force8 | 30 | 19.3029 | 22.8803 | 25.99 | 0 |
| long_transfer | adapter_force8_phase | 30 | 18.8229 | 22.7383 | 22.9885 | 0 |
| long_transfer | latent_residual6 | 30 | 9.97846 | 12.0988 | 13.9745 | 0 |
| long_transfer | latent_residual6_phase | 30 | 9.97146 | 11.9852 | 13.5501 | 0 |
| long_transfer | latent_standalone12 | 30 | 2.15894 | 2.55701 | 2.64242 | 0 |
| long_transfer | latent_standalone6 | 30 | 2.07915 | 2.60032 | 2.77667 | 0 |
| long_transfer | soft_symmetry | 30 | 9.37515 | 10.968 | 12.8008 | 0 |
| primary | adapter_direct8 | 30 | 20.5721 | 22.2208 | 22.3953 | 0 |
| primary | adapter_force16 | 30 | 22.437 | 24.8606 | 30.4551 | 0 |
| primary | adapter_force8 | 30 | 22.7715 | 24.27 | 24.3169 | 0 |
| primary | adapter_force8_phase | 30 | 22.4272 | 24.6549 | 26.5268 | 0 |
| primary | latent_residual6 | 30 | 11.7983 | 13.1367 | 14.2132 | 0 |
| primary | latent_residual6_phase | 30 | 11.8599 | 12.9123 | 13.2721 | 0 |
| primary | latent_standalone12 | 30 | 2.45256 | 2.93904 | 3.30525 | 0 |
| primary | latent_standalone6 | 30 | 2.52444 | 2.82125 | 3.43546 | 0 |
| primary | soft_symmetry | 30 | 10.7896 | 11.9884 | 15.8712 | 0 |
| short_transfer | adapter_direct8 | 30 | 19.5376 | 20.5692 | 21.5147 | 0 |
| short_transfer | adapter_force16 | 30 | 21.0361 | 21.8173 | 22.1507 | 0 |
| short_transfer | adapter_force8 | 30 | 21.0808 | 22.1166 | 23.837 | 0 |
| short_transfer | adapter_force8_phase | 30 | 21.0722 | 22.5846 | 22.6893 | 0 |
| short_transfer | latent_residual6 | 30 | 11.277 | 11.7403 | 11.8973 | 0 |
| short_transfer | latent_residual6_phase | 30 | 11.1988 | 11.9726 | 12.5125 | 0 |
| short_transfer | latent_standalone12 | 30 | 2.35094 | 2.51276 | 2.55437 | 0 |
| short_transfer | latent_standalone6 | 30 | 2.34863 | 2.55615 | 3.04233 | 0 |
| short_transfer | soft_symmetry | 30 | 10.2559 | 11.0934 | 11.4458 | 0 |

Прогретый CPU, batch=1, актор + observer + ансамбль трёх угловых моделей. Измерение не включает загрузку весов, sensor I/O и оптимизацию MPC. Сравнение с 50 мс описывает этот запуск и не является гарантией real-time или WCET.

Run: `artifacts/runs/structured_models_20261107`. Результаты: `results/structured_models_20261107`. Архив создаётся отдельно после отчёта с SHA/CRC; полный replay требует сохранённых upstream данных и весов. Публикация в GitHub не выполняется генератором отчёта или архиватором и проверяется отдельно от научных результатов.

Переносимый комплект прогнозирования описан в [RESEARCH_BUNDLE](RESEARCH_BUNDLE_20261107.md). Он разделяет сохранённый default и новые исследовательские candidates; экспорт не меняет результаты gate и не объявляет выбранный candidate принятым для эксплуатации.
