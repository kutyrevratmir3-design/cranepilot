# Fine-tuning и критерии из управления — 7 ноября 2026

Завершены 90 дообучений: 10 стратегий × 3 разбиения × 3 сохранённых seed. Общий выбор по validation — `previous_local`; заранее заданный критерий принятия **не выполнен**. Локальные решения: primary: `phase_space`, short_transfer: `previous_local`, long_transfer: `incumbent`. В 59 из 90 fit выбран исходный checkpoint эпохи −1; это отсутствие принятого обновления, а не улучшение от fine-tuning.

## Выбранные локальные решения

| fold | regime | mae_deg | change_vs_incumbent_percent | change_vs_soft_symmetry_percent | change_vs_previous_local_percent |
| --- | --- | --- | --- | --- | --- |
| long_transfer | active | 0.931853 | 0 | 4.37164 | 0 |
| long_transfer | all | 0.512641 | 0 | 2.33665 | 0 |
| primary | active | 1.03684 | -2.96717 | -0.789902 | -0.84924 |
| primary | all | 0.490598 | -1.97802 | -0.325405 | -0.339266 |
| short_transfer | active | 1.64397 | -9.23508 | -8.96192 | 0 |
| short_transfer | all | 0.383598 | -11.2311 | -12.1468 | 0 |

Это ретроспективное исследование ранее использованных реальных записей. Стратегии и checkpoints выбраны по validation до evaluation и новой симуляции; нет нового независимого аппаратного теста или основания для SOTA. `selected`/`common` являются псевдонимами уже выбранных моделей. `previous_local` сохраняет прошлые решения, включая LPV на коротких тросах; одна эта строка не означает одинаковую архитектуру между разбиениями.

## Что менялось

Все стратегии начинают с соответствующего soft-symmetry checkpoint, архитектура и актор сохраняются. Проверены обычное продолжение; только observer; только innovation; L2-SP к исходным параметрам; привязка выходов к исходной модели; SAM; EMA; фазовый/конечный surrogate; minibatch group-DRO; L-BFGS polish. Предыдущие данные, веса и отдельный APHYNITY не смешиваются с обучением этой серии.

История — 41 отсчёт, 30 шагов до 1.5 с. Метрики используют горизонты 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5 с. Усреднение: равные веса целых сценариев на каждом горизонте, затем равные веса горизонтов. Headline — evaluation/extrapolation; interpolation и нагрузка записаны отдельно.

Литературная основа: [методы fine-tuning](FINETUNING_METHODS_RESEARCH_20261107.md) и [выбор методов в управлении](CONTROL_METHOD_SELECTION_RESEARCH_20261107.md). [DiffTune-MPC](https://arxiv.org/html/2312.11384v2) различает внутреннюю стоимость планирования и оценку замкнутой траектории. [Sakib–Pan, раздел V](https://arxiv.org/html/2408.06607v6) отдельно проверяет optimizer switching, расписание и rollout; эти результаты служат мотивацией эксперимента, не обещанием переноса эффекта на CranePilot.

Дополнительный обзор постановки задачи и регуляризации: [данные для управления, DeePC и W-PGNN](CONTROL_DATA_AND_REGULARIZATION_20261107.md).

Существующий LinearMPC не использует этот нейронный forecaster. Здесь не измерялись settling time, управленческое усилие или замкнутый cost. Phase-space loss — ошибка нормированных приращений и конечного угла, а не точное измерение скорости и не градиент реального управления. Квантизация меток может влиять на такой loss.

## Все методы: MAE на экстраполяции, градусы

| model | long_transfer / active | long_transfer / all | primary / active | primary / all | short_transfer / active | short_transfer / all |
| --- | --- | --- | --- | --- | --- | --- |
| combined | 0.89419 | 0.511508 | 1.04572 | 0.492268 | 1.80563 | 0.43618 |
| common | 0.931853 | 0.512641 | 1.04572 | 0.492268 | 1.64397 | 0.383598 |
| continued | 0.892822 | 0.500935 | 1.04507 | 0.491737 | 1.78874 | 0.439642 |
| ema | 0.892464 | 0.500917 | 1.04514 | 0.492314 | 1.8058 | 0.436635 |
| function_anchor | 0.892822 | 0.500935 | 1.04499 | 0.491649 | 1.78902 | 0.44131 |
| group_dro | 0.892822 | 0.500935 | 1.04509 | 0.492199 | 1.80011 | 0.443455 |
| incumbent | 0.931853 | 0.512641 | 1.06854 | 0.500498 | 1.81124 | 0.43213 |
| innovation_only | 0.892822 | 0.500935 | 1.02327 | 0.488774 | 1.7981 | 0.435376 |
| l2sp | 0.892822 | 0.500935 | 1.04499 | 0.491742 | 1.79735 | 0.438129 |
| lbfgs_polish | 0.892822 | 0.500935 | 1.04509 | 0.492199 | 1.79899 | 0.438023 |
| observer_only | 0.892822 | 0.500935 | 1.04509 | 0.492199 | 1.8058 | 0.436635 |
| phase_space | 0.892822 | 0.500935 | 1.03684 | 0.490598 | 1.8058 | 0.436635 |
| physical | 1.25281 | 0.656627 | 1.26491 | 0.554201 | 1.92444 | 0.462755 |
| previous_local | 0.931853 | 0.512641 | 1.04572 | 0.492268 | 1.64397 | 0.383598 |
| sam | 0.888767 | 0.499993 | 1.04085 | 0.492692 | 1.7738 | 0.433558 |
| selected | 0.931853 | 0.512641 | 1.03684 | 0.490598 | 1.64397 | 0.383598 |
| soft_symmetry | 0.892822 | 0.500935 | 1.04509 | 0.492199 | 1.8058 | 0.436635 |

## Нагрузка в основном разбиении

| model | active | all | quiet |
| --- | --- | --- | --- |
| combined | 1.30461 | 0.568001 | 0.55403 |
| common | 1.30461 | 0.568001 | 0.55403 |
| continued | 1.30511 | 0.555684 | 0.541533 |
| ema | 1.30387 | 0.554074 | 0.539932 |
| function_anchor | 1.30537 | 0.555502 | 0.541342 |
| group_dro | 1.30433 | 0.553962 | 0.539809 |
| incumbent | 1.31801 | 0.562599 | 0.548329 |
| innovation_only | 1.2786 | 0.552884 | 0.539147 |
| l2sp | 1.3048 | 0.555624 | 0.54148 |
| lbfgs_polish | 1.30433 | 0.553962 | 0.539809 |
| observer_only | 1.30433 | 0.553962 | 0.539809 |
| phase_space | 1.27684 | 0.548625 | 0.534905 |
| physical | 1.55809 | 0.592218 | 0.574092 |
| previous_local | 1.30461 | 0.568001 | 0.55403 |
| sam | 1.28919 | 0.551313 | 0.537378 |
| selected | 1.27684 | 0.548625 | 0.534905 |
| soft_symmetry | 1.30433 | 0.553962 | 0.539809 |

## Сохранённый критерий принятия общей стратегии

| scope | fold | regime | reference | ratio | maximum_ratio | passed |
| --- | --- | --- | --- | --- | --- | --- |
| evaluation_extrapolation | primary | all | soft_symmetry | 1.00014 | 1 | False |
| evaluation_extrapolation | primary | active | soft_symmetry | 1.0006 | 1 | False |
| evaluation_extrapolation | short_transfer | all | previous_local | 1 | 1.01 | True |
| evaluation_extrapolation | short_transfer | active | previous_local | 1 | 1.01 | True |
| evaluation_extrapolation | long_transfer | all | incumbent | 1 | 1.01 | True |
| evaluation_extrapolation | long_transfer | active | incumbent | 1 | 1.01 | True |
| load_stress | primary | all | incumbent | 1.0096 | 1.01 | True |
| load_stress | primary | active | incumbent | 0.989836 | 1.01 | True |

Primary сравнивается с soft symmetry; short-transfer должен сохранить предыдущий LPV-результат с допуском 1%; long-transfer и нагрузка сравниваются с incumbent с тем же допуском. Это практические пороги, а не статистические границы. Все отрицательные результаты и начальные checkpoints сохранены.

## Фактическое обучение и стоимость вычислений

| model | fits | epoch_minus_one | selected_epoch_min | selected_epoch_median | selected_epoch_max | epochs_executed | trainable_min | trainable_max | total_parameter_count | optimizer_steps | backward_passes | closure_evaluations | fit_seconds | runner_elapsed_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continued | 9 | 5 | -1 | -1 | 28 | 179 | 8821 | 8821 | 8821 | 895 | 895 | 895 | 42.0502 | 45.2066 |
| ema | 9 | 7 | -1 | -1 | 0 | 137 | 8821 | 8821 | 8821 | 685 | 685 | 685 | 32.2656 | 33.1575 |
| function_anchor | 9 | 5 | -1 | -1 | 28 | 187 | 8821 | 8821 | 8821 | 935 | 935 | 935 | 49.1032 | 50.4198 |
| group_dro | 9 | 8 | -1 | -1 | 0 | 136 | 8821 | 8821 | 8821 | 680 | 680 | 680 | 32.8408 | 37.7357 |
| innovation_only | 9 | 4 | -1 | 0 | 57 | 230 | 1093 | 1093 | 8821 | 1150 | 1150 | 1150 | 52.7581 | 53.5996 |
| l2sp | 9 | 5 | -1 | -1 | 15 | 157 | 8821 | 8821 | 8821 | 785 | 785 | 785 | 39.6948 | 40.6242 |
| lbfgs_polish | 9 | 8 | -1 | -1 | 9 | 100 | 8821 | 8821 | 8821 | 100 | 213 | 213 | 28.4837 | 29.2806 |
| observer_only | 9 | 9 | -1 | -1 | -1 | 135 | 231 | 231 | 8821 | 675 | 675 | 675 | 29.4714 | 30.3623 |
| phase_space | 9 | 6 | -1 | -1 | 51 | 186 | 8821 | 8821 | 8821 | 930 | 930 | 930 | 44.6847 | 45.484 |
| sam | 9 | 2 | -1 | 5 | 58 | 303 | 8821 | 8821 | 8821 | 1515 | 3030 | 3030 | 132.433 | 133.358 |

Полные selected epochs и замороженные параметры каждого fit — в `fits_flat.csv`, исходные метаданные — `fits.json`. `fit_seconds` суммируют время fit; `runner_elapsed_seconds` дополнительно включает сохранение/перезагрузку и локальную проверку. Одинаковый максимум эпох не означает одинаковый compute: SAM имеет два backward на update; L-BFGS делает переменное число closure в line search. Поле `closure_evaluations` в исходной реализации считает оценивания training closure с backward для всех оптимизаторов; для обычного AdamW это один проход. EMA выбирает только усреднённые веса. Число параметров и фактические проходы приведены, без заявления о равном FLOP-бюджете.

Дополнительный benchmark сохранён в `inference_latency.json`: прогретый CPU, batch=1, ансамбль трёх seed с актором и observer. Измеренная задержка на этом устройстве не является гарантией real-time или worst-case execution time.

## Разброс по трём исходным seed

| fold | model | seed_mae_mean_deg | seed_mae_std_deg | seed_mae_min_deg | seed_mae_max_deg | ensemble_mae_deg |
| --- | --- | --- | --- | --- | --- | --- |
| long_transfer | continued | 0.506394 | 0.0161985 | 0.492852 | 0.524338 | 0.500935 |
| long_transfer | ema | 0.506329 | 0.0162801 | 0.492657 | 0.524338 | 0.500917 |
| long_transfer | function_anchor | 0.506394 | 0.0161985 | 0.492852 | 0.524338 | 0.500935 |
| long_transfer | group_dro | 0.506394 | 0.0161985 | 0.492852 | 0.524338 | 0.500935 |
| long_transfer | innovation_only | 0.506394 | 0.0161985 | 0.492852 | 0.524338 | 0.500935 |
| long_transfer | l2sp | 0.506394 | 0.0161985 | 0.492852 | 0.524338 | 0.500935 |
| long_transfer | lbfgs_polish | 0.506394 | 0.0161985 | 0.492852 | 0.524338 | 0.500935 |
| long_transfer | observer_only | 0.506394 | 0.0161985 | 0.492852 | 0.524338 | 0.500935 |
| long_transfer | phase_space | 0.506394 | 0.0161985 | 0.492852 | 0.524338 | 0.500935 |
| long_transfer | sam | 0.506339 | 0.0162676 | 0.492687 | 0.524338 | 0.499993 |
| primary | continued | 0.497479 | 0.0180027 | 0.478673 | 0.514554 | 0.491737 |
| primary | ema | 0.498237 | 0.0166696 | 0.481236 | 0.514554 | 0.492314 |
| primary | function_anchor | 0.497319 | 0.0183086 | 0.478099 | 0.514554 | 0.491649 |
| primary | group_dro | 0.498131 | 0.016664 | 0.481236 | 0.514554 | 0.492199 |
| primary | innovation_only | 0.494486 | 0.0157561 | 0.481372 | 0.511964 | 0.488774 |
| primary | l2sp | 0.497467 | 0.0180199 | 0.47864 | 0.514554 | 0.491742 |
| primary | lbfgs_polish | 0.498131 | 0.016664 | 0.481236 | 0.514554 | 0.492199 |
| primary | observer_only | 0.498131 | 0.016664 | 0.481236 | 0.514554 | 0.492199 |
| primary | phase_space | 0.49745 | 0.0216586 | 0.478905 | 0.521253 | 0.490598 |
| primary | sam | 0.498206 | 0.0179785 | 0.480181 | 0.516137 | 0.492692 |
| short_transfer | continued | 0.446677 | 0.00847048 | 0.441203 | 0.456434 | 0.439642 |
| short_transfer | ema | 0.441078 | 0.00285595 | 0.437801 | 0.443038 | 0.436635 |
| short_transfer | function_anchor | 0.44819 | 0.00596399 | 0.442395 | 0.45431 | 0.44131 |
| short_transfer | group_dro | 0.44955 | 0.0165307 | 0.437801 | 0.468452 | 0.443455 |
| short_transfer | innovation_only | 0.440035 | 0.00318687 | 0.43641 | 0.442395 | 0.435376 |
| short_transfer | l2sp | 0.443877 | 0.00526278 | 0.437801 | 0.447015 | 0.438129 |
| short_transfer | lbfgs_polish | 0.442367 | 0.000685373 | 0.441668 | 0.443038 | 0.438023 |
| short_transfer | observer_only | 0.441078 | 0.00285595 | 0.437801 | 0.443038 | 0.436635 |
| short_transfer | phase_space | 0.441078 | 0.00285595 | 0.437801 | 0.443038 | 0.436635 |
| short_transfer | sam | 0.438333 | 0.00447806 | 0.433464 | 0.442274 | 0.433558 |

Это описательные mean/std/min/max MAE отдельных seed после равного усреднения сценариев и горизонтов. Средняя MAE seed отличается от MAE ансамбля, который сначала усредняет прогнозы. Три seed не являются независимыми аппаратными испытаниями и не дают доверительный интервал для новой длины. У пяти сохранённых контролей в этом run нет отдельных `__seeds` массивов; им не приписывается искусственный нулевой разброс.

## Диагностика крупных ошибок и фазы

| scope | fold | model | regime | mean_packet_peak_deg | mean_of_record_packet_mae_p95_deg | tail_mae_deg | normalized_increment_mae_deg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| evaluation_extrapolation | long_transfer | common | all | 1.32584 | 1.87441 | 0.686145 | 0.968525 |
| evaluation_extrapolation | long_transfer | incumbent | all | 1.32584 | 1.87441 | 0.686145 | 0.968525 |
| evaluation_extrapolation | long_transfer | previous_local | all | 1.32584 | 1.87441 | 0.686145 | 0.968525 |
| evaluation_extrapolation | long_transfer | selected | all | 1.32584 | 1.87441 | 0.686145 | 0.968525 |
| evaluation_extrapolation | long_transfer | soft_symmetry | all | 1.29127 | 1.84484 | 0.626412 | 0.978246 |
| evaluation_extrapolation | primary | common | all | 1.30928 | 1.83493 | 0.669565 | 0.921619 |
| evaluation_extrapolation | primary | incumbent | all | 1.322 | 1.86137 | 0.68574 | 0.926621 |
| evaluation_extrapolation | primary | previous_local | all | 1.30928 | 1.83493 | 0.669565 | 0.921619 |
| evaluation_extrapolation | primary | selected | all | 1.29611 | 1.82885 | 0.661027 | 0.916922 |
| evaluation_extrapolation | primary | soft_symmetry | all | 1.30097 | 1.83485 | 0.666947 | 0.921198 |
| evaluation_extrapolation | short_transfer | common | all | 1.21985 | 1.31657 | 0.698498 | 0.806816 |
| evaluation_extrapolation | short_transfer | incumbent | all | 1.23808 | 1.41567 | 0.680219 | 0.796909 |
| evaluation_extrapolation | short_transfer | previous_local | all | 1.21985 | 1.31657 | 0.698498 | 0.806816 |
| evaluation_extrapolation | short_transfer | selected | all | 1.21985 | 1.31657 | 0.698498 | 0.806816 |
| evaluation_extrapolation | short_transfer | soft_symmetry | all | 1.27538 | 1.45104 | 0.703849 | 0.805125 |

`mean_of_record_packet_mae_p95_deg` — **среднее отдельных 95-х процентилей записей**, не pooled p95 всех окон. Peak — среднее максимума абсолютной ошибки каждого 30-шагового пакета; tail — MAE последних 10 шагов; increment — MAE разности ошибки, нормированной на omega×0.05. Эти показатели относятся к ошибке прогноза, не к нарушениям физических ограничений. В `risk_summary.csv` сохранены все методы и режимы, а не только таблица выше.

## Симуляция и проверка воспроизводимости

Новая сетка содержит {"long_transfer": 1152, "primary": 1152, "short_transfer": 1152} наблюдений по разбиениям: всего 3456 случаев. Все 15 методов и два псевдонима получают одинаковое заданное ускорение; актор обойдён только в этой угловой диагностике. Сохраняются отдельные force-law/drive/gain/noise группы, p95 и максимумы ошибок случаев, длины, частоты, затухания и горизонты. Это прежнее семейство симулятора и не проверка реального контроллера.

SHA-проверки и этапные проверки: {"guarded_hashes": 107, "checkpoint_hashes": 90, "result_hashes": 126, "source_run_checkpoints": 37, "broad_run_checkpoints": 82, "parent_run_checkpoints": 48, "train_checks": 108, "evaluate_checks": 96}. Проверены исходники/данные, 90 новых checkpoints, исходные контрольные веса, полный manifest результатов и selection. Дополнительно runner проверяет перезагрузку checkpoints, контрольные прогнозы, подмену future-target, историю без targets, сходимость интегратора и скрытый future-gain. Полный replay: 840 сравнений на 17496 пакетах; максимальное расхождение 0°. Проверены все 90 наборов весов, точная заморозка параметров и возврат к эпохе −1. Исходный snapshot основной версии и APHYNITY совпадает. Подробности: `replay_verification.json`.

Run: `artifacts/runs/finetune_campaign_20261107`. Результаты: `results/finetune_campaign_20261107`. Архив этого этапа формируется отдельно после завершения отчёта; полный replay требует сохранённых upstream данных и весов. Ничего не публикуется в GitHub.
