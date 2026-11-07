# Неопределённость модели и physical Jacobian: 7 ноября 2026

Завершены36development и576evaluation эпизодов:96на каждый из6методов.
Веса не переобучались. Новый grid отличается от предыдущего control-опыта;
настройки зафиксированы до запуска в `de5d678b10013158e61f8a765d967c3a09edde98` и после development
не менялись. Все пять альтернатив публикуются, а не выбираются по evaluation.

## Основные результаты на новом наборе

| method | realized_cost | tracking_ise_m2_s | sway_ise_deg2_s | tail_rms_deg | final_position_error_m | settled | rail_violation_100hz | angle_violation_100hz | speed_violation_100hz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frequency | 76.8908 | 0.0535724 | 117.994 | 2.92551 | 0.0231377 | 0.260417 | 0 | 8 | 4 |
| hybrid | 66.9535 | 0.0521116 | 72.9074 | 2.2206 | 0.0241967 | 0.291667 | 4 | 8 | 0 |
| joint | 54.2403 | 0.0472184 | 47.7427 | 1.83907 | 0.0169431 | 0.572917 | 0 | 0 | 0 |
| joint_hybrid | 50.0653 | 0.0495449 | 46.424 | 1.74727 | 0.0179465 | 0.479167 | 0 | 0 | 0 |
| nominal | 57.4139 | 0.0464039 | 73.6801 | 2.09128 | 0.0175944 | 0.583333 | 0 | 8 | 0 |
| spread | 53.5192 | 0.0472116 | 47.0732 | 1.82627 | 0.0166604 | 0.572917 | 0 | 0 | 0 |

Процентные изменения относительно nominal physical (меньше ошибки/стоимость лучше):

| method | command_variation | effort_acceleration2 | final_position_error_m | realized_cost | sway_ise_deg2_s | tail_rms_deg | tracking_ise_m2_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| frequency | 70.743 | 109.441 | 31.5063 | 33.9237 | 60.1439 | 39.8906 | 15.4481 |
| hybrid | 10.2453 | 7.19356 | 37.5248 | 16.6155 | -1.04872 | 6.18365 | 12.3001 |
| joint | 27.4935 | 18.002 | -3.70186 | -5.52752 | -35.2028 | -12.0599 | 1.75542 |
| joint_hybrid | 19.6033 | 4.92224 | 2.00112 | -12.7993 | -36.9925 | -16.4498 | 6.76899 |
| nominal | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| spread | 27.5367 | 17.3094 | -5.30866 | -6.78352 | -36.1114 | -12.6721 | 1.74077 |

Парные выигрыши по реализованной стоимости:

| method | change_percent | paired_case_wins | cases |
| --- | --- | --- | --- |
| nominal | 0 | 0 | 96 |
| frequency | 33.9237 | 22 | 96 |
| joint | -5.52752 | 44 | 96 |
| spread | -6.78352 | 44 | 96 |
| hybrid | 16.6155 | 18 | 96 |
| joint_hybrid | -12.7993 | 40 | 96 |

Меньшая средняя стоимость может отражать улучшение нескольких тяжёлых случаев.
Поэтому среднее не заменяет отдельную проверку большинства парных выигрышей.

Фиксированные практические gates: `{"frequency": {"cost_improved": false, "sway_improved": false, "tracking_improved": false, "final_error_within_1percent": false, "no_additional_raw_violations": false, "majority_cost_wins": false, "passed": false}, "joint": {"cost_improved": true, "sway_improved": true, "tracking_improved": false, "final_error_within_1percent": true, "no_additional_raw_violations": true, "majority_cost_wins": false, "passed": false}, "spread": {"cost_improved": true, "sway_improved": true, "tracking_improved": false, "final_error_within_1percent": true, "no_additional_raw_violations": true, "majority_cost_wins": false, "passed": false}, "hybrid": {"cost_improved": false, "sway_improved": true, "tracking_improved": false, "final_error_within_1percent": false, "no_additional_raw_violations": false, "majority_cost_wins": false, "passed": false}, "joint_hybrid": {"cost_improved": true, "sway_improved": true, "tracking_improved": false, "final_error_within_1percent": false, "no_additional_raw_violations": true, "majority_cost_wins": false, "passed": false}}`.
Требовалось одновременно улучшить средние cost/sway/tracking, удержать final
error в1%, не увеличить raw100Hz violations и выиграть cost в большинстве96cases.
Это критерии этой конечной синтетической задачи, не статистический сертификат.

## Какие методы проверены

Nominal physical — общий reference. Frequency усредняет cost по частотам
.85/1/1.15 от номинальной. Joint добавляет3пары(tau,gain) и2закона forcing,
всего18сценариев. Joint одновременно меняет actuator и forcing uncertainty;
их отдельные вклады этот набор ablations не разделяет. Spread дополнительно
штрафует .25 variance угла сценариев.
Finite scenario mean не является minimax, а spread — не chance constraint.
Сетка не калибрована как распределение; часть истинных lag лежит вне неё.

Hybrid вычисляет bounded offset clip(theta_adapter(warm)−theta_physical(warm),
±.015rad) один раз на тик. Внутри solver он фиксирован, поэтому command Jacobian
задаёт физика. Joint_hybrid применяет тот же offset к18сценариям. Это новые
controller predictors, а не переименование исходного end-to-end forecaster.
Старые forecast/load gates и default portable bundle от этого не меняются.

У всех одинаковые measured history41×100Hz, causal observer, backward v/a fits,
committed command delay .05s, horizon30,6блоков, input box±.8, trust region±.2,
200FISTAsteps и fractions0/.25/.5/1. Frozen-offset features аффинны; Jacobian
восстановлен unit central probes, без small-epsilon cancellation. Nominal
контроль тоже использует этот solver, поэтому сравнивать его абсолютные числа
с прошлой сеткой96cases как прямой прогресс нельзя.

Никаких true v/rate/a/future measurements в controller. Native xr features0,
NeuralActuator bypass. Это отдельная acceleration-command angular ablation,
не полноценный native-command пилот и не hardware validation.

## Основание из исследований

[Cautious MPC Hewing et al.](https://arxiv.org/abs/1705.10702) учитывает модельную
неопределённость при планировании. Здесь нет GP/вероятностного распространения
или гарантий этой работы; использована только общая идея учитывать uncertainty.
[Mayne et al.](https://www.sciencedirect.com/science/article/abs/pii/S0005109899002149)
рассматривают условия свойств constrained MPC. Наша конечная сетка и box limit
не устанавливают выполнение условий устойчивости/инвариантности. Подробности
и ограничения были записаны в SCENARIO_CONTROL_DESIGN_20261107.md до запуска.

## Набор условий и статистические пределы

24контекста L=.29/.64/1.11 × frequency ratio .86/1.14 × tau .075/.205 × gain
.85/1.15. Parity nuisance: damping .015/.09, deadzone0/.025, effective/geometric
forcing, initialangle2.5/5.5deg. Это сбалансированные назначения, не независимый
full factorial nuisance. Каждый контекст exact/noisy×pulse absent/present.
Pulse .17rad/s² на[2.65,3)s, новый seed1007112026. Шумы и disturbances одинаковы
между controller. .205s lag выходит за проектную сетку .07/.1/.18.
Development:6условий на метод, L=.36/.56/.76×angle2/6deg, без tuning.

Mean metrics — равные веса96эпизодов; group tables и все отдельные эпизоды
сохранены. context_bootstrap.csv resamples24plant contexts целиком, включая
4sensor/disturbance cases; интервалы описывают эту выбранную семью симулятора,
не новую популяцию устройств. Все rail/angle/speed violations сохранены.
ISE на50ms grid, отдельные пики/нарушения на100Hz. Contact/clipping состояния нет.

## Время и воспроизведение

Batch runtime:

| stage | method | cases | batch_median_ms | batch_p95_ms | batch_max_ms | amortized_median_ms_per_case |
| --- | --- | --- | --- | --- | --- | --- |
| development | nominal | 6 | 5.01252 | 5.1824 | 6.35975 | 0.83542 |
| development | frequency | 6 | 5.3486 | 5.62872 | 6.12783 | 0.891434 |
| development | joint | 6 | 5.98256 | 7.17706 | 10.7558 | 0.997094 |
| development | spread | 6 | 5.69648 | 6.05851 | 6.36954 | 0.949413 |
| development | hybrid | 6 | 22.3733 | 25.8715 | 45.6091 | 3.72888 |
| development | joint_hybrid | 6 | 23.4055 | 24.576 | 25.5143 | 3.90091 |
| evaluate | nominal | 96 | 6.68773 | 7.15952 | 8.21687 | 0.0696638 |
| evaluate | frequency | 96 | 9.13729 | 9.90144 | 11.1009 | 0.0951801 |
| evaluate | joint | 96 | 21.2402 | 22.5986 | 29.7976 | 0.221252 |
| evaluate | spread | 96 | 22.8865 | 23.8448 | 25.7923 | 0.238401 |
| evaluate | hybrid | 96 | 32.8566 | 34.1926 | 35.9341 | 0.342256 |
| evaluate | joint_hybrid | 96 | 48.1155 | 49.6221 | 53.724 | 0.501204 |

Изолированное warm batch1 полное решение,15измерений/метод, milliseconds:

| method | median | max | fraction_over_50ms |
| --- | --- | --- | --- |
| frequency | 4.82667 | 4.89371 | 0 |
| hybrid | 19.3561 | 19.5279 | 0 |
| joint | 4.98442 | 5.04575 | 0 |
| joint_hybrid | 19.3856 | 19.5698 | 0 |
| nominal | 4.86392 | 5.48854 | 0 |
| spread | 5.00979 | 5.07162 | 0 |

Амортизация batch времени — throughput, не single-controller latency.
Ни один замер не является WCET/scheduling гарантией. Simulation офлайн: кроме
объявленного committed тика wall-clock solver delay на plant не действует.
Batch1/batch96 округление сохранено в runtime_batch1.csv.

157unit tests пройдены. Повторно интегрированы 612эпизодов,
воспроизведены выбранные forecasts/costs для 73440решений с точной
реконструкцией warm plan и frozen hybrid offset. Проверены все inputs/source SHA,
paired noise/disturbances, pipeline, bounds, warm-cost monotonicity, causal
estimates, половинный RK4 шаг и 1281original files.
Полный replay НЕ перезапускал optimizer. Все таблицы и trajectories сохранены,
первичные результаты/неудачи не удаляются. Existing saved weights неизменны.
