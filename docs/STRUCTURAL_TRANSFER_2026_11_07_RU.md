# Структурная ошибка привода и разворот, 7 ноября 2026

Завершены 54 development и 864 evaluation эпизода (96 на каждый из 9 методов),
146880 решений. Протокол и исполняемый код зафиксированы до запуска:
`540477f7be091293d3c58ceb80230df51c0bd00f`. Настройки после development не менялись, neural training отсутствует.
Прошедшие полный global+family gate: force_actuator, fixed_force_observer, adaptive_force, adaptive_force_observer, frozen_adaptive_force_observer.
Все альтернативы и отрицательные случаи сохранены; новый default не назначается.

## Основной вывод

Прежний adaptive+force+observer пережил structural mismatch: относительно nominal
mean cost −81.33%, sway −65.59%, tracking −4.53%, final error −41.11%; 76/96
cost wins. Raw angle>15deg остаётся в 8/96 случаях (nominal 12/96), поэтому
успех сравнительный, не отсутствие опасных excursions. Пять из восьми альтернатив
прошли полный критерий, причём frozen/new observer дают одинаковые траектории.

Сильный static force_actuator сохраняет преимущество по tracking: −8.57% против
nominal; settling 52/96 против 36/96 у adaptive+force+observer. У adaptive mean
cost −25.62% и sway −42.05% относительно static, но tracking +4.41%, final error
+24.83% и всего 25/96 cost wins. Данный средний выигрыш не означает преимущество
на типичном случае. Posthoc разбиение по семействам показывает пользу прежде
всего при hidden delay/combined, а на two_lag cost хуже static на 21.24%.

Fixed observer с теми же 2s истории даёт лишь −0.28% mean cost против nominal,
28/96 cost wins и tracking +1.01%. Adaptive force+observer против fixed force+
observer: cost −70.63%, но tracking +0.94%, 56/96 cost wins. Значит преимущество
online adaptation выходит за рамки одного длинного state-fit на этом grid;
универсального улучшения всех целей нет. New qualification gate не изменил
ни одного nontiming array относительно frozen winner — подтверждено отдельно.

Posthoc comparison adaptive_force_observer против force_actuator (не новый gate):

| family | cost_change_percent | cost_wins | cases | tracking_change_percent |
| --- | --- | --- | --- | --- |
| matched | 5.00007 | 4 | 16 | 6.08185 |
| delay | -49.6964 | 4 | 16 | 9.54829 |
| two_lag | 21.2384 | 0 | 16 | 7.26734 |
| friction | 24.729 | 1 | 16 | 2.96267 |
| gain_change | 6.33859 | 4 | 16 | 7.0621 |
| combined | -42.1034 | 12 | 16 | -3.32991 |

## Все результаты

| method | realized_cost | tracking_ise_m2_s | sway_ise_deg2_s | final_position_error_m | settled | rail_violation_100hz | speed_violation_100hz | angle_violation_100hz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive_force | 69.9719 | 0.0660222 | 127.832 | 0.0159245 | 0.375 | 0 | 0 | 8 |
| adaptive_force_observer | 63.4707 | 0.065941 | 116.022 | 0.011399 | 0.375 | 0 | 0 | 8 |
| adaptive_observer | 65.2557 | 0.0683422 | 116.378 | 0.0141476 | 0.34375 | 0 | 0 | 5 |
| adaptive_params | 86.4154 | 0.0683546 | 144.523 | 0.0199007 | 0.354167 | 0 | 0 | 5 |
| fixed_force_observer | 216.109 | 0.0653279 | 287.518 | 0.0152598 | 0.46875 | 0 | 0 | 12 |
| fixed_observer | 339.026 | 0.069771 | 334.36 | 0.0184715 | 0.416667 | 0 | 0 | 12 |
| force_actuator | 85.3365 | 0.0631529 | 200.207 | 0.00913149 | 0.541667 | 0 | 0 | 12 |
| frozen_adaptive_force_observer | 63.4707 | 0.065941 | 116.022 | 0.011399 | 0.375 | 0 | 0 | 8 |
| nominal | 339.963 | 0.0690707 | 337.221 | 0.0193558 | 0.375 | 0 | 0 | 12 |

Изменения средних относительно nominal, проценты (меньше лучше):

| method | realized_cost | tracking_ise_m2_s | sway_ise_deg2_s | final_position_error_m |
| --- | --- | --- | --- | --- |
| adaptive_force | -79.4178 | -4.41362 | -62.0925 | -17.7273 |
| adaptive_force_observer | -81.3301 | -4.53125 | -65.5945 | -41.1081 |
| adaptive_observer | -80.805 | -1.05474 | -65.4892 | -26.9076 |
| adaptive_params | -74.5809 | -1.03677 | -57.143 | 2.81514 |
| fixed_force_observer | -36.4315 | -5.41888 | -14.7389 | -21.1618 |
| fixed_observer | -0.275562 | 1.0139 | -0.848463 | -4.56882 |
| force_actuator | -74.8983 | -8.56777 | -40.6305 | -52.823 |
| frozen_adaptive_force_observer | -81.3301 | -4.53125 | -65.5945 | -41.1081 |
| nominal | 0 | 0 | 0 | 0 |

Парные выигрыши realized cost:

| method | paired_case_wins | cases |
| --- | --- | --- |
| nominal | 0 | 96 |
| force_actuator | 78 | 96 |
| fixed_observer | 28 | 96 |
| fixed_force_observer | 67 | 96 |
| adaptive_params | 54 | 96 |
| adaptive_force | 75 | 96 |
| adaptive_observer | 60 | 96 |
| adaptive_force_observer | 76 | 96 |
| frozen_adaptive_force_observer | 76 | 96 |

## Проверка переноса по семействам

| method | family | cost_change_percent | tracking_change_percent | sway_change_percent |
| --- | --- | --- | --- | --- |
| nominal | matched | 0 | 0 | 0 |
| nominal | delay | 0 | 0 | 0 |
| nominal | two_lag | 0 | 0 | 0 |
| nominal | friction | 0 | 0 | 0 |
| nominal | gain_change | 0 | 0 | 0 |
| nominal | combined | 0 | 0 | 0 |
| force_actuator | matched | -44.7767 | -5.92769 | -58.2366 |
| force_actuator | delay | -73.9976 | -19.158 | -36.4318 |
| force_actuator | two_lag | -78.3552 | -11.6262 | -47.5432 |
| force_actuator | friction | -32.1179 | -3.55892 | -53.7063 |
| force_actuator | gain_change | -18.3762 | -6.51865 | -44.1185 |
| force_actuator | combined | -79.7313 | -3.99786 | -38.2109 |
| fixed_observer | matched | -1.26278 | 1.2663 | -3.90883 |
| fixed_observer | delay | -0.853745 | -3.71399 | -0.925099 |
| fixed_observer | two_lag | -0.497588 | 0.728978 | -1.1879 |
| fixed_observer | friction | -2.21013 | 0.285909 | -5.3294 |
| fixed_observer | gain_change | 1.75146 | 0.444922 | -3.71729 |
| fixed_observer | combined | 0.167294 | 6.07224 | -0.144964 |
| fixed_force_observer | matched | -36.8853 | -3.10167 | -46.5401 |
| fixed_force_observer | delay | -15.4453 | -11.1532 | -5.09635 |
| fixed_force_observer | two_lag | -73.1923 | -6.14586 | -37.1095 |
| fixed_force_observer | friction | -27.4039 | -1.11887 | -44.9399 |
| fixed_force_observer | gain_change | -13.6695 | -3.8424 | -33.272 |
| fixed_force_observer | combined | -36.6396 | -5.78498 | -6.89086 |
| adaptive_params | matched | 32.9891 | 2.66515 | 32.5192 |
| adaptive_params | delay | -77.4718 | -8.09021 | -59.1102 |
| adaptive_params | two_lag | -70.7663 | -1.38816 | -52.5725 |
| adaptive_params | friction | 48.0216 | 2.89309 | 59.8272 |
| adaptive_params | gain_change | 44.3959 | 3.14906 | 29.0612 |
| adaptive_params | combined | -89.6351 | -3.14943 | -73.199 |
| adaptive_force | matched | -24.3795 | -1.84465 | -30.5542 |
| adaptive_force | delay | -85.3397 | -12.973 | -60.9898 |
| adaptive_force | two_lag | -70.91 | -6.77192 | -62.5595 |
| adaptive_force | friction | -13.2799 | -1.32206 | -19.6263 |
| adaptive_force | gain_change | 0.544244 | -1.69467 | -18.1781 |
| adaptive_force | combined | -87.7583 | -1.19021 | -68.528 |
| adaptive_observer | matched | -29.0216 | 2.12444 | -36.6057 |
| adaptive_observer | delay | -83.6158 | -9.41676 | -68.2326 |
| adaptive_observer | two_lag | -81.1707 | -2.31887 | -65.1325 |
| adaptive_observer | friction | 24.685 | 4.23644 | 37.8074 |
| adaptive_observer | gain_change | 5.78588 | 2.70822 | -13.2129 |
| adaptive_observer | combined | -89.6361 | -1.63243 | -73.5686 |
| adaptive_force_observer | matched | -42.0155 | -0.206345 | -58.2118 |
| adaptive_force_observer | delay | -86.9199 | -11.439 | -64.4119 |
| adaptive_force_observer | two_lag | -73.7582 | -5.20376 | -67.7388 |
| adaptive_force_observer | friction | -15.3313 | -0.701692 | -23.3812 |
| adaptive_force_observer | gain_change | -13.2024 | 0.0830901 | -42.6788 |
| adaptive_force_observer | combined | -88.2651 | -7.19465 | -68.9545 |
| frozen_adaptive_force_observer | matched | -42.0155 | -0.206345 | -58.2118 |
| frozen_adaptive_force_observer | delay | -86.9199 | -11.439 | -64.4119 |
| frozen_adaptive_force_observer | two_lag | -73.7582 | -5.20376 | -67.7388 |
| frozen_adaptive_force_observer | friction | -15.3313 | -0.701692 | -23.3812 |
| frozen_adaptive_force_observer | gain_change | -13.2024 | 0.0830901 | -42.6788 |
| frozen_adaptive_force_observer | combined | -88.2651 | -7.19465 | -68.9545 |

Полные gates: `results/structural_transfer_20261107/acceptance.json`.
Global требует lower mean cost/sway/tracking, final error≤1.01 reference,
не больше raw нарушений каждого типа, cost wins>48/96 и ни одного nonfinite solve.
Transfer дополнительно требует cost и tracking≤1.10 reference, без роста raw
нарушений, отдельно в КАЖДОМ семействе. Проверка сравнительная и описательная,
не гарантия безопасности. Bootstrap cost resamples 24 context clusters (2000 draws,
seed 7112026, одни и те же resample indices для всех методов), не отдельные correlated ticks: `context_bootstrap.csv`.

## Сколько даёт адаптация сверх длинного наблюдателя

| method | reference | realized_cost_change_percent | realized_cost_paired_wins | tracking_ise_m2_s_change_percent | tracking_ise_m2_s_paired_wins | sway_ise_deg2_s_change_percent | sway_ise_deg2_s_paired_wins | final_position_error_m_change_percent | final_position_error_m_paired_wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive_observer | fixed_observer | -80.752 | 56 | -2.04788 | 38 | -65.1938 | 80 | -23.4082 | 44 |
| adaptive_force_observer | fixed_force_observer | -70.6303 | 56 | 0.938484 | 48 | -59.6469 | 80 | -25.3004 | 43 |
| fixed_observer | nominal | -0.275562 | 28 | 1.0139 | 23 | -0.848463 | 44 | -4.56882 | 34 |
| fixed_force_observer | fixed_observer | -36.2559 | 74 | -6.36821 | 72 | -14.0093 | 48 | -17.3874 | 46 |
| adaptive_force_observer | adaptive_force | -9.29122 | 38 | -0.12306 | 33 | -9.23851 | 64 | -28.4187 | 39 |
| adaptive_force_observer | frozen_adaptive_force_observer | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| frozen_adaptive_force_observer | force_actuator | -25.6231 | 25 | 4.41477 | 24 | -42.0486 | 68 | 24.8316 | 28 |

Fixed_observer и fixed_force_observer используют тот же двухсекундный буфер,
профилирование initial x/v/a и current fit-RMSE gate, что и новые adaptive observer
варианты; tau=.1, K=1 остаются фиксированными. Adaptive counterparts отличаются
online tau/K. Вариант frozen_adaptive_force_observer сохраняет старый gate:
ever_valid parameter fit AND current fit-RMSE. Побитовый unit test подтверждает
прежние update states и планы. Смена gate отдельно отражена сравнением frozen/new.
Это policy comparisons: истории зависят от своих решений; RMSE между политиками
не является чистым fixed-history тестом оценивателя.

| method | current_velocity_estimate_rmse | current_acceleration_estimate_rmse | tail_estimated_tau_s | tail_estimated_gain | observer_use_fraction | id_update_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| adaptive_force | 0.0253201 | 0.121825 | 0.139708 | 0.943636 | 0 | 0.145182 |
| adaptive_force_observer | 0.0170487 | 0.078962 | 0.138008 | 0.912456 | 0.484115 | 0.156185 |
| adaptive_observer | 0.0185065 | 0.0820458 | 0.139902 | 0.919978 | 0.484831 | 0.158789 |
| adaptive_params | 0.0286123 | 0.129578 | 0.139349 | 0.946903 | 0 | 0.147331 |
| fixed_force_observer | 0.0233558 | 0.103653 | 0.1 | 1 | 0.0949219 | 0 |
| fixed_observer | 0.0262406 | 0.111188 | 0.1 | 1 | 0.0876953 | 0 |
| force_actuator | 0.0213577 | 0.105094 | 0.1 | 1 | 0 | 0 |
| frozen_adaptive_force_observer | 0.0170487 | 0.078962 | 0.138008 | 0.912456 | 0.484115 | 0.156185 |
| nominal | 0.029933 | 0.131705 | 0.1 | 1 | 0 | 0 |

Tau и gain здесь effective first-order approximations. У cascade/delay/friction
нет единственного истинного tau данной модели. Tail estimates не ошибки параметров;
у static scenario bank id_tau/gain — только point prior. Fit qualification и
конечный tau grid не доказывают идентифицируемость, persistent excitation или stability.

## Что именно усложнено

Первый порядок; скрытая задержка .04/.09s; два последовательных лага; friction
fc*tanh(v/.015)+.08v; изменение gain в 3.5s на .7/1.3; все эффекты вместе.
Четыре заранее заданных archetypes × 6 families × exact/noisy × pulse/noPulse.
Nuisances в archetypes связаны: это не независимый factorial test параметров.
Новые L .28/.98m, omega ratio .91/1.09 и seed 1407112026. Target .3m до 4s,
затем .15m; начальное x .1m, горизонт эксперимента 8s. Новая цель доступна
controller только начиная с 4s. Pulse .13rad/s² на [5.25,5.6)s.

Все controller получают .4s angular history и имеют доступ к одинаковой 2s
истории measured x/собственных ISSUED acceleration commands. Задержанная команда,
истинные состояния/параметры и будущие измерения доступны только plant/audit.
Endpoint v/a, tau/K профилируются причинно; future committed interval не входит
в estimator. Данные сохраняются на 100Hz, MPC работает на 20Hz с committed tick.
Cost и tracking ISE учитывают ступенчатую цель на control grid; trapezoid convention
в точке разворота задана заранее. Settling относится только к final target после 4s.
Actual acceleration effort учитывает friction, не только latent motor state.

Работа [Rohrs et al., 1985](https://doi.org/10.1109/TAC.1985.1104070)
([NASA record](https://ntrs.nasa.gov/citations/19850063162)) мотивирует явный тест
unmodeled dynamics. Наша конечная simulation grid и MPC не повторяют их алгоритм
или теоремы. Истинный actuator в прошлом опыте был близок к fitted first-order
структуре; этот этап проверяет именно практический предел такого совпадения.

## Проверка и время

172 unit tests PASS. Повторно интегрированы все 918 эпизода,
причинно восстановлены все estimator states, перепроверены 146880
выбранных forecasts/costs. Solver не запускался заново при полном replay.
SHA всех frozen inputs/results и 1281 исходных
перенесённых файлов проверены. Net acceleration/hidden motor state разделены;
raw issued queues и sensor measurements восстановлены. Metrics independently
recomputed; halfstep convergence проверен, подробности в verification.json. Дополнительный independent_metrics.json
восстанавливает net acceleration из latent motor/friction без plant/metric helpers
и заново вычисляет cost для всех 864 evaluation episodes. Distribution tails
сохранены в posthoc_cost_distribution.csv; анализа для tuning не было.

Warmed batch1 полный current decision, миллисекунды:

| method | median | max | fraction_over_50ms |
| --- | --- | --- | --- |
| adaptive_force | 7.21204 | 7.87171 | 0 |
| adaptive_force_observer | 7.83929 | 8.31321 | 0 |
| adaptive_observer | 7.76871 | 8.16775 | 0 |
| adaptive_params | 7.19162 | 7.26042 | 0 |
| fixed_force_observer | 5.37483 | 5.62517 | 0 |
| fixed_observer | 5.56988 | 5.78912 | 0 |
| force_actuator | 5.17254 | 5.454 | 0 |
| frozen_adaptive_force_observer | 8.15971 | 8.97263 | 0 |
| nominal | 5.16742 | 5.421 | 0 |

15 измерений на метод: случаи/тики (0,0),(31,40),(95,100), один warmup на точку,
5 repeats с одинаковым восстановленным pre-tick estimator state. Включены current
window assembly/update/observer/solver; исключены model I/O и восстановление
pre-tick state. Нет real-time/WCET guarantee. Wall-clock timing не добавляет
simulation delay сверх declared committed tick. Batch throughput отдельно в
runtime_batches.csv; plan differences относительно batch run в runtime_batch1.csv.

## Практические пределы

Это synthetic acceleration-command ablation: native xr features zero, trained
NeuralActuator bypass. Нельзя переносить вывод на нативный контроллер или реальный
кран без согласования xr/command timing и hardware tests. Нет доказательства
stability, отсутствия violations вне grid или SOTA. Все neural weights/defaults
и APHYNITY остаются прежними. Следующая новая методика должна сравниваться с
сильным static force_actuator и matched-memory observer, а не только nominal.
