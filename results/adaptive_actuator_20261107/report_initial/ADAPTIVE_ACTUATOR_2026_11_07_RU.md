# Online actuator identification и разделение uncertainty, 7 ноября2026

Завершены54development и864evaluation эпизода:96на каждый из9controller.
Новый протокол/код зафиксированы до запуска: `522d01d16192635d0759ff0e5906906f497a7b8a`.
Нейронные веса неизменны; online tau/gain updates являются заранее заданной
частью adaptive controller, а не подбором offline hyperparameters по test.
После development настройки не менялись. Все8альтернатив публикуются.

## Результаты на новом наборе

| method | realized_cost | tracking_ise_m2_s | sway_ise_deg2_s | tail_rms_deg | final_position_error_m | settled | rail_violation_100hz | angle_violation_100hz | speed_violation_100hz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| actuator_only | 38.0615 | 0.0417123 | 52.6026 | 1.36774 | 0.00834834 | 0.770833 | 0 | 3 | 0 |
| adaptive_force | 27.8379 | 0.0421617 | 41.1301 | 0.96985 | 0.00265624 | 0.75 | 0 | 0 | 0 |
| adaptive_force_observer | 24.1274 | 0.0424286 | 27.0716 | 0.472541 | 0.00414978 | 0.875 | 0 | 0 | 0 |
| adaptive_observer | 33.8294 | 0.043876 | 49.2098 | 0.948526 | 0.00634522 | 0.84375 | 0 | 0 | 0 |
| adaptive_params | 43.9206 | 0.0441155 | 82.3385 | 1.66257 | 0.00570907 | 0.75 | 0 | 8 | 0 |
| force_actuator | 26.9992 | 0.040528 | 32.0435 | 0.869094 | 0.00614033 | 0.833333 | 0 | 0 | 0 |
| force_only | 48.5797 | 0.0435534 | 62.0441 | 1.69831 | 0.009116 | 0.666667 | 0 | 6 | 0 |
| joint_previous | 50.5803 | 0.0465004 | 50.6946 | 1.79386 | 0.0164684 | 0.625 | 0 | 0 | 0 |
| nominal | 72.1223 | 0.0449328 | 114.721 | 2.3295 | 0.010058 | 0.645833 | 0 | 8 | 0 |

Процентные изменения относительно nominal physical:

| method | command_variation | effort_acceleration2 | final_position_error_m | realized_cost | sway_ise_deg2_s | tail_rms_deg | tracking_ise_m2_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| actuator_only | -20.2999 | -41.8795 | -16.9976 | -47.2264 | -54.1472 | -41.2862 | -7.16735 |
| adaptive_force | -26.4663 | -53.0484 | -73.5906 | -61.4018 | -64.1476 | -58.3666 | -6.16729 |
| adaptive_force_observer | -48.872 | -72.195 | -58.7413 | -66.5466 | -76.4021 | -79.7149 | -5.57319 |
| adaptive_observer | -40.899 | -54.3745 | -36.9135 | -53.0944 | -57.1047 | -59.282 | -2.35201 |
| adaptive_params | -13.5257 | -22.5891 | -43.2382 | -39.1026 | -28.227 | -28.6299 | -1.81904 |
| force_actuator | -31.3347 | -60.1889 | -38.9505 | -62.5647 | -72.0683 | -62.6918 | -9.80319 |
| force_only | -14.1248 | -26.8817 | -9.36531 | -32.6426 | -45.9172 | -27.0954 | -3.06999 |
| joint_previous | 23.9408 | 7.28632 | 63.7354 | -29.8687 | -55.8104 | -22.9938 | 3.48872 |
| nominal | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Парные выигрыши стоимости:

| method | change_percent | paired_case_wins | cases |
| --- | --- | --- | --- |
| nominal | 0 | 0 | 96 |
| force_only | -32.6426 | 82 | 96 |
| actuator_only | -47.2264 | 82 | 96 |
| force_actuator | -62.5647 | 78 | 96 |
| joint_previous | -29.8687 | 45 | 96 |
| adaptive_params | -39.1026 | 66 | 96 |
| adaptive_force | -61.4018 | 80 | 96 |
| adaptive_observer | -53.0944 | 63 | 96 |
| adaptive_force_observer | -66.5466 | 76 | 96 |

Фиксированные gates: `{"force_only": {"cost_improved": true, "sway_improved": true, "tracking_improved": true, "final_error_within_1percent": true, "no_additional_raw_violations": true, "majority_cost_wins": true, "passed": true}, "actuator_only": {"cost_improved": true, "sway_improved": true, "tracking_improved": true, "final_error_within_1percent": true, "no_additional_raw_violations": true, "majority_cost_wins": true, "passed": true}, "force_actuator": {"cost_improved": true, "sway_improved": true, "tracking_improved": true, "final_error_within_1percent": true, "no_additional_raw_violations": true, "majority_cost_wins": true, "passed": true}, "joint_previous": {"cost_improved": true, "sway_improved": true, "tracking_improved": false, "final_error_within_1percent": false, "no_additional_raw_violations": true, "majority_cost_wins": false, "passed": false}, "adaptive_params": {"cost_improved": true, "sway_improved": true, "tracking_improved": true, "final_error_within_1percent": true, "no_additional_raw_violations": true, "majority_cost_wins": true, "passed": true}, "adaptive_force": {"cost_improved": true, "sway_improved": true, "tracking_improved": true, "final_error_within_1percent": true, "no_additional_raw_violations": true, "majority_cost_wins": true, "passed": true}, "adaptive_observer": {"cost_improved": true, "sway_improved": true, "tracking_improved": true, "final_error_within_1percent": true, "no_additional_raw_violations": true, "majority_cost_wins": true, "passed": true}, "adaptive_force_observer": {"cost_improved": true, "sway_improved": true, "tracking_improved": true, "final_error_within_1percent": true, "no_additional_raw_violations": true, "majority_cost_wins": true, "passed": true}}`.
Требуются одновременно lower mean cost/sway/tracking, final error<=1.01×nominal,
raw100Hz violations не выше и cost wins>48/96. Меньшее среднее не заменяет
большинство пар. Это finite-family criteria, не hardware или stability guarantee.

## Что отделено в этом опыте

Static: nominal; force_only с2законами forcing; actuator_only с3парами(tau,gain);
force_actuator с6сценариями; joint_previous с прежними18сценариями (ещё3частоты).
Поэтому force_only vs nominal и actuator_only vs nominal разделяют факторы,
которые в предыдущем joint изменялись одновременно. Joint_previous против
force_actuator отдельно показывает вклад frequency bank на этом новом grid.

Adaptive: adaptive_params с online lag/gain и старым trolley state observer;
adaptive_force добавляет2force laws. Adaptive_observer и adaptive_force_observer
дополнительно используют fitted endpoint v/a. Angular observer q/p остаётся общим.
Эти policy-ablations имеют одинаковые plant/noise cases, но разные trajectories;
качество идентификации endogenous к сигналам своего controller.

## Новый информационный контракт и оцениватель

Всем доступен буфер2s measured x и собственных уже применённых acceleration
commands на50ms grid. Angular history по-прежнему41samples100Hz (.4s). Static
методы не оценивают actuator по длинному буферу. Это явное расширение памяти,
поэтому опыт не является чистой заменой архитектуры старого41sample forecaster.
True v/a, параметры plant и будущие измерения контроллер не получает.

Profiled model: first-order lag tau и scalar gain K, initial x/v/a nuisance.
Для8tau grid values unit-gain forced response точно интегрируется. Initial
state basis[1,t,tau*(t-tau*(1-exp(-t/tau)))] профилируется, gain оценивается ridgeLS
с prior1/clips[.5,1.6]. Minimum21positions, informationRMS>=.0004m, fitRMSE<=.001m,
update раз в.2s, EMA.25; иначе prior/последняя оценка. Deadzone не fitted.

Endpoint observer при filtered tau/K профилирует initial state на past window
и вычисляет current v/a. Применяется после valid fit при текущем RMSE<=.001m;
clips v±.6,a±1.5, measured current x сохраняется. Только trolley initial state
меняется; angular observer остаётся прежним. Всё фиксировано внутри решения MPC.

Диагностика по истинному plant ПОСЛЕ решения (не вход controller):

| method | current_velocity_estimate_rmse | current_acceleration_estimate_rmse | tail_tau_error_s | tail_gain_error | id_update_fraction | observer_use_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| actuator_only | 0.0172099 | 0.0699934 | 0.065 | 0.22 | 0 | 0 |
| adaptive_force | 0.0164585 | 0.064957 | 0.0176855 | 0.109973 | 0.224826 | 0 |
| adaptive_force_observer | 0.0085346 | 0.0354705 | 0.0206782 | 0.144126 | 0.221354 | 0.82309 |
| adaptive_observer | 0.00955697 | 0.0386658 | 0.0206492 | 0.132867 | 0.223351 | 0.823698 |
| adaptive_params | 0.0197319 | 0.0756633 | 0.0121372 | 0.0983631 | 0.223351 | 0 |
| force_actuator | 0.0149498 | 0.0626913 | 0.065 | 0.22 | 0 | 0 |
| force_only | 0.0192722 | 0.0763434 | 0.065 | 0.22 | 0 | 0 |
| joint_previous | 0.0230307 | 0.106678 | 0.065 | 0.22 | 0 | 0 |
| nominal | 0.0220544 | 0.0880096 | 0.065 | 0.22 | 0 | 0 |

Tau/gain errors у adaptive методов описывают effective fit; из-за unmodelled
deadzone он не обязан совпадать с истинными параметрами. У static банков
id_tau/id_gain — nominal point prior, не единственный параметр всего банка.
Information threshold относится к residualized forced response и не гарантирует
идентифицируемость tau. При почти постоянной команде tau может смешиваться
с nuisance initial state и определяться prior. Отдельного fixed-parameter
2s-observer baseline здесь нет: влияние длинной памяти/сглаживания не полностью
отделено от fitted dynamics.

Update/use fractions показывают, сколько раз qualification реально пропустила
обновление или новый observer; low-information окна не объявляются успешным fit.

Связь с первичными работами: [Åström/Wittenmark1973](https://portal.research.lu.se/en/publications/on-self-tuning-regulators-2/)
совмещают LS estimator и self-tuning regulator. Наш finite-window profiled fit
и MPC не повторяют их алгоритм/теоремы. Связанный1972technical report отличается
от journal1973. [Gevers2005](https://perso.uclouvain.be/michel.gevers/PublisMig/P78.pdf)
рассматривает роль цели управления и информативности модели. Наша heuristic
qualification не доказывает persistent excitation, consistency или stability.

## Область проверки

L=.25/.62/1.07 × omega ratio.87/1.13 × tau.06/.19 × gain.78/1.22 =24contexts,
каждый exact/noisy×pulse absent/present. Balanced nuisance: damping.02/.08,
deadzone0/.02,effective/geometric forcing, initialangle3/5deg. Pulse.16rad/s²
на[2.45,2.85)s, новый seed1207112026. Это новый grid прежнего simulator family,
не independent hardware. Nuisance назначения не независимый full factorial.
Development L=.34/.54/.74×angle2/6deg — qualification без tuning.

Paired metrics имеют равные веса96episodes; context bootstrap resamples24contexts
целиком, не correlated ticks. Все негативные cases/violations сохранены. ISE
на50ms grid, peak/violations также на100Hz. Contact/state clipping отсутствуют.
Native xr features0, NeuralActuator bypass: acceleration-command ablation.
Оригинальные real-forecast gates не отменяются этим синтетическим опытом.

## Runtime и проверка

Batch time, включая history assembly/online update/observer/solver:

| stage | method | cases | batch_median_ms | batch_p95_ms | batch_max_ms | amortized_median_ms_per_case |
| --- | --- | --- | --- | --- | --- | --- |
| development | nominal | 6 | 5.41798 | 5.61226 | 26.5062 | 0.902997 |
| development | force_only | 6 | 5.38808 | 5.78398 | 5.93267 | 0.898014 |
| development | actuator_only | 6 | 5.48538 | 5.76399 | 6.14346 | 0.914229 |
| development | force_actuator | 6 | 5.8686 | 6.06179 | 6.32288 | 0.978101 |
| development | joint_previous | 6 | 6.29531 | 6.60979 | 7.01429 | 1.04922 |
| development | adaptive_params | 6 | 5.3204 | 7.95693 | 9.20154 | 0.886733 |
| development | adaptive_force | 6 | 4.90706 | 7.45973 | 7.62579 | 0.817844 |
| development | adaptive_observer | 6 | 5.88475 | 9.05458 | 9.55067 | 0.980792 |
| development | adaptive_force_observer | 6 | 5.8196 | 8.67819 | 9.05129 | 0.969934 |
| evaluate | nominal | 96 | 6.66629 | 7.0707 | 11.421 | 0.0694405 |
| evaluate | force_only | 96 | 7.85358 | 8.2637 | 8.67792 | 0.0818082 |
| evaluate | actuator_only | 96 | 8.74956 | 9.21719 | 9.67671 | 0.0911413 |
| evaluate | force_actuator | 96 | 11.2403 | 11.8939 | 12.4437 | 0.117086 |
| evaluate | joint_previous | 96 | 21.3029 | 21.8241 | 30.0883 | 0.221906 |
| evaluate | adaptive_params | 96 | 7.08104 | 10.1815 | 10.6259 | 0.0737608 |
| evaluate | adaptive_force | 96 | 8.16775 | 11.2337 | 11.5789 | 0.0850807 |
| evaluate | adaptive_observer | 96 | 7.75675 | 10.9689 | 11.1452 | 0.0807995 |
| evaluate | adaptive_force_observer | 96 | 8.87819 | 11.9773 | 12.2979 | 0.0924811 |

Isolated warm batch1 full decision,15измерений на метод, milliseconds:

| method | median | max | fraction_over_50ms |
| --- | --- | --- | --- |
| actuator_only | 4.98825 | 5.27604 | 0 |
| adaptive_force | 7.62746 | 8.3705 | 0 |
| adaptive_force_observer | 8.631 | 9.26258 | 0 |
| adaptive_observer | 8.23063 | 8.724 | 0 |
| adaptive_params | 7.54442 | 7.62379 | 0 |
| force_actuator | 5.18033 | 5.52854 | 0 |
| force_only | 4.91821 | 5.068 | 0 |
| joint_previous | 5.18446 | 5.39667 | 0 |
| nominal | 4.91025 | 5.27629 | 0 |

Для каждого повтора восстанавливается одно и то же pre-tick estimator state.
Его восстановление и чтение файлов не timed; текущий online update timed.
Выбраны ticks0/40/100 (update ticks), а не только дешёвые промежуточные тики.
Amortization — throughput, не latency. Simulation офлайн; wall-clock delay
сверх committed tick не действует на plant. Это не WCET/scheduling guarantee.

164unit tests прошли. Все 918episodes повторно интегрированы,
предикторные планы 110160решений replayed. Online identifier state
восстановлен последовательно из measured position/applied commands, а не взят
из сохранённых estimates. SHA, bounds, pipeline, shared noise/disturbances,
warm-cost monotonicity, halfstepRK и 1281original
files проверены. Full replay не перезапускал optimizer. Все fitted traces
и raw100Hz trajectories опубликованы; neural weights не переобучались.
