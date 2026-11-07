# Первое замкнутое сравнение angular predictor, 7 ноября 2026

Завершены 18 development и 288 evaluation эпизодов. Сравниваются физический
осциллятор, прежний phase_space и новый adapter_direct8, все на PRIMARY.
Веса заморожены; настройки не подбирались по development/evaluation.
Протокол и код: `b0e9d2bb659604ab015b2ec0e59fd6d03642b1cd`. Gate замены: **False**.

## Средние результаты на 96 одинаковых условиях

| method | realized_cost | tracking_ise_m2_s | sway_ise_deg2_s | tail_rms_deg | final_position_error_m | effort_acceleration2 | command_variation | settled | solver_nonfinite | fallbacks | current_acceleration_estimate_rmse | rail_violation_100hz | angle_violation_100hz | speed_violation_100hz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate | 89.9145 | 0.0559154 | 76.9474 | 2.45796 | 0.033754 | 0.49722 | 7.2325 | 0.28125 | 0 | 0 | 0.112011 | 4 | 8 | 0 |
| physical | 74.5446 | 0.0440388 | 83.8089 | 2.0427 | 0.0127054 | 0.433333 | 5.58394 | 0.666667 | 0 | 0 | 0.0919264 | 0 | 8 | 0 |
| retained | 90.7861 | 0.054368 | 87.344 | 2.54857 | 0.0259537 | 0.542778 | 6.73342 | 0.104167 | 0 | 0 | 0.109594 | 4 | 12 | 0 |

Изменения нового adapter относительно baseline и прежней сети:

| reference | metric | candidate_mean | reference_mean | change_percent | paired_case_wins | cases |
| --- | --- | --- | --- | --- | --- | --- |
| physical | realized_cost | 89.9145 | 74.5446 | 20.6184 | 12 | 96 |
| physical | tracking_ise_m2_s | 0.0559154 | 0.0440388 | 26.9685 | 44 | 96 |
| physical | sway_ise_deg2_s | 76.9474 | 83.8089 | -8.18701 | 21 | 96 |
| physical | tail_rms_deg | 2.45796 | 2.0427 | 20.329 | 23 | 96 |
| physical | final_position_error_m | 0.033754 | 0.0127054 | 165.666 | 15 | 96 |
| physical | effort_acceleration2 | 0.49722 | 0.433333 | 14.7432 | 12 | 96 |
| physical | command_variation | 7.2325 | 5.58394 | 29.5232 | 27 | 96 |
| retained | realized_cost | 89.9145 | 90.7861 | -0.960015 | 43 | 96 |
| retained | tracking_ise_m2_s | 0.0559154 | 0.054368 | 2.84625 | 54 | 96 |
| retained | sway_ise_deg2_s | 76.9474 | 87.344 | -11.903 | 62 | 96 |
| retained | tail_rms_deg | 2.45796 | 2.54857 | -3.55541 | 50 | 96 |
| retained | final_position_error_m | 0.033754 | 0.0259537 | 30.0549 | 39 | 96 |
| retained | effort_acceleration2 | 0.49722 | 0.542778 | -8.39354 | 69 | 96 |
| retained | command_variation | 7.2325 | 6.73342 | 7.41197 | 42 | 96 |

Критерии из preregistration: `{"sway_improved": true, "tracking_improved": false, "final_error_within_1percent": false, "no_additional_raw_violations": true, "majority_paired_cost_wins": false, "passed": false}`.
Gate относится только к этой новой angular-control задаче. Результат
не отменяет не пройденный load gate в реальном forecast benchmark.

## Что именно измерено

На каждом .05s тике optimizer читает 41 измеренную точку theta/x при100Hz,
причинные оценки v/a/q/p и номинальную частоту из TRAIN calibration. Истинные
v, angle_rate, actuator acceleration и его будущие значения скрыты. Команда
ускорения проходит через общий номинальный lag, а angular predictor получает
его среднее предсказанное ускорение. Реальный plant использует независимые
lag/gain/deadzone, нелинейный угол и скрытую frequency error.
Native reference features обнулены, NeuralActuator обходится. Это явная
замена входного контракта; данный результат не оценивает native-command
контроллер, реальный кран или аппаратную безопасность.

Один move-blocked horizon30, 6 уровней, hard bound±.8, trust region±.2,
central FD Jacobian (.002), 100 FISTA шагов и nonlinear backtracking у всех
трёх методов. Первый command уже committed; оптимизированный следующий
применяется через один тик. Fallback/nonfinite и все нарушения сохраняются.
Модельные веса и controller weights не обучались. Planning cost не равен
realized episode cost; отдельные tracking/sway/effort показатели сохранены.

Вдохновение: [DiffTune-MPC](https://arxiv.org/html/2312.11384v2) подчёркивает
разницу внутренней цели MPC и результата всего замкнутого эпизода. Мы не
реализуем обучение DiffTune или KKT differentiation. [Дифференцируемый MPC
Amos et al.](https://proceedings.neurips.cc/paper/2018/hash/ba6d843eb4251a4526ce65d1807a9309-Abstract.html)
служит общей основой; наш solver использует локальную linearization и FD.

## Диагностика чувствительности: отдельно после основного опыта

| method | absolute_jacobian_error | jacobian_cosine | mean_counterfactual_forecast_error_deg |
| --- | --- | --- | --- |
| candidate | 0.34738 | 0.838173 | 1.77821 |
| physical | 0.310385 | 0.883255 | 1.49906 |
| retained | 0.325708 | 0.868478 | 1.87539 |

Это exploratory post-hoc audit, не использованный для выбора модели, весов
или controller. Все три predictor получают одну историю физического baseline
для всех96cases на заранее фиксированных ticks20/60/100. Сравнивается Jacobian
30 углов по6 command levels с независимым нелинейным plant. Его истинное
состояние используется только здесь, в oracle диагностике, не в контроллере.
Всего864 модельных сравнения. У8 из288 историй true Jacobian практически
нулевой из-за deadzone: relative error и cosine для них не определены;
absolute error сохранена для всех. Начальная диагностическая версия с делением
на epsilon сохранена отдельно; её огромные relative means не интерпретируются.

Новый adapter улучшает среднюю counterfactual forecast error относительно
прежней сети, но хуже по средней absolute Jacobian error и cosine. Это
согласуется с необходимостью проверять реакцию на управляющие команды отдельно
от MAE. Диагностика не доказывает единственную причину ухудшения closed loop:
есть также неопределённость привода, наблюдателя и изменённый zero-xr контракт.

## Область сценариев и статистические пределы

24 контекста L=.27/.57/1.03 × omega_ratio .88/1.12 × tau .065/.18 × gain .8/1.2.
Дополнительные damping/deadzone/force-law/initial-angle назначены заранее
по сбалансированной таблице, но не образуют независимый full factorial.
Каждый контекст содержит exact/noisy sensors × pulse absent/present.
Sensor noise и disturbances совпадают между controllers по физическому времени.
Поэтому сравнение парное; единица resampling — целый контекст, а не tick.
`context_bootstrap.csv` даёт описательные интервалы по этому выбранному
семейству, не гарантию переноса на новое устройство.

Development L=.32/.52/.72 × angle2/6deg используется только как qualification.
Настройки после него не менялись; его SHA зафиксированы до evaluation.
История прогревается физически .4s, episode6s, target .1→.3m.
Raw100Hz trajectories сохранены. ISE/settling/terminal метрики рассчитаны
на50ms grid; 100Hz peak/нарушения находятся отдельно в raw_sampling_metrics.csv.

## Время вычислений

| stage | method | batch_cases | decisions | batch_median_ms | batch_p95_ms | batch_max_ms | amortized_median_ms_per_case | batch_total_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | physical | 6 | 120 | 3.68892 | 4.43416 | 25.0934 | 0.61482 | 0.480858 |
| development | retained | 6 | 120 | 23.4171 | 26.1113 | 133.802 | 3.90286 | 2.96683 |
| development | candidate | 6 | 120 | 46.609 | 50.7027 | 60.2892 | 7.76817 | 5.6583 |
| evaluate | physical | 96 | 120 | 5.41031 | 5.67174 | 28.644 | 0.0563574 | 0.671992 |
| evaluate | retained | 96 | 120 | 76.7887 | 79.024 | 85.8225 | 0.799882 | 9.16171 |
| evaluate | candidate | 96 | 120 | 167.778 | 184.214 | 350.215 | 1.74769 | 20.6952 |

Это полное время решения для батча, включая history/observer/прямые проходы,
local QP и проверку nonlinear proposals. Амортизация на case — throughput,
не задержка одиночного контроллера и не гарантия50ms deadline.
Изолированный warmed batch-one замер (15 измерений на метод, три фиксированных
истории; median/max в миллисекундах):

| method | median | max |
| --- | --- | --- |
| candidate | 36.2717 | 37.4925 |
| physical | 3.55229 | 3.87892 |
| retained | 19.3361 | 27.0676 |

Batch-one и batch96 планы различаются не более .000461m/s² у candidate
из-за округления FD/линейной алгебры; это записано в runtime_batch1.csv.

Симуляция офлайн: кроме объявленного committed тика фактическое wall-clock
время не задерживает plant. Deadline-aware исполнение и scheduling здесь
не проверены; превышения50ms нельзя интерпретировать как real-time успех.

## Проверка воспроизведения

Unit tests: 150 PASS. Повторно интегрированы все 306 эпизодов и
воспроизведены прогнозы для 36720 решений. Проверены точная
реконструкция измерений, paired exogenous arrays, pipeline, hard bounds,
best objective против warm start, causal state estimates и source SHA.
Original main/APHYNITY: 1281 файлов совпадают.
Полная проверка: results/angular_control_20261107/verification.json.
Предикторные планы воспроизведены, сам optimizer во full replay не запускался.
Численная сходимость plant проверена половинным RK4 шагом на всех эпизодах.
Первоначальная проверка остановилась на различии causal acceleration estimate
2.53e-15 из-за layout-dependent NumPy dot; после воспроизведения contiguous
истории из runner оценки совпали точно, tolerance не ослаблялся. Initial log
и JSON сохранены; веса и результаты управления не менялись.

Следующие обоснованные гипотезы: обучать реакцию на команду/многошаговый
closed-loop критерий на НОВОМ train simulator set; либо сохранять физический
command Jacobian, обучая ограниченную residual поправку/наблюдатель. Они ещё
не реализованы и потребуют нового протокола, без подгонки по этим96cases.
Native-controller по-прежнему требует согласования xr и модели привода.

Все неудачные/отрицательные результаты остаются в таблицах. Улучшение forecast
на реальных записях не даёт автоматического улучшения closed loop; решение
о следующем шаге следует принимать по sway/tracking/violations/runtime вместе.
