# Scenario/physical-Jacobian MPC: протокол до нового опыта

7 ноября 2026. Источник 1daf4b1; run scenario_control_20261107.
Предыдущие96cases использованы только как источник гипотез: MAE adapter лучше,
но его command Jacobian хуже, tracking и terminal error выросли. Новые данные
ниже не использованы для настройки. Веса прежние; APHYNITY отдельно.

## Шесть заранее фиксированных методов

- nominal: один physical predictor, tau=.1, gain1, effective forcing.
- frequency: средняя стоимость по omega ratios .85/1/1.15.
- joint: 18 сценариев = 3 частоты × 3 actuator пары × 2 закона forcing.
- spread: joint + .25 mean[(theta_s−mean_s theta)²/.035²] по30steps/scenarios.
- hybrid: nominal + фиксированная на текущем тике bounded neural residual.
- joint_hybrid: joint + та же фиксированная neural residual.

Actuator пары (tau,gain): (.07,.8),(.1,1),(.18,1.2). Это коррелированная
проектная сетка, не оценённое распределение и не Cartesian всех lag/gain.
Forcing laws: b=omega²/g или b=1/L. Общие q и theta_rate от номинального
observer сохраняются; p_s=theta_rate/omega_s. Сценарии не получают true parameters.

Hybrid offset=clip(theta_adapter(warm)−theta_nominal(warm),−.015,.015)rad.
Он вычисляется один раз на каждом тике из измеренной истории и warm command
plan; не меняется при пробных командах solver. Веса frozen, default bundle
профиль не меняется. Это контрольная модификация функции прогноза: её command
Jacobian физический; она не равна исходному end-to-end forecaster.

Joint минимизирует среднее квадратное residual cost по конечной сетке, а не
worst-case. Spread добавляет эвристический penalty разброса. Ни один метод
не обеспечивает tube invariance, chance constraints или доказанную hardware
устойчивость. Сценарии могут не включать true plant; часть новых lag вне сетки.

## Связь с первичными исследованиями

Hewing/Kabzan/Zeilinger, Cautious MPC using Gaussian Process Regression:
https://arxiv.org/abs/1705.10702 — неопределённость модели влияет на планирование.
Здесь GP/вероятностное распространение и вероятностные гарантии этой статьи
не реализуются. Вдохновение ограничено учётом prediction uncertainty.
Mayne et al., Constrained MPC: Stability and optimality (2000):
https://www.sciencedirect.com/science/article/abs/pii/S0005109899002149 — свойства
устойчивости зависят от условий конструкции. Наша конечная сетка и input box
не являются выполнением этих условий. Frozen residual с physical sensitivity —
наша гипотеза по предыдущей диагностике; отдельная публикационная теорема не заявлена.

## Общий информационный контракт и solver

Как в angular_control: acceleration-command ablation, native xr features0,
NeuralActuator bypass, 41 реальных измерений100Hz, общий causal observer и
backward velocity/acceleration fits. Никаких true v/rate/a/future measurements.
6s, target .1→.3m, committed command delay .05s, horizon30, 6 block levels,
input box±.8 и trust region±.2. Tracking/theta/effort/slew/terminal weights прежние.
Joint uses mean scenario cost; сохранившиеся individual costs не подгоняются.

Все trial forecasts при frozen offset аффинны по шести command levels.
Jacobian восстанавливается unit central probes (±1) для точной affine карты:
это виртуальные математические пробы, не применяемые к plant команды.
В отличие от малого epsilon снижается cancellation roundoff. QP решается
200 projected FISTA steps у всех методов; fractions0/.25/.5/1 проверяются
на полном текущем objective. Нет заявления о точном optimum. Без state constraints:
rail[0,.5], speed>.5 и angle>15deg — сохранённые нарушения.

## Новые случаи и замораживание

Development: L=.36/.56/.76 × initial_angle2/6deg, nominal plant, exact sensors,
6эпизодов на метод. Только qualification, без выбора гиперпараметров/модели.
Его source/результаты SHA фиксируются до evaluation.

Evaluation: L=.29/.64/1.11 × frequency ratio .86/1.14 × lag .075/.205 × gain
.85/1.15 =24contexts. Частота внутри проектного интервала, lag .205 вне сетки.
Parity назначение nuisance как в предыдущем протоколе: damping .015/.09,
deadzone0/.025, force laws effective/geometric; initial_angle теперь2.5/5.5deg.
Каждый контекст × exact/noisy sensors × pulse absent/present:96cases на метод,
576 evaluation +36 development эпизодов. Noise seed1007112026; pulse .17rad/s²
на[2.65,3)s. Набор не совпадает со старым. Не считать nuisance независимым factorial.
Все методы получают одинаковые noise/disturbance массивы по физическому времени.
Симулятор семейства прежний, поэтому это новый grid, не независимое оборудование.

## Метрики, gates и проверка

Главный reference — nominal physical на ЭТОМ новом наборе. Все методы
публикуются; evaluation не выбирает задним числом обученную модель.
Практический gate для каждого варианта: средние realized cost, sway ISE и
tracking ISE ниже nominal; final position error <=1.01×nominal; raw100Hz
violations не больше nominal; cost wins >половины96cases. Gate — не статистический
сертификат. Показывать также tail RMS, effort, settled rate, по-context пары
и описательный cluster bootstrap по24contexts, а не correlated ticks.

До запуска: parity с nominal objective, scenario mean, сохранение physical rate,
exact affine features, fixed/bounded residual с неизменным physical Jacobian,
solver batch independence, committed step, hard bounds и future target poisoning.
Source/config/design/tests коммитятся до development. После: полный plant replay
и half-step convergence, все chosen forecasts/objectives из causal measured history
с точной реконструкцией warm plan для hybrid offset, проверки SHA и1281originals.
Результат mean angle forecast хранится явно как среднее сценариев, не neural
prediction. Runtime batches отдельно от изолированного full decision batch1.
Simulation офлайн, wall-clock solver delay сверх committed tick не применяется.
Никакого утверждения о native-command/hardware/real-time успехе.
