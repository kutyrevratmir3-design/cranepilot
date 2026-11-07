# Замкнутый angular-control эксперимент: протокол до запуска

7 ноября 2026. Новый run `angular_control_20261107`, исходный HEAD a4c1f5c.
Это отдельная синтетическая ablation с командами ускорения. Native xr features
задаются нулями; обученный NeuralActuator обходится. Поэтому estimand — влияние
углового прогноза при общем условном приводе, а не полный пилот на родных командах.
Исходные веса, обработка данных, APHYNITY и старые результаты не меняются.

## Основание и исследовательский вопрос

Новый adapter снизил primary forecast MAE на 3.46%, но load-all ухудшилась на
2.02%. Проверяем, помогает ли этот заранее выбранный predictor замкнутому
управлению. Сравнение: physical, retained phase_space, candidate adapter_direct8,
все из PRIMARY и зафиксированного research_bundle_20261107. Fold не маршрутизирует
модель по длине. Нет нового обучения, выбора controller по test или fine-tuning.

DiffTune-MPC различает внутренний planning cost и более длинный реализованный
closed-loop результат: https://arxiv.org/html/2312.11384v2 . Его способ обучения
MPC здесь не воспроизводится. Работа Amos et al. обосновывает дифференцируемое
представление MPC: https://proceedings.neurips.cc/paper/2018/hash/ba6d843eb4251a4526ce65d1807a9309-Abstract.html .
У нас один локальный linearization/QP на тик, конечные разности и проверка
реального нелинейного predictor cost, а не KKT differentiation или DiffTune.
Предыдущий аудит: CONTROL_INTEGRATION_AUDIT_20261107.md.

## Информация и индексирование

Датчики theta/x измеряются каждые .01s. Реальный прогрев от -.4 до 0s с нулевой
командой создаёт 41 отсчёт. Копирование постоянного состояния вместо истории
не допускается. Все методы используют один build_packet/forced observer из
измеренной истории, TRAIN-calibration omega(L), backward linear velocity fit
на 21 точке, quadratic acceleration fit на 21 точке с clip ±1.5m/s².
Будущие реальные состояния и истинные v/rate/a не входят в контроллер.
Они используются только plant и метриками после принятия команды.

Тик .05s, горизонт 30 шагов/1.5s. Первый плановый command уже committed и
одинаково действует следующие .05s; выбранный plan[1] применяется на следующем
тике. Решение: шесть постоянных уровней для шагов 1..29, блоки 5/5/5/5/5/4.
Все методы получают один номинальный lag a_dot=(c-a)/.1. Положение и скорость
интегрируются точно под ZOH command; в угловую модель передаётся среднее
предсказанное ускорение за шаг, не true future acceleration.

## Одинаковый оптимизатор и цель

Сначала был только вычислительный toy smoke: обратное распространение по
нелинейной сети существенно дороже локального Jacobian; evaluation не читался.
Окончательный алгоритм до запусков: central FD по шести уровням, epsilon=.002,
13 векторизованных прогнозов. По Jacobian residual features строится convex
least-squares QP. 100 projected FISTA steps, Lipschitz по max eigenvalue Hessian;
trust region ±.2 вокруг warm levels и hard command box ±.8m/s². После QP
проверяются nonlinear objectives при fractions 0/.25/.5/1, выбирается лучший
для каждого эпизода, включая warm start. Нет смешения optimizer моментов или
градиентов между эпизодами. Warm sequence сдвигается по физическим шагам.
Один исходный нулевой план, одни параметры solver и objective у всех моделей.

Плановый cost: mean30[(x-target)²/.05² + theta²/.035² + .02v²/.2² + .05a²/.5²]
+ .1mean29[(delta c/.2)²] + 5 final_position_error²/.05²
+ 5 final_theta²/.035² + final_velocity²/.1².
State constraints не обеспечены: записываются нарушения rail[0,.5]m,
|theta|>15deg, |v|>.5m/s. Штраф slew не является жёстким rate constraint.
Nonfinite predictor/plant — сохранённая ошибка run, а не исключаемый эпизод.

## Независимый plant и новые условия

Нелинейный маятник: theta_ddot=-omega_p²sin(theta)-b_p a cos(theta)-d theta_dot+eta.
Lag a_dot=(gain*sign(c)*max(|c|-deadzone,0)-a)/tau. Истинные параметры скрыты.
Начальная x=.1m, цель .3m; до конца эпизода 6s. История 100Hz; RK4 substep .005s,
отдельно проверяется сходимость с .0025s. Контакты/обрезание состояния не вводятся;
нарушения сохраняются. Форсирование b_p=omega_p²/g или 1/L — объявленная stress
вариация, не идентифицированные параметры устройства.

Development: 6 эпизодов L=.32/.52/.72 × initial_angle2/6deg, nominal ratio1,
lag .1, gain1, damping.04, deadzone.01, effective force law, exact sensors.
Эти эпизоды только проверяют работоспособность. Настройки НЕ выбираются по ним;
после completion их SHA замораживаются до evaluation.

Evaluation: 24 контекста L=.27/.57/1.03 × frequency ratio .88/1.12 ×
lag .065/.18 × gain .8/1.2. Внутри каждой длины по паритетам трёх бинарных
факторов заранее назначены damping .015/.09, deadzone0/.025,
effective/geometric force law и initial_angle2/6deg. Назначение сбалансировано,
но это не полный независимый factorial всех nuisance factors; их отдельные
каузальные эффекты не оцениваются. К каждому контексту — exact/noisy-quantized
sensors × angular pulse absent/present .14rad/s² на [2.5,2.8)s. Всего 96 эпизодов
на controller, 288 evaluation +18 development. Noise seed917112026,
angle quantum2pi/4096, x quantum.0001m, noise SD .25quantum. Временные массивы
шумов и disturbances одинаковы между всеми controller для каждого case.
Вычисление батчами экономит время, но не меняет независимость эпизодов.

## Критерии, проверка и интерпретация

По каждому эпизоду: tracking ISE, sway ISE, tail RMS за последние2s, peak angle,
final position error, settled (финальное пребывание в |x-error|<.01m, |v|<.02m/s,
|theta|<1deg), effort integral a²dt, total command variation и нарушения.
Realized cost — integral того же normalized state/effort stage по всем6s плюс
та же terminal state cost. Slew отдельно. Это не физическая энергия в джоулях.
Все скалярные state metrics вычисляются на .05s control grid; raw100Hz states
также сохранены и позволяют проверять межтактовые пики отдельно.

Нет заранее объявленной единственной скалярной победы: для принятия замены
candidate должен улучшать средние sway и tracking, не ухудшать final error
более1%, не увеличивать нарушения и иметь больше половины парных cost wins.
Невыполнение любого условия сохраняется; никакой post-hoc смены candidate.
Common simulator не даёт гарантии hardware transfer или устойчивости.

До запуска: unit tests для точного nominal lag, angular ZOH, objective feature
identity, Jacobian против autograd, batch independence, bounds/pipeline, real
warmup, target poisoning и RK convergence. Код/config/design/tests коммитятся.
После: SHA всех inputs, повторная интеграция всех raw plant trajectories,
пересборка history-only inputs и replay выбранных forecast plans, парные
шумы/возмущения, проверка best-cost/fixed-first-step и независимые отчётные метрики.
Unit bootstrap — целый plant context (четыре stress cases), не correlated ticks.
Время включает сборку history/observer/solver, отдельно batch seconds и
amortized seconds/case. Они не считаются batch1 latency или WCET. Simulation
идёт офлайн с committed pipeline; фактические долгие вычисления не задерживают
plant дополнительно. Реальный deadline-aware режим пока не проверен.
