# Разделение неопределённости и online actuator identification

7 ноября2026. Исходный HEAD ec9be1f. Новый run adaptive_actuator_20261107.
Старые96scenario cases используются только как источник гипотез. Новый config,
код и тесты коммитятся до запусков. Neural weights не меняются; APHYNITY отдельно.

## Вопрос и первоисточники

Joint uncertainty снижала sway и средний cost, но повышала tracking и не
выигрывала большинство пар. В ней одновременно менялись actuator и forcing.
Здесь разделяем эти факторы, затем проверяем идентификацию actuator по известным
прошлым applied commands и измеренным x. Отдельно выделяем текущую оценку v/a.

Åström/Wittenmark, On Self-Tuning Regulators (1973), DOI
10.1016/0005-1098(73)90073-3: совмещаются LS estimator и controller, использующий
оценённую модель. Запись опубликованной версии:
https://portal.research.lu.se/en/publications/on-self-tuning-regulators-2/ . Наш profiled finite-window lag fit и MPC отличаются; результаты
о convergence/stability этой работы не переносятся. Связанный technical report1972(TFRT-7017), не journal PDF1973:
https://lup.lub.lu.se/search/files/8888379/7017.pdf .
Gevers, Identification for Control (2005), авторский обзор:
https://perso.uclouvain.be/michel.gevers/PublisMig/P78.pdf — назначение модели и
информативность эксперимента существенны для управления. Наш порог возбуждения
является инженерным фильтром, а не доказательством persistent excitation.

## Новый информационный контракт

Угловая модель/forced observer всё ещё получают41past samples100Hz (.4s).
Все controllers дополнительно имеют доступ к общему буферу2s: измеренная x
на50ms сетке и собственные уже применённые acceleration commands. Static
методы этим буфером не оценивают параметры. Это явное расширение memory,
не чистое изменение архитектуры прежнего41-sample forecaster.

В окно в моментt входят позиции доt включительно и команды интервалов,
заканчивающихся не позжеt. Committed command следующего интервала НЕ участвует
в fit. Никакие true v/a/actuator parameters, будущие измерения или case labels
не передаются estimator. Ground truth применяется только plant и метриками.
Native xr features0, NeuralActuator bypass; задача по-прежнему acceleration ablation.

## Девять заранее фиксированных controllers

1. nominal: frequency1, nominal(tau=.1,gain1), effective force law.
2. force_only: то же, два закона b=omega²/g и b=1/L, средний cost.
3. actuator_only: три прежние пары(.07,.8)/(.1,1)/(.18,1.2), effective law.
4. force_actuator: 3пары×2laws, frequency1.
5. joint_previous: прежние3частоты×3пары×2laws,18scenarios.
6. adaptive_params: identified lag/gain point, common backward v/a estimates.
7. adaptive_force: identified point×2force laws, common v/a.
8. adaptive_observer: identified point и fitted trolley endpoint v/a.
9. adaptive_force_observer: identified point×2laws и fitted endpoint v/a.

Все используют общие nominal q/p и measured current x. Даже observer-варианты
не меняют angular state observer; меняются только trolley initial v/a.
Параметры и state estimates фиксированы внутри каждого MPC решения.

## Идентификация и observer

Модель a_dot=(K c-a)/tau, x_dot=v, v_dot=a. Для каждого tau в сетке
.05/.065/.08/.1/.125/.16/.2/.25s интегрируется unit-gain forced x response.
Свободная initial x/v/a профилируется на basis[1,t,tau*(t-tau*(1-exp(-t/tau)))].
После проекции forced response и observed x на дополнение nuisance basis
K оценивается scalar ridge LS с prior1, lambda=1e-7m², clip[.5,1.6].
Выбирается минимум SSE+lambda*(K-1)²+1e-9*log(tau/.1)².

Окно до40интервалов/41позиции (2s); минимум21позиция. Порог RMS residualized
forced x >=.0004m; fit RMSE <=.001m. Обновление раз в4control ticks (.2s),
EMA .25 для tau/gain, иначе prior/последняя оценка. Начальные tau=.1,gain1.
Нет принудительного возбуждения или tuning на evaluation. Gain здесь может
быть эффективным при unmodelled deadzone; ошибка против true gain не доказывает
неправильность такого effective fit. Feedback/noise могут смещать LS.

Observer-варианты на каждом тике профилируют nuisance state при filtered tau/K
на доступном окне и получают endpoint v/a аналитически. После первого valid
fit они используются при текущем fit RMSE<=.001m. Clips v±.6m/s,a±1.5m/s².
Current measured x остаётся общей. До qualification используется прежний
backward velocity/acceleration fit. Повторный replay обязан восстановить эти
оценки последовательно из истории, а не читать saved fitted parameters.

## Общие controller/plant и новый набор

Те же horizon30/1.5s,6blocks,committed delay.05s, box±.8, trust±.2,
unit central affine probes и200FISTAsteps. Objective weights прежние.
MPC weights не обучаются. Нет state-constraint/stability certificate.
True plant остаётся независимым: nonlinear angle, hidden lag/gain/deadzone,
hidden frequency ratio, disturbances. RK4 raw.01s/substep.005s; halfstep audit.

Development: L=.34/.54/.74×angle2/6deg,6cases/method, nominal plant, exact sensors.
Qualification без tuning; SHA замораживаются до evaluation.
Evaluation: L=.25/.62/1.07×omega ratio.87/1.13×tau.06/.19×gain.78/1.22:
24contexts × exact/noisy × pulse absent/present =96cases/controller.
Balanced parity nuisance damping.02/.08, deadzone0/.02, force effective/geometric,
initial_angle3/5deg. Новый seed1207112026, pulse.16rad/s² на[2.45,2.85)s.
Итого864evaluation+54development=918episodes. Nuisance не независимыйfactorial;
сравнения strategy парные, но adaptive estimates endogenous к policy.

## Принятие, проверка и пределы

Reference nominal НАЭТОМновомgrid. Gates для всех8альтернатив: mean realized
cost/sway/tracking ниже, final error<=1.01×reference, raw100Hz violations не выше,
cost wins>48/96. Не менять gates по результатам. Все результаты публикуются.
Отдельные точные ablations: force_only vs nominal; actuator_only vs nominal;
force_actuator vs force_only/actuator_only; joint_previous vs force_actuator;
adaptive_observer vs adaptive_params и их force-варианты. Не приписывать observer
выигрыш только параметрам или наоборот.

Unit tests: exact lag/gain under excited synthetic commands, unknown initial
state elimination, endpoint reconstruction, translation invariance, no-excitation
fallback, schedule/EMA/clips, factorial banks, physical Jacobian, independent
cases и pipeline. После: SHA,1281originals, полный raw plant/measurement replay,
последовательный повтор всех identifier states, всех forecasts/costs, halfstep
convergence, независимые итоговые метрики. Full replay не перезапускает optimizer.

Сохраняются parameter/error/use traces, raw100Hz episodes и все violations.
Расчёт error τ/K/v/a использует true plant только после решения. Finite-window
LS не является RLS/IV/dual control и не имеет заявленной consistency guarantee.
Isolated batch1 timing включает window assembly, online update, observer и solver;
между повторами восстанавливается estimator state. Simulation офлайн, wall-clock
latency сверх committed tick не моделируется. Нет hardware/native/real-time claim.
