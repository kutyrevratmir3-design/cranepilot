# CranePilot: состояние исследования на 7 ноября 2026

Актуальный завершённый эксперимент: **external_benchmark_20261107**.

Получен новый для проекта публичный источник реальных raw-camera записей
другого поворотного крана (Khemakhem et al., pinned commit1aa26b5). Whole-record
TRAIN12/VALIDATION4/закрытыйTEST4, протокол **a52db9d** до обучения/просмотраTEST.
Причинное преобразование raw pixels; author filtfilt/whole-test centering исключены.
Общие9samples over nominal.4s и30future prescribed input steps. Оценивается
только tangential sway при fixed rope1m, не hardware controller.

| Метод | TEST MAE, ° | Доступ к новому TRAIN |
|---|---:|---|
| ARX residual | **0.037993** | да |
| Fitted damped oscillator | 0.079552 | да |
| LSTM residual,3seeds | 0.081588 | да |
| Geometric oscillator | 0.266333 | только camera offset |
| Persistence | 0.376173 | только camera offset |
| Zero | 0.399518 | только camera offset |
| Frozen CranePilot candidate | 0.852720 | только camera offset |
| Frozen CranePilot retained | **0.895344** | только camera offset |

**Перенос текущих угловых весов НЕ подтверждён**: retained проиграл geometric
на всех4TESTrecords, mean error +236.18%. ARX residual лучше fitted oscillator
на52.24%,4/4records. Нет доказательства превосходства CranePilot над сильными
классическими/нейронными сравнениями. При этом locally fitted и frozen методы
имеют разный доступ к новой обучающей выборке: это не изолированное сравнение
архитектур. Все результаты и отрицательный перенос сохранены без tuning послеTEST.

Это отдельный domain adapter: omega из metadataновогостенда, x=R*boom_angle,
a=R*planned_acceleration; native xr0, actor bypass,20→100Hz previous-samplehold
без новой информации. Полная native pipeline не проверена. Одна TRAIN-only
camera-centering constant применена ко всем. Actual camera time нерегулярен:
nominal30steps не ровно1.5s; futuretimestamps не input. TRAIN meaninterval.051795s;
часть fitted frequencyshift может объясняться timebase (diagnostic inference).
Syntheticzero-input проверка дала ненулевой frozen отклик до.361° — posthoc,
не аппаратное измерение и не доказанная единственная причина ошибки.

179unit tests PASS. Все14784validation+TEST forecasts (1848packets×8methods)
replayed EXACT0; метрики пересчитаны независимо. SHA всех данных/источников/
checkpoints/selection и1281originalmain+APHYNITYfiles проверены. Originalexternal
MAT/source остаются у авторов/вignoredcache; публикуются owncode,weights,metrics,
derivedpredictions и pinned downloadmanifest. APHYNITY/defaultbundle unchanged.

[Отчёт](docs/EXTERNAL_BENCHMARK_2026_11_07_RU.md),
[протокол](docs/EXTERNAL_BENCHMARK_DESIGN_20261107.md),
[все результаты](results/external_benchmark_20261107/).

Следующая работа должна отделить timebase/domain adapter и локальное обучение
от архитектуры: causal clock handling, zero-equilibrium constraint и same-data
reference comparison. Текущие4TESTrecords уже использованы: следующая настроенная
модель на них будет retrospective ablation; новое независимое подтверждение
потребует свежих records. Нельзя заявлять SOTA/hardware/nativecontrol успех.

Предыдущий завершённый эксперимент: **structural_transfer_20261107**.

918 новых эпизодов, 9 регуляторов, шесть семейств привода: первый порядок,
скрытая задержка, два лага, трение, смена gain и комбинированный случай.
Разворот цели в 4s, длительность 8s. Протокол зарегистрирован в **540477f**,
настройки после development не менялись. APHYNITY и neural weights без изменений.

| Вариант против nominal | Cost | Sway ISE | Tracking ISE | Final error | Cost wins |
|---|---:|---:|---:|---:|---:|
| Static force+actuator | −74.90% | −40.63% | −8.57% | −52.82% | 78/96 |
| Frozen adaptive+force+observer | −81.33% | −65.59% | −4.53% | −41.11% | 76/96 |

Пять из восьми альтернатив прошли global и все шесть family gates; новая и
прежняя квалификация adaptive observer дали одинаковые trajectories. Против
сильного static варианта adaptive cost −25.62%, но только **25/96 cost wins**,
tracking +4.41%, final error +24.83%; settling 36/96 против 52/96 static.
Польза адаптации сосредоточена в тяжёлых delay/combined случаях, доминирования нет.
Raw angle>15deg: adaptive 8/96, static и nominal 12/96; rail/speed нарушений нет.

Fixed-parameter observer с той же 2s памятью дал cost всего −0.28% против
nominal и tracking +1.01%. С учётом forcing bank адаптация относительно fixed
observer дала cost −70.63%, но tracking +0.94%. Длинная история сама по себе не
объясняет основной выигрыш; comparisons остаются policy-dependent.

172 unit tests PASS; все 918 plant trajectories, identifier/observer states и
146880 chosen forecasts/costs replayed EXACTLY. RK halfstep max<4.8e−7.
Все 1281 original main/APHYNITY файлов сохранены. Независимо пересчитан cost из
latent raw traces с учётом net acceleration/friction. Warm batch1 решения ~5–9ms;
не WCET guarantee. Прежние 122 EOL-only изменения остаются отдельно.

[Отчёт](docs/STRUCTURAL_TRANSFER_2026_11_07_RU.md),
[протокол](docs/STRUCTURAL_TRANSFER_DESIGN_20261107.md),
[полные данные](results/structural_transfer_20261107/).

Это synthetic acceleration-command ablation, native xr zero и trained actuator
bypass. Hardware/native/SOTA/stability успех не заявлен. Ни один default не
заменён. Следующий метод должен уменьшить потери на обычных случаях относительно
static force_actuator и улучшить слежение при задержке: явная delay/cascade model
bank либо причинное blending static/adaptive на новых preregistered contexts.
Текущие 96 cases повторно не использовать для tuning и подтверждающей оценки.

Предыдущий завершённый эксперимент: **adaptive_actuator_20261107**.

Выполнены918новых closed-loop эпизодов:9методов ×(6development+96evaluation).
Протокол/код зарегистрированы до испытаний в522d01d. Neural weights неизменны.
Семь из восьми альтернатив прошли все заранее заданные gates относительно
nominal на этом новом симуляционном наборе.

| Метод | Cost | Sway ISE | Tracking ISE | Final error | Cost wins |
|---|---:|---:|---:|---:|---:|
| Static force+actuator bank | −62.56% | −72.07% | −9.80% | −38.95% | 78/96 |
| Online actuator+force+observer | −66.55% | −76.40% | −5.57% | −58.74% | 76/96 |

У adaptive_force_observer settled84/96 против62/96 nominal, angle>15deg
0/96 против8/96. Он точнее по sway/cost, но static force_actuator точнее по
tracking. Нельзя объявлять один метод доминирующим по всем целям.
Идентификатор использует только прошлые x и собственные applied commands,
не true v/a/parameters. Явно добавлена память2s для actuator; angular history
остаётся41samples/.4s. Observer меняет только trolley v/a.

164tests PASS;918траекторий, online identifier states и прогнозы110160решений
воспроизведены точно. Сохранены1281original main/APHYNITY файлов.
Полное warm batch1 решение adaptive_force_observer median8.63ms в этом запуске;
это не WCET/scheduling гарантия. Изменённых neural weights нет.

Это acceleration-command simulation с zero-xr features. Структура actuator
близка к fitted first-order model; победа ещё требует проверки delay/двух lag/
friction/time-varying gain, fixed-parameter observer с той же2s памятью,
а затем native xr qualification. Hardware/SOTA/native-controller успех не заявлен.

[Отчёт](docs/ADAPTIVE_ACTUATOR_2026_11_07_RU.md),
[протокол](docs/ADAPTIVE_ACTUATOR_DESIGN_20261107.md),
[все fitted traces и результаты](results/adaptive_actuator_20261107/).

Предыдущий завершённый эксперимент: **scenario_control_20261107**.

Проверены6стратегий на612новых синтетических closed-loop эпизодах:
nominal physical, frequency, joint uncertainty, spread penalty, frozen neural
residual и joint+residual. Параметры зафиксированы до запуска в de5d678.

На новом наборе96условий относительно nominal:

| Вариант | Realized cost | Sway ISE | Tracking ISE | Final error | Cost wins |
|---|---:|---:|---:|---:|---:|
| Joint+spread | −6.78% | −36.11% | +1.74% | −5.31% | 44/96 |
| Joint+hybrid | −12.80% | −36.99% | +6.77% | +2.00% | 40/96 |

У обоих0из96 превышений15deg против8из96 у nominal; это наблюдение конечного
набора, не гарантия безопасности. Ни один вариант не прошёл все gates: tracking
выросла, cost выигрывается не в большинстве пар. Средняя выгода сосредоточена
в части тяжёлых случаев. Frequency-only и nominal+residual ухудшили общий cost.
Joint одновременно меняет actuator и forcing uncertainty; отдельный вклад
каждого этот опыт не устанавливает. Новая универсальная замена не принята.

157tests PASS;612trajectories и forecasts73440решений воспроизведены точно.
Сохранены1281исходный main/APHYNITY файл. Default neural weights не менялись.
Это acceleration-command ablation с zero-xr features, не native-command пилот.

[Новый отчёт](docs/SCENARIO_CONTROL_2026_11_07_RU.md),
[протокол](docs/SCENARIO_CONTROL_DESIGN_20261107.md),
[траектории и результаты](results/scenario_control_20261107/).
Следующие гипотезы: разделить forcing/actuator ablations и уменьшить tracking
tradeoff на новом TRAIN/development наборе. Не выбирать параметры по этим96cases.

Предыдущий завершённый эксперимент: **angular_control_20261107**.

Впервые выполнен отдельный замкнутый тест углового прогноза: 18 development
и 288 evaluation эпизодов, по96 условий на три регулятора. Веса заморожены,
native xr features нулевые; NeuralActuator обходится, nominal lag общий.
Это новая acceleration-command ablation, не полный native-command пилот.

Новый adapter против прежней сети: sway ISE −11.9%, tracking ISE +2.85%,
final position error +30.05%; замена не принята. У физического baseline
меньше средняя realized task cost (74.54 против89.91 у adapter),
64/96 settled эпизода против27/96, нет rail violations против4/96.

Проверены150unit tests, повторно интегрированы306эпизодов, воспроизведены
прогнозы36720решений. Все1281исходныхфайлов сохранены. Batch-one decision
median: physical3.55ms, retained19.34ms, candidate36.27ms в этом измерении;
это не hardware/WCET гарантия.

На общей истории физического controller post-hoc audit показал, что у нового
adapter выше абсолютная ошибка command Jacobian и хуже cosine, хотя средняя
ошибка counterfactual forecast ниже прежней сети. Следующие гипотезы —
control-oriented response/Jacobian loss на новом TRAIN-наборе либо физический
command Jacobian с ограниченной learned residual/observer. Они ещё не выполнены.

[Полный closed-loop отчёт](docs/ANGULAR_CONTROL_2026_11_07_RU.md),
[протокол](docs/ANGULAR_CONTROL_DESIGN_20261107.md),
[траектории и результаты](results/angular_control_20261107/).

Предыдущий этап прогноза: **structured_models_20261107**.
Основной проект и APHYNITY остаются отдельными. Старые отчёты сохранены как
исторические записи, а не переписаны под новый результат.

## Что изменилось

После 90 fine-tune запусков проверены восемь новых архитектур, 72 обучения:
малые нулевые residual adapters над замороженной soft-базой и глобальные
латентные модели с сокращающимся автономным переходом. Протокол и код
зафиксированы до обучения в `b2183ab`; выбор по validation заморожен до
evaluation и симуляции.

| Режим | Прежний выбранный прогноз, MAE ° | Новый заранее выбранный кандидат, MAE ° | Изменение |
|---|---:|---:|---:|
| Primary, extrapolation, все окна | 0.490598 | 0.473621 | −3.46% |
| Primary, активное движение | 1.036838 | 0.954652 | −7.93% |
| Long transfer, extrapolation | 0.512641 | 0.501157 | −2.24% |
| Short transfer, extrapolation | 0.383598 | 0.383598 | прежний LPV сохранён |
| Primary, изменённая нагрузка, все окна | 0.548625 | 0.559697 | **+2.02%** |
| Primary, изменённая нагрузка, активное движение | 1.276844 | 1.051392 | −17.66% |

Primary-кандидат — adapter_direct8 с 321 новым параметром; long-кандидат —
latent_residual6. Все четыре primary extrapolation-сценария улучшены, но
load-all ухудшение превышает заранее установленный допуск 1%. Поэтому новый
локальный набор не принят как общая замена. Common сохранил `previous_local`;
прохождение gate этим контролем означает сохранение прежнего результата.
Long-кандидат слегка хуже soft по общей MAE (0.501157 против 0.500935),
а его ошибка последних десяти шагов выше прежнего incumbent на 5.64%.
Улучшение относительно выбранного контроля не означает выигрыша во всех метриках.

`primary`, `short_transfer`, `long_transfer` — разные исследовательские splits.
Это не готовое правило автоматического выбора модели по текущей длине троса.

## Проверки и воспроизведение

Полный replay: **678 сравнений, 17496 пакетов, максимальная разница 0°**.
Проверены 72 state_dict, 54 неизменные базы, 36 сокращающихся латентных
операторов и 9 точных начальных fallback. Все 1281 файла исходного snapshot
сохранены. 3456 новых симуляционных случаев относятся к прежнему семейству
симулятора, а не к независимому реальному оборудованию.

- [Полный отчёт и границы выводов](docs/STRUCTURED_MODELS_2026_11_07_RU.md).
- [Протокол до обучения](docs/STRUCTURED_MODELS_DESIGN_20261107.md).
- [Первичные работы о структурированных моделях](docs/STRUCTURED_MODELS_LITERATURE_20261107.md).
- [Все таблицы, проверки и локальный архив](results/structured_models_20261107/).
- [Переносимый комплект весов и команды запуска](docs/RESEARCH_BUNDLE_20261107.md).
- [Аудит подключения прогноза к управлению](docs/CONTROL_INTEGRATION_AUDIT_20261107.md).

В переносимом комплекте `retained` — профиль по умолчанию; `candidate` явно
экспериментальный. Он не требует ancestral run trees или исходного датасета
для прогнозирования, но требует код и Python-зависимости основной версии.
Большие полные evidence ZIP остаются локально; небольшой комплект весов,
код, конфигурации, таблицы и проверки сохраняются в Git.

## Следующий исследовательский шаг

Прирост MAE не доказывает улучшение контроллера: существующий LinearMPC не
вызывает этот прогноз. Его вход — ускорение, тогда как neural actuator
получает нативный `xr` с пока не установленным преобразованием в действие.
На TRAIN один и тот же постоянный `xr` сопровождается разными остановившимися
положениями тележки. Это исключает наивный position-servo как установленную
модель привода, но не определяет истинные единицы `xr`.

Выполнен отдельный closed-loop тест углового прогноза
с командами ускорения, общим наблюдателем и независимым plant/actuator lag.
Это новый контракт angular-ablation; полноценный native-command пилот
потребует согласования `xr`, привода и временных индексов. Подробный протокол
и варианты квалификации описаны в аудите интеграции. Acceleration ablation
уже выполнена; native-command closed loop остаётся следующим отдельным опытом.

## GitHub

Пользователь разрешил публикацию и отдельно подтвердил её состав после
автоматической блокировки первой попытки. Ветка
[`research/broad-search-20261107`](https://github.com/kutyrevratmir3-design/cranepilot/tree/research/broad-search-20261107)
сохраняет локальную историю исследований. `main` не изменяется. Обе истории
имеют разные корни, поэтому слияние не выполнялось. Детали — в
[области публикации](docs/GITHUB_PUBLICATION_SCOPE_20261107.md).

Все основные оценочные splits многократно использовались ранее. Эти выводы
ретроспективны; независимое испытание, SOTA и hardware improvement не заявлены.

## Дополнение: равный доступ и аудит новых данных

После внешнего отрицательного transfer результата модели обучены на одинаковых
12 TRAIN записях rotary-camera набора. Сильный ARX получил более широкую сетку
регуляризации и direct/residual варианты. StructuredNARX уменьшил validation
MAE на4.24%, общая импульсная характеристика на4.67%; порог10% не пройден.
Дообученные neural варианты существенно хуже ARX. Старые TEST17–20 уже открыты,
их новая оценка только ретроспективная. Все30 matched fits и18 shared fits,
исходы, replay и архивы сохранены. 187 tests PASS перед shared fitting;
1281 исходных main/APHYNITY файлов неизменны.

Ссылки: [равный доступ](docs/MATCHED_ADAPTATION_2026_11_07_RU.md),
[общая характеристика](docs/SHARED_RESPONSE_2026_11_07_RU.md),
[доступ к данным авторов](docs/SAME_RIG_DATA_ACCESS_20261107.md).
Пользователь сообщил, что может запросить у авторов новые raw logs; сообщений
авторам от агента не отправлялось.

Уточнение прежней общей фразы о33 записях: metadata-only audit сохранённых
артефактов не нашёл fitting/validation/score использования7сценариев
9,13,21,22,23,32,33. Набор целиком ранее импортировался, неизвестное незаписанное
использование исключить нельзя. Это возможные **ранее не оценивавшиеся условия**,
не новые независимые измерения. Перед modal этапом они были закрыты. Отдельный протокол
[MODAL_HISTORY_DESIGN](docs/MODAL_HISTORY_DESIGN_20261107.md) определяет условный
тест прогноза остаточных колебаний с одинаковой4sисторией против усиленного ARX
и локальной авторегрессии. Сначала development gate; лишь затем возможен
однократный holdout. Активное управление и прогноз во время подъёма не заявляются.

## Итог первого modal этапа

Development−26.38%,4/5case wins, затем зафиксированная проверка7условий:
modal .261421° vs initial full-inputARX2.338230°,6/7wins. Дополнительный
ретроспективный challenge240простых/invariantARXfits показал существенно более
сильную базу .492306°. Консервативный выигрыш теперь−46.90%,6/7wins,
95%описательный whole-scenario interval[−85.05%,+15.60%] включает0. Scenario21
проигран(.7000° vs .3535°). Цель уверенного независимого преимущества **не закрыта**.

Все33публичных сценария теперь открыты; новая оценка на них retrospective.
200tests PASS,37704modal conditional forecasts replayточно,120+240ARXfits
проверены, все1281оригинальных файла сохранены. Новая задача использует4sисторию
при settled-input gate; это не исходный .4sactive-input forecast или hardware.
Portable modal bundle12files107738bytes: прогноз и более сильный ARX без
обучения/данных/старыхrun trees. Kernel latency median11.3–11.6msнаэтомMac.

Главный новый отчёт: [MODAL_CAMPAIGN](docs/MODAL_CAMPAIGN_2026_11_07_RU.md).
[Комплект и API](docs/MODAL_BUNDLE_20261107.md),
[следующий шаг с данными авторов](docs/AUTHOR_DATA_INTAKE_20261107_RU.md).
Пользователь может попросить авторов; агент сообщений не отправлял.

## Текущий этап: выбор по прошлой ошибке прогноза

Новая причинная проверка использует4sисторию: изпервых3s предсказывает
последнюю уже наблюдённую1s и задаёт вес междуmodal иARX. Будущие после0s
targets не участвуют. Одинsoftоператор выбран только по прежнимvalidation:
−30.14%MAE,5/5cases. Reference дополнительно выбирает3s/4s историю поVAL.

На7ранее открытых условиях .189779° vs .478509° уусиленногоARX:
−60.34%,6/7wins, descriptive paired95%interval[−84.03%,−23.25%].
Но это**ретроспективная разработка**, не новая независимая проверка.
Scenario21 всёещё хужеARX(.403819°vs.375804°), прежнийmodal=.700020°.
Общая цель остаётся открытой до подходящих новых сырых измерений.

206tests PASS,80700forecastcomparisons replayточно, portableAPI9302окна
во17case-recordsсовпали точно. Все1281оригинальные main/APHYNITY файла,
старыеbundles/defaults сохранены. Новый отдельный bundle19files182278bytes,
безtrainingdata/ancestralruns; observedmedian21–22ms/packet.
Zero-inputbias смеси≤.035304°на15фиксированных probes; точный equilibrium
или closed-loop guarantee не заявляется. Ошибкаimportвпервомverifierсохранена,
исправлен толькоverifier, безпереобучения.

Новыйотчёт: [causal backtest gate](docs/MODAL_BACKTEST_GATE_2026_11_07_RU.md),
[bundle/API](docs/RELIABILITY_BUNDLE_20261107.md),
[диагностика21](docs/MODAL_FAILURE_DIAGNOSIS_20261107_RU.md),
[дополнительныйпоискданных](docs/DATA_ACCESS_FOLLOWUP_20261107.md).
IRVоказалсятаблицейfreq/zeta/peakacceleration, Haarhoffрепозиторийописывает
симуляции. Новые подходящие независимые raw records не получены.
