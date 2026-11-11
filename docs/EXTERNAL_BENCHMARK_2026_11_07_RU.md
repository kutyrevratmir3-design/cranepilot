# Независимый стенд: причинный прогноз по raw camera, 7 ноября 2026

Впервые проверен прогноз на записях **другого лабораторного крана**, ранее не
использованных в CranePilot. Протокол/код зафиксированы в `a52db9d182678407476c207bacee22efe69c6248` до
обучения. TRAIN12 records, VALIDATION4, закрытый TEST4 (17–20); все8методов
проверены на каждой записи. Source branch: research/broad-search-20261107.

## Основной результат

**Замороженный CranePilot не подтвердил перенос на этот стенд.** Он проигрывает
геометрическому baseline на всех4закрытых records и хуже даже zero/persistence
по общей MAE. В этом сравнении минимальная средняя ошибка у ARX residual:
0.037993°. Это обычная линейная residual model,
а не достижение новой нейронной архитектуры. ARX выигрывает у fitted oscillator
на всех4records, примерно на52.24% по среднему. Neural candidate тоже не спасает
перенос; выбор между frozen profiles по TEST не делался.

Frozen retained CranePilot MAE **0.895344°**, geometric **0.266333°**:
изменение ошибки **+236.18%**. Это проверка замороженных угловых весов
после заранее заданной геометрической/частотной адаптации и TRAIN-only centering,
не строгое zero-calibration применение полной native pipeline.

Locally fitted LSTM residual MAE **0.081588°**, изменение против geometric
**-69.37%**. У него есть12новых тренировочных записей, у frozen angular
weights их нет: этот разрыв нельзя приписывать только качеству архитектуры.
Это predictive error, не снижение sway реальным контроллером. Ни один default
или bundle weight не изменён; дальнейшего обучения после TEST не было.

## Все результаты закрытого TEST

| method | mae_deg | rmse_deg | p95_absolute_error_deg | track |
| --- | --- | --- | --- | --- |
| arx_residual | 0.0379932 | 0.0474696 | 0.0904891 | locally_fitted |
| cranepilot_candidate | 0.85272 | 1.07927 | 2.11163 | frozen_external_weights |
| cranepilot_retained | 0.895344 | 1.13129 | 2.19952 | frozen_external_weights |
| fitted_oscillator | 0.079552 | 0.0888701 | 0.143377 | locally_fitted |
| geometric | 0.266333 | 0.341439 | 0.660041 | unfitted_reference |
| lstm_residual | 0.0815882 | 0.0935884 | 0.161091 | locally_fitted |
| persistence | 0.376173 | 0.4662 | 0.869469 | unfitted_reference |
| zero | 0.399518 | 0.49606 | 0.983955 | unfitted_reference |

Среднее равновесно по четырём records, затем по семи horizons. RMSE и p95 здесь
средние record×horizon диагностик, НЕ единый pooled RMSE/p95 всех окон.

| record | zero | persistence | geometric | fitted_oscillator | arx_residual | lstm_residual | cranepilot_retained | cranepilot_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 17 | 0.159953 | 0.152734 | 0.248467 | 0.0712925 | 0.0313688 | 0.0662439 | 0.836626 | 0.81799 |
| 18 | 0.201523 | 0.154562 | 0.229121 | 0.102367 | 0.0332922 | 0.105026 | 0.790505 | 0.772665 |
| 19 | 0.620926 | 0.684919 | 0.248508 | 0.0893356 | 0.0469049 | 0.0984904 | 0.900773 | 0.849425 |
| 20 | 0.61567 | 0.512475 | 0.339234 | 0.0552132 | 0.040407 | 0.0565929 | 1.05347 | 0.970801 |

Сравнения и число выигранных RECORDS (не overlapping windows):

| method | reference | change_percent | record_wins | records | descriptive_low_percent | descriptive_high_percent |
| --- | --- | --- | --- | --- | --- | --- |
| zero | geometric | 50.0072 | 2 | 4 | -24.3123 | 111.801 |
| persistence | geometric | 41.2417 | 2 | 4 | -35.6565 | 126.68 |
| geometric | geometric | 0 | 0 | 4 | 0 | 0 |
| fitted_oscillator | geometric | -70.1306 | 4 | 4 | -79.8631 | -59.5661 |
| arx_residual | geometric | -85.7347 | 4 | 4 | -87.787 | -82.6876 |
| lstm_residual | geometric | -69.366 | 4 | 4 | -79.0988 | -57.3903 |
| cranepilot_retained | geometric | 236.175 | 0 | 4 | 216.879 | 256.034 |
| cranepilot_candidate | geometric | 220.171 | 0 | 4 | 195.556 | 239.613 |
| zero | fitted_oscillator | 402.21 | 0 | 4 | 108.152 | 755.487 |
| persistence | fitted_oscillator | 372.864 | 0 | 4 | 76.954 | 728.367 |
| geometric | fitted_oscillator | 234.791 | 0 | 4 | 147.318 | 396.602 |
| fitted_oscillator | fitted_oscillator | 0 | 0 | 4 | 0 | 0 |
| arx_residual | fitted_oscillator | -52.241 | 4 | 4 | -62.9747 | -35.5976 |
| lstm_residual | fitted_oscillator | 2.55959 | 1 | 4 | -3.94849 | 8.13323 |
| cranepilot_retained | fitted_oscillator | 1025.48 | 0 | 4 | 747.834 | 1492.78 |
| cranepilot_candidate | fitted_oscillator | 971.903 | 0 | 4 | 728.765 | 1375.37 |
| zero | arx_residual | 951.551 | 0 | 4 | 459.032 | 1316.3 |
| persistence | arx_residual | 890.105 | 0 | 4 | 375.243 | 1271.4 |
| geometric | arx_residual | 601 | 0 | 4 | 477.621 | 718.801 |
| fitted_oscillator | arx_residual | 109.385 | 0 | 4 | 55.2737 | 170.086 |
| arx_residual | arx_residual | 0 | 0 | 4 | 0 | 0 |
| lstm_residual | arx_residual | 114.744 | 0 | 4 | 59.5643 | 181.757 |
| cranepilot_retained | arx_residual | 2256.59 | 0 | 4 | 1956.53 | 2533.34 |
| cranepilot_candidate | arx_residual | 2144.4 | 0 | 4 | 1842.94 | 2432.71 |
| zero | lstm_residual | 389.676 | 0 | 4 | 111.057 | 750.326 |
| persistence | lstm_residual | 361.063 | 0 | 4 | 79.4229 | 672.097 |
| geometric | lstm_residual | 226.435 | 0 | 4 | 134.689 | 378.441 |
| fitted_oscillator | lstm_residual | -2.49571 | 3 | 4 | -7.52149 | 4.1108 |
| arx_residual | lstm_residual | -53.433 | 4 | 4 | -64.5085 | -37.3294 |
| lstm_residual | lstm_residual | 0 | 0 | 4 | 0 | 0 |
| cranepilot_retained | lstm_residual | 997.394 | 0 | 4 | 731.029 | 1438.71 |
| cranepilot_candidate | lstm_residual | 945.151 | 0 | 4 | 697.033 | 1356.23 |

Bootstrap resamples4whole-record clusters,2000draws,seed7112027,одни индексы для
всех методов. Интервалы описательные: четыре records одного троса/груза не
дают точную оценку переноса на популяцию кранов. Все horizons/records/negatives
сохранены. По TEST не выбирались коэффициенты, seed, epoch или preprocessing.

## Новый источник и обработка

[Khemakhem et al., arXiv2605.14944](https://arxiv.org/abs/2605.14944),
[публичный author repository](https://github.com/ikhemakhem/BehavioralTrajGenCrane),
commit `1aa26b5c358e885325f75cc0e89ef613ff4b26f3` от 2026-04-12T20:59:25Z. README упоминает раннюю2024версию,
preprintv1 опубликован14мая2026 и помечен under review; статус рецензирования
не предполагается. Записи физического поворотного крана с камерой nominal20Hz,
rope1m. Авторский validation_data разделён НАМИ на12/4/4 целых records по IDs;
их29processedTRAINrecords не использовались.

Одна TRAIN record1 и calibration были просмотрены для schema qualification до
preregistration, без model fitting/prediction. Остальные test17–20не разбирались
до lock выбора. Копирование/проверка двоичных файлов по Git SHA не было просмотром
углов. Download manifest содержит Git-blob SHA1, SHA256, URLs/размеры41MATфайла.
Сами внешние raw/source не републикуются: fetch script получает pinned source.

Author pipeline использует filtfilt и mean по ВСЕЙ записи. Мы независимо
преобразуем raw pixels в углы с фиксированной camera calibration на каждом frame;
не используем ни filtfilt, ни per-test mean, ни author vision_data, ни simulated
input-file yd. Масштаб1.7 взят из source preprocessing, не fitted по нашему TEST.
Один constant offset=-0.007774699rad — equal-record TRAIN mean.
Это калибровочный выбор, не доказательство физической точности камеры1.7.

Общий контракт:9native samples(theta2,boom encoder,issued profile) over nominal.4s;
30future prescribed velocity/acceleration steps. Будущие measured x/theta не
входы. Raw поля velocity/acceleration совпадают с prescribed input profile;
они не объявляются независимыми сенсорными измерениями. Углы radial theta1
не используются ни как вход, ни как выход — это tangential-only benchmark.

Frozen input adapter x=R*boom_angle, a=R*planned_acceleration — локальное
приближение trolley координаты. Coupling/centrifugal terms из rotary dynamics
не моделируются: перенос сложнее исходного trolley task. Omega=sqrt(g/1m) по
метаданным нового стенда; старая Inteco length calibration не переносится.
Нативный xr=0, старый NeuralActuator bypass. Из9nativepoints получены41samples
путём previous-sample hold; −10ms берёт−50ms, без будущей интерполяции. Новая
информация при upsampling не возникает. Это адаптированный angular submodel,
не перенос полной native-command CranePilot.

## Часы камеры и горизонт

Task по sample index, nominal dt=.05s. Истинные frame timestamps нерегулярны;
мы не подавали будущие timestamps модели. Поэтому1.5s — nominal30steps, а не
подтверждённый exact wall-clock horizon. Длительности реальных окон:

| stage | record | samples | packets | raw_timestamp_span_s | nominal_span_s | maximum_frame_interval_s | minimum_frame_interval_s | median_frame_interval_s | no_future_angle_filter | whole_record_test_mean_used | input_file_simulated_yd_ignored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation | 13 | 500 | 231 | 25.727 | 24.95 | 0.065 | 0.047 | 0.048 | True | False | True |
| validation | 14 | 500 | 231 | 25.758 | 24.95 | 0.065 | 0.046 | 0.048 | True | False | True |
| validation | 15 | 500 | 231 | 25.744 | 24.95 | 0.065 | 0.046 | 0.048 | True | False | True |
| validation | 16 | 500 | 231 | 25.711 | 24.95 | 0.066 | 0.046 | 0.048 | True | False | True |
| test | 17 | 500 | 231 | 25.727 | 24.95 | 0.065 | 0.046 | 0.048 | True | False | True |
| test | 18 | 500 | 231 | 25.679 | 24.95 | 0.066 | 0.046 | 0.048 | True | False | True |
| test | 19 | 500 | 231 | 25.696 | 24.95 | 0.065 | 0.046 | 0.048 | True | False | True |
| test | 20 | 500 | 231 | 25.727 | 24.95 | 0.065 | 0.047 | 0.048 | True | False | True |

Range/median каждого фактического horizon находятся в all_metrics.csv (для
каждого record, без поиска лучших сдвигов). Physical command-camera timing
проверен только через supplied index alignment, не независимой аппаратурой.

## Диагностика причин, без изменения предикторов

Средний TRAIN camera interval 0.051794756s
вместо nominal.05s. Fitted omega 3.265578
на nominal clock после ДИАГНОСТИЧЕСКОГО пересчёта по TRAIN ratio соответствует
3.152421rad/s;
geometric omega=3.132092rad/s. Это позволяет
предположить, что часть fitted frequency shift связана с timebase, а не только
с неизвестной механикой. Скорректированные прогнозы здесь НЕ строились и не
выбирались; нельзя объявлять clock drift доказанной единственной причиной.

На отдельном synthetic zero-history/zero-command packet frozen retained выдаёт
до 0.360817°, candidate до
0.323208°, geometric ровно0.
Это posthoc проверка математического equilibrium, не аппаратная ошибка на реальном
test и не количественное разложение его MAE. Она показывает, что текущая learned
correction не гарантирует нулевой отклик при нулевом входе. Camera centering,
upsampling и геометрический domain adapter также меняют распределение входов.
На этом основании ни один weight/offset/preprocessing не исправлялся после TEST.

## Базовые методы и связь с публикациями

Geometric — forced observer+ZOH oscillator с геометрической частотой. PEM grey-box
fitted_oscillator имеет omega,damping,gain, fitted по TRAIN multi-step output error;
initial state профилируется по9pastangles/issuedacceleration в каждом origin.
Три train-only startingpoints, bounds в config; validation не выбирает параметры.
Финальные параметры [3.265577988448446, 0.05732448261272044, 0.9949534546912502].

ARX residual — direct linear command-prefix prediction поверх geometric, ridge
alpha=0.0001, выбранный на VALIDATION из3значений. Constant length
в этом dataset не позволяет проверить LPV scheduling; метод не назван SMILE.
LSTM residual — один32unit encoder/decoder,3seeds,zero-output epoch−1 baseline,
early stop по VALIDATION, equal-record seven-horizon MSE training. Selection:

| seed | selected_epoch | validation_mae_deg | fit_seconds |
| --- | --- | --- | --- |
| 0 | 83 | 0.0737955 | 15.6712 |
| 1 | 95 | 0.0729548 | 14.9216 |
| 2 | 94 | 0.0665957 | 15.002 |

[Zhou2025 PINN](https://doi.org/10.1016/j.aei.2025.103269) и
[CoDIT2026 LPV](https://doi.org/10.1109/CoDIT70676.2026.11631165) подтверждены по
аннотациям; полного проверенного code/loss/architecture нет. Наши результаты
НЕ репликация их published scores. ARX/multistep identification мотивированы
[Kusznir/Smoczek2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8749516/),
а сравнительный набор recurrent/classical methods — также
[Kusznir et al.2024](https://www.mdpi.com/2076-3417/14/8/3492). Их genetic search
не повторяется. Методика author behavioral trajectory optimization в этом этапе
не реализована: goal — общий короткоисторический причинный прогноз, их task/input
history и acausal offline filtering отличаются. Нельзя сообщать, что мы обошли
эти статьи по их собственному протоколу.

## Проверки и воспроизведение

179unit tests PASS: geometry scalar/vector equivalence,source corruption,partition
seals,futurepixelprefix poisoning,known-system PEM recovery,ARX/LSTM commandprefix,
frozen target invariance и unchangedweights. Все 14784
validation+TEST forecasts replayed из сохранённых checkpoints/оригинальных raw,
а не из saved estimates. Семь metric horizons всех records пересчитаны независимо.
SHA guarded source/data/checkpoints/selection проверены, все
1281 original main+APHYNITY файлов сохранены.
Raw original cache excluded from public artifacts. APHYNITY без изменений.

Новый запуск: fetch_external_crane,run_external_benchmark --stage train, затем
--stage evaluate; verify_external_benchmark,report_external_benchmark. Все через
restored main Python3.12launcher. Scripts запрещают overwrite. Для повторения
benchmark нужно новое --output и согласованные RUN/OUT paths report/verifier.
Artifacts/runs копируется в results/external_benchmark_20261107/frozen_run,
включая source manifest, все fitted weights и численные predictions. Сами raw
сигналы скачиваются заново по pinned manifest, не вложены в evidence ZIP/Git.

## Что это доказывает и что остаётся

Есть действительно новый для проекта стенд и закрытая record-level проверка
прогноза. Есть общие причинные inputs и сильные locally fitted comparators,
все отрицательные переносы сохраняются. Нет нового hardware control опыта,
нет доказательства стабильности/SOTA, нет переносимости на новые длины/грузы
по одному fixed1m dataset. Frozen/local training access различается явно.
Native xr semantics старого стенда по-прежнему не установлены.
После TEST этот набор уже нельзя снова называть нетронутым подтверждением
следующей настроенной модели. Любая адаптация CranePilot потребует нового
confirmation set или явной маркировки retrospective ablation.
