# Датированная компенсация сенсорной задержки

Протокол/исходники доfitting: `5d1ba2a2dbde505c3e2392ba116e5764c3c76c92`. Все33исходные записи уже открыты;
это ретроспективный synthetic-age/noise эксперимент. Age дан каждому методу
из условныхtimestamps; неизвестная задержка не оценивалась по будущим ошибкам.

## Контракт и причинность

401измерениеугла заканчивается на current_time−age;401измерение L/support/
native-reference — наcurrent_time. Все100Hz. Началоorigins4.25s,stride1s:
длямаксимальногоage150ms существует настоящая предыстория, безpadding.
Не сравнивать эти matchedчисла напрямую с прежним .25s/padded50msstress.
Каждый считавшийсяorigin сохранён, ноль target/amplitude-basedисключений.
Цели семь раз после текущего момента(.10..1.50s), не послеsensorclock.

Наблюдённыйугол сначала переставляется на свойtimestamp. Только недостающая
новейшаяheadчасть предсказывается: hold, linear.10/.25s,quadratic.25s,
free-modal3s,forced3sfixed/adapt. Forcedobserver подгоняет состояние на датированных
углах и синхронном прошломsupport, затем распространяет его черезgap по
support, уже измеренному к currenttime. Эти supportданные не из будущего
относительноforecastorigin. Нелинейныеnuisanceforceканалы и effectivefrequency
на короткомgap удерживаются; это ограниченная физическая аппроксимация.

Предсказаннаяheadчасть никогда не выдаётся заpasttruth длявесов. Из causal
backtest исключаются головы с timestamp новее последнего полученногоугла:
пять validheads приpositiveage,шесть приage0. Приage0 каждый адаптер и новый
API точно совпадают с прежним frozenregimepredictor. Это не доказательство
аппаратнойустойчивости. Длясвободногоcos(w(t−d)+phi) наблюдается phi−w*d,
а неphiиdотдельно: unknownage изодногоугла вобщемслучае неидентифицируем.

## Сильный соперник

120ARXfits на техжеTRAINrecordings: angle/ioLPV × direct/dated-modal-residual
×6ridge,5cases. Теже синтетическиеages0/20/50/100ms и jitter0/±q.
Известныйage,phase sin(w*age),cos(w*age)−1 и взаимодействияистории — входы
ARX; никакого trialID/noiselabel/target вfeatures. ScalingTRAIN-only,
равныеrecordиquiet/driveвеса. Datedmodalbaseline предсказываетh+age.

ПоVAL percase выбран сильнейший из новых ARX и обоих старых ARX под всеми
темижeadapters. Это сравнение не даёт timestamps только нашемуметоду.
Frozenbank не переобучался и сохраняет прежние TRAINfits на более плотных
origins; новыйaugmentedARX использует предзарегистрированную1sсетку. В пуле
matchedreference остаются теже плотные frozenARX, что и вbank. Это сравнение
полных pipelines с доступом к темжеTRAINrecordings; разницу с augmentation
не следует интерпретировать как отдельный доказанный эффект архитектуры.

| case | reference_type | reference | augmented_model |
| --- | --- | --- | --- |
| trolley_theta | aligned | {'type': 'aligned', 'method': 'forced_3s_fixed', 'expert': 'active_arx'} | arx_angle_lpv_direct_1e-06.npz |
| jib_theta | aligned | {'type': 'aligned', 'method': 'forced_3s_fixed', 'expert': 'frozen_arx'} | arx_angle_lpv_direct_0.0001.npz |
| jib_beta | aligned | {'type': 'aligned', 'method': 'quadratic_0.25', 'expert': 'active_arx'} | arx_angle_lpv_direct_0.0001.npz |
| simult_theta | augmented | {'type': 'augmented', 'file': 'arx_angle_lpv_direct_1e-10.npz'} | arx_angle_lpv_direct_1e-10.npz |
| simult_beta | aligned | {'type': 'aligned', 'method': 'quadratic_0.25', 'expert': 'active_arx'} | arx_io_lpv_modal_age_residual_1.npz |

## Замороженный выбор по validation

Выбран **quadratic_0.25**. Developmentgate: **True**;
casewins противmatchedARX 5/5. Невыполненные критерии:
нет. Ни один old/stressscore не менялвыбор.

| method | balanced_mae_deg | positive_age_mae_deg | selected |
| --- | --- | --- | --- |
| unaware | 2.8548 | 3.34291 | False |
| hold | 2.25731 | 2.54627 | False |
| linear_0.10 | 1.68772 | 1.78681 | False |
| linear_0.25 | 2.21328 | 2.48756 | False |
| quadratic_0.25 | 1.51098 | 1.55116 | True |
| modal_3s | 1.88034 | 2.04363 | False |
| forced_3s_fixed | 1.82433 | 1.96896 | False |
| forced_3s_adapt | 1.77189 | 1.89904 | False |
| matched_arx | 1.74044 | nan | False |
| augmented_arx | 1.98417 | nan | False |

Gate и этот VALвыбор относятся к исходному globally-selectedARXreference.
Поздний age-conditionedchallenge не менял ни gate, ни выбранныйадаптер.

Выбранный адаптер сравнивается также с простым timestamp shift + hold; результат не сводится к сравнению только с игнорированием меток времени.

## Диагностика восстановления угла

После freeze дополнительно оценена самаheadчасть на всех положительныхVALages,
шумах и адаптерах. Это поздний diagnostic, не новыйкритерий выбора.
При50ms чистогоVAL, равныйвеспятиcases:

| method | current_angle_mae_deg | head_mae_deg |
| --- | --- | --- |
| forced_3s_adapt | 0.675042 | 0.603457 |
| forced_3s_fixed | 0.762795 | 0.693823 |
| hold | 2.1234 | 1.27613 |
| linear_0.10 | 0.877648 | 0.56254 |
| linear_0.25 | 2.2388 | 1.67372 |
| modal_3s | 0.788097 | 0.712794 |
| quadratic_0.25 | 0.50051 | 0.332911 |
| unaware | 2.1234 | 2.12422 |

Истинный дляsyntheticgap исходныйencoderangle использовался только как
offlineцелевой сигнал этойдиагностики. Его не было вinference или selection.
Clockpoisoning (15probes): замена unreceivedуглов и всех каналов
послеcurrentorigin на большие фиктивныезначения не изменила входы/прогноз.
Ошибка восстановления относительноencoder не является физическимgroundtruth.

## Старые оценочные записи

При известномage50ms и чистойистории: unaware 1.329577°,
selected 0.577427° (-56.57%), timestamp-hold
0.795655° (-27.43%vsit), matchedARX
1.167975° (-50.56%vsit). Снижение ошибки относительно
игнорированияtimestamps само по себе не доказывает новую physicsnovelty.

| age_ms | augmented_arx | conditional_arx | dated_modal | matched_arx | selected | timestamp_hold | unaware |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.13237 | 1.0738 | 0.756674 | 1.0738 | 0.525065 | 0.525065 | 0.525065 |
| 20 | 1.1243 | 1.09615 | 0.775769 | 1.09563 | 0.554539 | 0.598439 | 0.765346 |
| 30 | 1.13267 | 1.11453 | 0.783741 | 1.11099 | 0.560087 | 0.656029 | 0.946681 |
| 50 | 1.17834 | 1.0654 | 0.794328 | 1.16798 | 0.577427 | 0.795655 | 1.32958 |
| 70 | 1.20481 | 1.10806 | 0.800317 | 1.20735 | 0.605676 | 0.935587 | 1.72284 |
| 100 | 1.26024 | 1.09105 | 0.814696 | 1.28158 | 0.667135 | 1.16683 | 2.34768 |
| 150 | 1.75381 | 1.33167 | 0.833476 | 1.67846 | 0.846449 | 1.57485 | 3.43099 |

ConditionalARX — дополнительный позднийконтроль после просмотра исходной old
кривой, явно post-hoc. Длякаждогоcase/TRAINage0/20/50/100ms он выбирает
лучшиеARX/adapter поVAL, усредняяобаnoise. Дляновыхages используетсяближайший
TRAINage (tie→меньший). Его20решений закоммичены доегоотдельногоreplay,
никакого новогоTRAINfitting или измененияосновноймодели. При50ms этот
соперник имеет 1.065405°; всеусловия приведенывыше.
Это делает сравнениестроже, но не возвращает независимость старымданным.

Режимы при50ms:

| age_ms | noise_quantum | group | method | mae_deg | scenarios |
| --- | --- | --- | --- | --- | --- |
| 50 | 0 | all | augmented_arx | 1.17834 | 7 |
| 50 | 0 | all | conditional_arx | 1.0654 | 7 |
| 50 | 0 | all | dated_modal | 0.794328 | 7 |
| 50 | 0 | all | matched_arx | 1.16798 | 7 |
| 50 | 0 | all | selected | 0.577427 | 7 |
| 50 | 0 | all | timestamp_hold | 0.795655 | 7 |
| 50 | 0 | all | unaware | 1.32958 | 7 |
| 50 | 0 | drive_active | augmented_arx | 2.45313 | 6 |
| 50 | 0 | drive_active | conditional_arx | 2.28671 | 6 |
| 50 | 0 | drive_active | dated_modal | 2.61205 | 6 |
| 50 | 0 | drive_active | matched_arx | 2.47801 | 6 |
| 50 | 0 | drive_active | selected | 1.71475 | 6 |
| 50 | 0 | drive_active | timestamp_hold | 2.10331 | 6 |
| 50 | 0 | drive_active | unaware | 2.46979 | 6 |
| 50 | 0 | drive_and_length_active | augmented_arx | 3.5504 | 5 |
| 50 | 0 | drive_and_length_active | conditional_arx | 2.90943 | 5 |
| 50 | 0 | drive_and_length_active | dated_modal | 4.89154 | 5 |
| 50 | 0 | drive_and_length_active | matched_arx | 2.59952 | 5 |
| 50 | 0 | drive_and_length_active | selected | 2.42017 | 5 |
| 50 | 0 | drive_and_length_active | timestamp_hold | 2.94022 | 5 |
| 50 | 0 | drive_and_length_active | unaware | 2.48991 | 5 |
| 50 | 0 | length_active | augmented_arx | 0.750295 | 2 |
| 50 | 0 | length_active | conditional_arx | 0.937455 | 2 |
| 50 | 0 | length_active | dated_modal | 0.0892896 | 2 |
| 50 | 0 | length_active | matched_arx | 0.983204 | 2 |
| 50 | 0 | length_active | selected | 0.124285 | 2 |
| 50 | 0 | length_active | timestamp_hold | 0.137902 | 2 |
| 50 | 0 | length_active | unaware | 0.551097 | 2 |
| 50 | 0 | low_past_motion | augmented_arx | 0.817267 | 7 |
| 50 | 0 | low_past_motion | conditional_arx | 0.678953 | 7 |
| 50 | 0 | low_past_motion | dated_modal | 0.277192 | 7 |
| 50 | 0 | low_past_motion | matched_arx | 0.780716 | 7 |
| 50 | 0 | low_past_motion | selected | 0.225975 | 7 |
| 50 | 0 | low_past_motion | timestamp_hold | 0.397003 | 7 |
| 50 | 0 | low_past_motion | unaware | 1.04993 | 7 |
| 50 | 0 | old_numerical_scope | augmented_arx | 0.773852 | 7 |
| 50 | 0 | old_numerical_scope | conditional_arx | 0.648218 | 7 |
| 50 | 0 | old_numerical_scope | dated_modal | 0.272423 | 7 |
| 50 | 0 | old_numerical_scope | matched_arx | 0.740224 | 7 |
| 50 | 0 | old_numerical_scope | selected | 0.22109 | 7 |
| 50 | 0 | old_numerical_scope | timestamp_hold | 0.386841 | 7 |
| 50 | 0 | old_numerical_scope | unaware | 1.01704 | 7 |

Агрегация: равныеheads,оси внутриrecord,затем7records; разныеregimes имеют
разное покрытие. Всеcondition/head/caseпотери и выигрыши вall_paired_changes.
Balanced-score в отдельнойтаблице используетquiet/drive поровну и пятьcases;
его не смешивать с этими equal-scenario MAE. Overlapping origins неtrials.
30/70/150ms отсутствовали вTRAIN/VAL; это новые numericalconditions на старых
данных, не новые независимые измерения. Jitterпакетныйuniform±исходныйencoderq,
будущаяцель — прежнийencoderangle; не модель реального sensornoiseпроцесса.

## Проверки и отдельный профиль

235unit testsPASS доfitting. Независимая matrix-exponential проверка FOH с
двумя шагами времени; tests нулевогоage, genuineprehistory и truthmasking.
ИзTRAIN повторены5выбранныхaugmentedARXfits; все120VALscores, все8адаптеров,
selection/gate и метрики пересчитаны (41360comparisons).
Полныйraw/APIreplay 102272сравнений; дополнительно
15792replayусловногоARXсвоспроизведением20VALchoices;
все1281main/APHYNITY
оригинала целы, прежниеbundles/defaults не менялись.

Отдельныйbundle `artifacts/latency_bundle_20261108`: 44files,673872bytes.
`FrozenLatencyPilot.forecast(...,angle_age_samples=...)` требует известный
целыйage0..15samples. Меткивремени не из будущихtargets. Inference проверен
с запретом np.loadвнеbundle: raw/ancestralruns ему не нужны; нуженmainrepo+deps.
Experimentalprofile сохранён даже приgatefailure, не продвинут вneuraldefault.

| case | age_raw | repeats | median_ms |
| --- | --- | --- | --- |
| trolley_theta | 5 | 3 | 62.2247 |
| jib_theta | 5 | 3 | 52.6537 |
| jib_beta | 5 | 3 | 52.3013 |
| simult_theta | 5 | 3 | 54.5222 |
| simult_beta | 5 | 3 | 52.5405 |

15zero-inputprobes(L=.4m,5cases×age0/50/150ms): maxbias
0.033350°, не exactzero.
Latency — наблюдениеCPUнаданномMac, не real-time/hardwareguarantee.
Новая независимая запись, еёclock/units/encoderresolution и заранее frozen
протокол всёещё обязательны дляисходнойцели. Nativeactuator/MPC не доказаны.
