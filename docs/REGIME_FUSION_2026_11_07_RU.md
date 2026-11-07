# Причинный выбор режима и смешивание прогнозов

Протокол/численный код до выбора: `4f54dbc1db739fde1648ff4d0bd12c00a65e5daa`. Selection и data-free bundle
заморожены и закоммичены до нового replay. Все33записи уже многократно открыты;
здесь нет новой независимой проверки, аппаратного контроля или SOTA-результата.

## Метод

Четыре frozen-прогноза: прежнийARX, active-trainedARX, free-modal и forced-modal.
Новых TRAINfits не было. Из первых3sдоступной4sистории модели предсказывают
последнюю уже наблюдённую1s; шесть ошибок задают выпуклые веса с ridge к
контекстному prior. Никаких целей после текущего момента, будущих измеренных
позиций/L, будущих commands или scenario-ID в весах нет. Не используется
recurrent memory за пределами техже4s. Веса не считаются вероятностью верности.

Quiet определяется только по диапазонам прошлых401отсчётов x/J и прежнему
порогу2mm/2mrad. Там выдаётся прежний прогноз бит в бит. Это свойство конструкции,
не новая эмпирическая гарантия точности или устойчивости замкнутого контура.
В остальных окнах сравниваются fixed routing, inverse-error и constrained fit.
Никакая длина или угол ошибки не выбирают данные для исключения.
Legacy в таблицах означает ранее оценённый history-only kernel, включая
исследовательские origins вне прежнего publicAPIgate. Прежний API с knownplan
и ограничениями не ослаблен; новый predictor имеет отдельный контракт.

## Выбор по validation

Один вариант из11: **guarded_simplex_1**. Legacy/change по balancedVAL:
-19.31%, casewins 5/5.
Registered developmentgate: **True**.
Quiet feasibility проверена на каждом VALокне, а не только на среднем.

| method | balanced_mae_deg | quiet_feasible | selected |
| --- | --- | --- | --- |
| legacy | 1.67093 | True | False |
| active | 1.39697 | False | False |
| hard_context | 1.36904 | True | False |
| ramp_context | 1.36602 | True | False |
| guarded_inverse | 1.38259 | True | False |
| guarded_horizon_inverse | 1.49558 | True | False |
| guarded_winner | 1.46274 | True | False |
| guarded_simplex_0.01 | 1.40455 | True | False |
| guarded_simplex_0.1 | 1.37402 | True | False |
| guarded_simplex_1 | 1.34829 | True | True |
| guarded_simplex_10 | 1.36406 | True | False |
| frozen_arx | 1.88745 | None | False |
| active_arx | 1.58723 | None | False |
| free_modal | 2.06339 | None | False |
| forced_modal | 1.96296 | None | False |

## Старые7оценочных сценариев

Общий MAE: 0.503624° vs legacy
0.545399° (-7.66%), scenario wins 6/7.
Агрегация: равные горизонты, оси внутри сценария, затем равные целые сценарии;
каждый режим имеет своё покрытие. Overlapping windows не являются trials.

| condition | group | method | mae_deg | scenarios |
| --- | --- | --- | --- | --- |
| clean | all | active | 0.548902 | 7 |
| clean | all | active_arx | 1.03408 | 7 |
| clean | all | frozen_arx | 0.8581 | 7 |
| clean | all | hard_context | 0.518683 | 7 |
| clean | all | legacy | 0.545399 | 7 |
| clean | all | selected | 0.503624 | 7 |
| clean | drive_active | active | 1.65228 | 6 |
| clean | drive_active | active_arx | 2.22905 | 6 |
| clean | drive_active | frozen_arx | 1.80549 | 6 |
| clean | drive_active | hard_context | 1.65228 | 6 |
| clean | drive_active | legacy | 1.72642 | 6 |
| clean | drive_active | selected | 1.58165 | 6 |
| clean | drive_and_length_active | active | 2.28923 | 5 |
| clean | drive_and_length_active | active_arx | 2.3029 | 5 |
| clean | drive_and_length_active | frozen_arx | 3.21505 | 5 |
| clean | drive_and_length_active | hard_context | 2.28923 | 5 |
| clean | drive_and_length_active | legacy | 3.15703 | 5 |
| clean | drive_and_length_active | selected | 2.26034 | 5 |
| clean | length_active | active | 0.0983313 | 2 |
| clean | length_active | active_arx | 0.922915 | 2 |
| clean | length_active | frozen_arx | 0.894749 | 2 |
| clean | length_active | hard_context | 0.0989405 | 2 |
| clean | length_active | legacy | 0.0989405 | 2 |
| clean | length_active | selected | 0.0989405 | 2 |
| clean | low_past_motion | active | 0.241997 | 7 |
| clean | low_past_motion | active_arx | 0.69987 | 7 |
| clean | low_past_motion | frozen_arx | 0.495183 | 7 |
| clean | low_past_motion | hard_context | 0.193369 | 7 |
| clean | low_past_motion | legacy | 0.193369 | 7 |
| clean | low_past_motion | selected | 0.193369 | 7 |
| clean | old_numerical_scope | active | 0.237624 | 7 |
| clean | old_numerical_scope | active_arx | 0.672236 | 7 |
| clean | old_numerical_scope | frozen_arx | 0.477167 | 7 |
| clean | old_numerical_scope | hard_context | 0.188822 | 7 |
| clean | old_numerical_scope | legacy | 0.188822 | 7 |
| clean | old_numerical_scope | selected | 0.188822 | 7 |

Balanced-score отдельно даёт 1.120353° vslegacy 1.230896°:
quiet/drive поровну внутриaxis/record, равныеrecords внутриcase, пятьcases.
Эту величину нельзя напрямую сравнивать с общим MAE выше.

На общем старом MAE выбранный convex-fit лучше простого hard-context; это ретроспективное сравнение, а не независимое доказательство пользы усложнения.

Сохранены все per-head/per-record gains и losses, а не только выигрышные:

| case | scenario | legacy | active | hard_context | selected | change_vs_legacy_deg |
| --- | --- | --- | --- | --- | --- | --- |
| jib_beta | 21 | 1.25085 | 1.44432 | 1.29249 | 1.16708 | -0.0837683 |
| jib_beta | 22 | 0.0576705 | 0.0709286 | 0.0641734 | 0.0609546 | 0.00328415 |
| jib_beta | 23 | 0.0722224 | 0.084531 | 0.0732068 | 0.071254 | -0.000968349 |
| jib_theta | 21 | 1.25073 | 1.24921 | 1.24925 | 1.26696 | 0.01623 |
| jib_theta | 22 | 0.102709 | 0.100399 | 0.100399 | 0.101273 | -0.00143577 |
| jib_theta | 23 | 0.101152 | 0.101861 | 0.101861 | 0.100875 | -0.000276945 |
| simult_beta | 32 | 1.63229 | 1.75263 | 1.62126 | 1.54389 | -0.0883966 |
| simult_beta | 33 | 0.199121 | 0.188157 | 0.191085 | 0.18543 | -0.0136918 |
| simult_theta | 32 | 1.71792 | 1.72995 | 1.62702 | 1.60521 | -0.112711 |
| simult_theta | 33 | 0.126215 | 0.137095 | 0.135286 | 0.134872 | 0.00865682 |
| trolley_theta | 9 | 0.280554 | 0.208938 | 0.198191 | 0.197806 | -0.0827479 |
| trolley_theta | 13 | 0.281799 | 0.203831 | 0.20457 | 0.208658 | -0.0731412 |

## Синтетическая чувствительность

| condition | group | method | mae_deg | scenarios |
| --- | --- | --- | --- | --- |
| angle_jitter_half_quantum | all | active | 0.551395 | 7 |
| angle_jitter_half_quantum | all | active_arx | 1.04333 | 7 |
| angle_jitter_half_quantum | all | frozen_arx | 0.860753 | 7 |
| angle_jitter_half_quantum | all | hard_context | 0.520559 | 7 |
| angle_jitter_half_quantum | all | legacy | 0.546963 | 7 |
| angle_jitter_half_quantum | all | selected | 0.50528 | 7 |
| angle_jitter_one_quantum | all | active | 0.556336 | 7 |
| angle_jitter_one_quantum | all | active_arx | 1.06358 | 7 |
| angle_jitter_one_quantum | all | frozen_arx | 0.865581 | 7 |
| angle_jitter_one_quantum | all | hard_context | 0.524487 | 7 |
| angle_jitter_one_quantum | all | legacy | 0.550745 | 7 |
| angle_jitter_one_quantum | all | selected | 0.50969 | 7 |
| angle_lag_50ms | all | active | 1.36327 | 7 |
| angle_lag_50ms | all | active_arx | 1.47049 | 7 |
| angle_lag_50ms | all | frozen_arx | 1.4032 | 7 |
| angle_lag_50ms | all | hard_context | 1.34187 | 7 |
| angle_lag_50ms | all | legacy | 1.37752 | 7 |
| angle_lag_50ms | all | selected | 1.32998 | 7 |
| clean | all | active | 0.548902 | 7 |
| clean | all | active_arx | 1.03408 | 7 |
| clean | all | frozen_arx | 0.8581 | 7 |
| clean | all | hard_context | 0.518683 | 7 |
| clean | all | legacy | 0.545399 | 7 |
| clean | all | selected | 0.503624 | 7 |

Два равномерных jitterуровня ±q/2,±q добавлены только к истории угла всех
экспертов; q — исходный encoderquantum. Отдельно угол запаздывает на50ms.
Будущие цели — прежние encoderизмерения; метки режимов из прежних support/L
историй. Это пакетные synthetic perturbations, не новые physicalrecords,
не независимые повторы и не воспроизведение реального sensor-noise процесса.
Past-error truth тоже берётся из предоставленной noisy/lagged истории.
Quiet equality означает равенство legacy на тойже возмущённой истории,
а не равенство чистому legacyforecast. Ни одна стресс-оценка не меняла выбор.

## Проверка и переносимость

226unit tests PASS перед выбором; simplex-face solver независимо сверён с
SciPy constrained optimization. Development replay28125forecastcomparisons;
full raw/API replay 316125 сравнений на VAL/oldevaluation,
включая все три stressconditions. Все метрики и selection пересчитаны;
quietconstraint сохранён, все1281main/APHYNITYоригинала целы.

Bundle `artifacts/regime_bundle_20261107`: 36files,406575bytes, безrawdata
и ancestralruns. `FrozenRegimePilot.forecast(history,family=...,axis=...)`
принимает только(N,401)past100Hznewestfirst: угол/L/x/J/native-reference
истории в соответствии с family. Futureplanаргумента нет. Loader проверяет
хеши всех весов и численных исходников. Portable replay блокировал np.load
внеbundle; исходные данные нужны только verifierдля получения history/targets.

| case | packet_count | repeats | median_ms | minimum_ms |
| --- | --- | --- | --- | --- |
| trolley_theta | 1 | 5 | 65.3629 | 65.3143 |
| jib_theta | 1 | 5 | 55.1389 | 55.0267 |
| jib_beta | 1 | 5 | 55.2044 | 55.1568 |
| simult_theta | 1 | 5 | 55.5493 | 55.316 |
| simult_beta | 1 | 5 | 55.7145 | 55.6598 |

Это observedsingle-packet CPUlatency на данномMac, не гарантия real-time.
15fixedzero-inputprobes: максимальный bias 0.035304°,
точно как уlegacyв тихом режиме. Exactequilibrium не заявляется.

Прежние bundles/API/neuraldefault и APHYNITY не изменены. Новый API — отдельный
экспериментальный history-onlypredictor. Будущее движение support предположено,
nativeactuator xr/Jr→фактическое движение не идентифицирован; качество прогноза
не доказывает counterfactual MPC или hardwareclosed-loopperformance.
Дальнейшая решающая проверка требует новых авторских/стендовыхrawtrials и
заранее зафиксированного протокола до просмотра их результатов.
