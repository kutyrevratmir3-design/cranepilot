# Равный доступ к обучению: разработка против усиленного ARX

Протокол и код до fitting: `b3ce41d9ef29e59ae37e90e064a2fbd4d9b8961c`. TRAIN1–12, VALIDATION13–16
внешнего rotary-camera набора. Все модели получили одинаковые обучающие records
и доступные causal history/known-command signals. TEST17–20 уже использовался:
новая оценка на нём **ретроспективная**, НЕ независимое подтверждение.

## Итог

Выбран только по validation: **structured_narx**, изменение MAE против сильного ARX
**-4.243%**. Development gate(≥10% reduction и≥3/4recordwins): **False**.
**Итоговая цель ещё не достигнута: свежего подтверждающего набора нет.**
Ни один опубликованный default/bundle или APHYNITY не заменён.

| group | method | mae_deg | rmse_deg |
| --- | --- | --- | --- |
| retrospective | adapted_neural | 0.0910703 | 0.115719 |
| retrospective | equilibrium_neural | 0.0862028 | 0.109114 |
| retrospective | physics_prior_arx | 0.037918 | 0.0473815 |
| retrospective | strong_arx | 0.0379932 | 0.0474696 |
| retrospective | structured_narx | 0.0368474 | 0.0462128 |
| validation | adapted_neural | 0.0810431 | 0.110874 |
| validation | equilibrium_neural | 0.0805023 | 0.109907 |
| validation | physics_prior_arx | 0.0393671 | 0.0499104 |
| validation | strong_arx | 0.0395723 | 0.0501802 |
| validation | structured_narx | 0.0378934 | 0.0483836 |

Парные comparisons по целым records, не overlapping windows:

| group | method | change_percent | record_wins | records | descriptive_low | descriptive_high |
| --- | --- | --- | --- | --- | --- | --- |
| validation | strong_arx | 0 | 0 | 4 | 0 | 0 |
| validation | physics_prior_arx | -0.518423 | 3 | 4 | -0.818636 | 0.0205971 |
| validation | structured_narx | -4.24266 | 4 | 4 | -5.56875 | -2.69099 |
| validation | adapted_neural | 104.798 | 0 | 4 | 79.083 | 160.557 |
| validation | equilibrium_neural | 103.431 | 0 | 4 | 80.4497 | 149.66 |
| retrospective | strong_arx | 0 | 0 | 4 | 0 | 0 |
| retrospective | physics_prior_arx | -0.198093 | 3 | 4 | -0.473095 | -0.0248074 |
| retrospective | structured_narx | -3.01595 | 4 | 4 | -5.02749 | -1.15162 |
| retrospective | adapted_neural | 139.702 | 0 | 4 | 94.124 | 201.245 |
| retrospective | equilibrium_neural | 126.89 | 0 | 4 | 89.6103 | 177.229 |

Bootstrap4recordclusters,2000draws,seed7112028,общие resamples для методов;
описательные интервалы не устраняют reuse и малый размер данных.

## Что именно проверено

StrongARX выбирает direct или geometric residual и ridge среди6значений
1e−10..1; все12fit сохранены. Поэтому улучшение не обеспечивается намеренно
узким comparator grid. PhysicspriorARX использует ту же линейную feature class
поверх damped oscillator, fitted на тех же12TRAINrecords в предыдущем этапе.
StructuredNARX добавляет5групп ограниченных нелинейных history/command terms,
без новых measured signals и arbitrary polynomial search. Future command features
обрываются на соответствующем horizon. Все24ridge outcomes:

| method | base | alpha | validation_mae_deg |
| --- | --- | --- | --- |
| strong_arx | none | 1e-10 | 0.041366 |
| strong_arx | none | 1e-08 | 0.0414605 |
| strong_arx | none | 1e-06 | 0.0414129 |
| strong_arx | none | 0.0001 | 0.0395725 |
| strong_arx | none | 0.01 | 0.0624592 |
| strong_arx | none | 1 | 0.268614 |
| strong_arx | geometric | 1e-10 | 0.041366 |
| strong_arx | geometric | 1e-08 | 0.0414605 |
| strong_arx | geometric | 1e-06 | 0.0414129 |
| strong_arx | geometric | 0.0001 | 0.0395723 |
| strong_arx | geometric | 0.01 | 0.0623756 |
| strong_arx | geometric | 1 | 0.165958 |
| physics_prior_arx | fitted | 1e-10 | 0.041366 |
| physics_prior_arx | fitted | 1e-08 | 0.0414606 |
| physics_prior_arx | fitted | 1e-06 | 0.0414102 |
| physics_prior_arx | fitted | 0.0001 | 0.0393671 |
| physics_prior_arx | fitted | 0.01 | 0.054723 |
| physics_prior_arx | fitted | 1 | 0.0643611 |
| structured_narx | fitted | 1e-10 | 0.0401564 |
| structured_narx | fitted | 1e-08 | 0.040131 |
| structured_narx | fitted | 1e-06 | 0.0398367 |
| structured_narx | fitted | 0.0001 | 0.0378934 |
| structured_narx | fitted | 0.01 | 0.0535368 |
| structured_narx | fitted | 1 | 0.0606162 |

Прежняя retained neural architecture теперь дообучена на тех же12records;
3source seeds,100epochs max,AdamWlr.0003,record-balanced seven-horizon MSE.
Checkpoint только по validation, epoch−1 допустим. Equilibrium variant вычитает
отклик этой же сети на zero packet с той же omega. Это ограничение zero response,
не passivity/stability theorem. Все6neuralfits:

| seed | equilibrium | selected_epoch | validation_mae_deg | method | fit_seconds |
| --- | --- | --- | --- | --- | --- |
| 0 | False | 92 | 0.0829307 | adapted_neural | 35.0241 |
| 1 | False | 96 | 0.085406 | adapted_neural | 34.3131 |
| 2 | False | 94 | 0.0830355 | adapted_neural | 45.1324 |
| 0 | True | 99 | 0.0844941 | equilibrium_neural | 85.0963 |
| 1 | True | 96 | 0.0823117 | equilibrium_neural | 76.3048 |
| 2 | True | 99 | 0.0815485 | equilibrium_neural | 77.9769 |

Изменены только новые model copies. Virtual reference=2*knownvelocity/.2,
metadata/source-clock frequency и forcing convention используют TRAIN-only
PEM omega/gain. X и prescribed acceleration масштабированы gain*g/(L*omega²).
Это явный адаптер угловой подмодели; native actuator bypass остаётся. Angular
history9native samples каузально удерживаются до41entries. Никакой будущей
измеренной траектории, test mean или future timestamp в input нет.

## Данные авторов на том же стенде

По предложению пользователя проверены работы Kulash Talapiden и Ahmad Bala
Alhassan. [Talapiden et al.2026](https://journals.sagepub.com/doi/10.1177/10775463251411778)
прямо указывают выдачу данных по запросу corresponding author.
[Публичный набор](https://data.mendeley.com/datasets/x69hyyv3zw/2) — прежние33cases,
не свежий test. [Alhassan et al.2024](https://doi.org/10.1111/exsy.13484) также
указывают on-request data; принадлежность к тому же физическому экземпляру
не подтверждена. Для ISA2026 raw-file link не проверен. Полный source audit
и конкретный перечень сигналов: SAME_RIG_DATA_ACCESS_20261107.md.
Авторские письма/посты не отправлялись. Ссылки на результаты в PDF и статья
с теми же авторами сами по себе не предоставляют новые численные записи.

## Проверки и дальнейшая работа

184unit tests PASS до fitting. Все 9240 forecasts повторно рассчитаны
из raw source+saved checkpoints, metrics независимо пересчитаны, validation
selection воспроизведён. Все 1281 original
main/APHYNITY files сохранены. Source/parent/checkpoint/resultSHA verified.
Прогнозы игнорируют poisoned targets/futuretimestamps. Изменений старых runs нет.

Новый метод оценивается как разработка. Даже хороший retrospective score не
закрывает задачу независимого преимущества. Для подтверждения нужны новые raw
recordings, замороженный метод/бейслайн и заранее заданные metrics/conditions.
Не понижать comparator или критерии ради формального успеха. Если кандидат
не выигрывает, продолжать с конкретной диагностикой и сохранить отрицательный
результат; не объявлять нейросеть обязательной частью пилота.
