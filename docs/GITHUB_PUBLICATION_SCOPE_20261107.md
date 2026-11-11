# Область публикации исследовательской истории, 7 ноября 2026

Пользователь разрешил GitHub в текущем сообщении, заменив прежний запрет. Первая попытка push всей ветки остановлена автоматическим approval review: широкая передача содержимого, требующая уточнения. На момент первой попытки удалённая ветка не была создана.

Источник: `442d07a043c1ee2aba9be008c09af13ab3bd990c`, локальная ветка `research/broad-search-20261107`. Назначение: `https://github.com/kutyrevratmir3-design/cranepilot`, новая одноимённая ветка. `main` не меняется.

Локальная и удалённая истории имеют разные корневые коммиты; общей базы нет. Это публикация полной локальной истории (28 отсутствующих в remote коммитов), а не обычный небольшой fast-forward поверх main. Слияние/force-push не запланированы.

Отслеживаемый HEAD содержит 579 файлов, 81681030 байт без Git-сжатия. Полная передаваемая история может включать более старые версии. Сводка дерева:

| Каталог | Файлов | Байт |
|---|---:|---:|
| .github | 1 | 459 |
| .gitignore | 1 | 295 |
| CITATION.cff | 1 | 316 |
| LICENSE | 1 | 1228 |
| README.md | 1 | 11501 |
| configs | 23 | 53832 |
| docs | 35 | 369479 |
| examples | 4 | 4957 |
| paper | 31 | 2938923 |
| pyproject.toml | 1 | 657 |
| requirements-lock.txt | 1 | 110 |
| results | 352 | 77483037 |
| scripts | 86 | 644288 |
| src | 17 | 78644 |
| tests | 24 | 93304 |

Игнорируемые новые raw/processed datasets, новые artifact runs/weights, evidence.zip и APHYNITY не входят в push. Однако ранее отслеживаемые evidence.zip и наборы весов в старой истории передаются; .gitignore не удаляет уже отслеживаемые файлы. Исследовательские логи/манифесты могут содержать локальные абсолютные пути пользователя.

Крупнейшие файлы HEAD:

- `results/latest/research_20261106_evidence.zip`: 19716683 байт.
- `results/latest/research_20261103_evidence.zip`: 8947761 байт.
- `results/finetune_campaign_20261107/simulation_case_means.csv`: 5790955 байт.
- `results/history_backtest_20261107/simulation_case_means.csv`: 5213035 байт.
- `results/global_research_20261107/simulation_case_means.csv`: 4565304 байт.
- `results/forced_adaptive_20261107/simulation_case_means.csv`: 4449885 байт.
- `results/latest/cleanroom_20261106_metrics.csv`: 2269167 байт.
- `results/latest/all_family_neural_metrics.csv`: 1488878 байт.
- `results/combined_stress_20261107/simulation_case_means.csv`: 1372334 байт.
- `results/latest/CranePilot_Research_Draft.pdf`: 1095346 байт.

Коммиты к публикации на момент проверки:

```text
442d07a Record ninety fine-tunes, control-method research and complete replay
3b6a72f Preregister matched fine-tuning and control-inspired objective campaign
85f8fb0 Record global research and validated short-transfer LPV improvement
4c57451 Preregister global weak physics and direct delay lift research
362e9cc Record conditional history backtest results and retained adaptation failures
c82a569 Freeze conditional history backtest and guarded adaptation protocol
7a5f5be Record forced adaptation failure on measured data and causal diagnostics
f0eb7e6 Freeze forced parameter adaptation and residual transfer study
ee60fc5 Record combined candidate limits and off-grid adaptive physics comparison
2589473 Freeze off-grid frequency and damping follow-up with unchanged predictors
09bde96 Retain original incumbent flow for strict replay and preserve failed attempt
4515d2a Freeze combined candidate and structural mismatch falsification protocol
88bf626 Record broad strategy results, causal blend tradeoffs and verified inference
92d08f5 Freeze causal discrepancy blending before held-out evaluation
17ca172 Freeze nine-strategy matched continuation and regime-aware selection
0994dca Record frozen-model limit audit and retain validation tradeoffs
de16078 Use double precision for frozen angular limit diagnostics
0e88122 Freeze development-only physical-limit intervention audit
a3fdc8e Preserve November 6 model weights and verified research evidence
e28e1b5 Complete temporal transfer study and causal industrial data reconstruction
678b754 Freeze temporal observer ablations and independent free-sway diagnostic
7898fb5 Preserve verified local research weights and evidence archive
7cbb334 Complete local cross-length research, retain negative results and compile fast inference
84a7a7a Benchmark force-aware observers, robust losses, phase-normalized models and stronger references
c700063 Use CPU PyTorch for scientific CI checks
6fa98af Train causal innovation models with matched controls and retain complete evidence
dffb6dd Compile Overleaf draft and evaluate chronological prefix adaptation
ccd784d Build causal Python crane benchmark and retain first research results
```

## Подтверждение и успешная отправка

Пользователь отдельно подтвердил: «Да, публикуй эту исследовательскую ветку». После этого `git push -u origin research/broad-search-20261107` завершился успешно. На GitHub опубликованы 28 ранее подготовленных коммитов и preregistration `b2183ab` нового этапа (29 всего относительно исходного remote). Main не менялась. Итоговые результаты и небольшой переносимый комплект проверенных весов добавляются отдельным коммитом после проверки; новые большие evidence ZIP и datasets не публикуются.
