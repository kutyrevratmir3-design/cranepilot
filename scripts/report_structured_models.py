"""Report every structured-model outcome after the frozen evaluation and replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cranepilot.data import sha256
from cranepilot.rollout import HORIZONS_S
from scripts.structured_candidate import CONTROLS
from scripts.report_finetune_campaign import (check_horizons, scope, table,
    verify_mapping, seed_tables as retained_seed_tables)

ALIASES = ['selected', 'common']
REFERENCES = ['incumbent', 'soft_symmetry', 'previous_local']
RISK_COLUMNS = ['mean_packet_peak_deg', 'packet_mae_p95_deg', 'tail_mae_deg',
                'normalized_increment_mae_deg']
SEED_NOTE = ('Descriptive variation of three trained seed models; frozen-base seeds for additive '
             'models and initialization seeds for standalone latent models. Not confidence intervals '
             'or independent hardware repetitions. Mean seed MAE is not ensemble MAE.')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def verify_evidence(root, run):
    provenance = json.loads((run/'provenance.json').read_text())
    if provenance['status'] != 'complete':
        raise ValueError('Completed structured-model evaluation required')
    counts = {}
    for key, base in [('guarded_hashes', root), ('checkpoint_hashes', run), ('result_hashes', run)]:
        verify_mapping(provenance[key], base)
        counts[key] = len(provenance[key])
    actual = {str(p.relative_to(run)) for p in run.rglob('*') if p.is_file() and p.name != 'provenance.json'}
    if actual != set(provenance['result_hashes']):
        raise ValueError('Completed run membership changed')
    if sha256(run/'selection.json') != provenance['selection_sha256']:
        raise ValueError('Selection changed')
    cfg = provenance['config']
    expected = {f'{fold}/{name}_seed{seed}.pt' for fold in cfg['folds']
                for name in cfg['variants'] for seed in cfg['seeds']}
    if set(provenance['checkpoint_hashes']) != expected:
        raise ValueError('Checkpoint coverage does not match fixed configuration')
    for key in ['source_run', 'broad_run', 'parent_run']:
        directory = root/cfg[key]
        previous = json.loads((directory/'provenance.json').read_text())
        if previous['status'] != 'complete':
            raise ValueError(f'Incomplete upstream run: {key}')
        verify_mapping(previous['checkpoint_hashes'], directory)
        if key == 'parent_run':
            verify_mapping(previous['result_hashes'], directory)
        counts[key+'_checkpoints'] = len(previous['checkpoint_hashes'])
    for stage in ['train', 'evaluate']:
        checks = json.loads((run/f'{stage}_verification.json').read_text())
        if checks['passed'] is not True:
            raise ValueError(f'Failed {stage} verification')
        counts[stage+'_checks'] = len(checks['checks'])
    return provenance, counts


def compare(frame, keys, value='mae_deg'):
    result = frame.copy()
    for name in REFERENCES:
        reference = frame[frame.model == name][keys+[value]].rename(columns={value: name+'_'+value})
        result = result.merge(reference, on=keys, how='left', validate='many_to_one')
        denominator = result[name+'_'+value]
        if denominator.isna().any() or (denominator <= 0).any():
            raise ValueError(f'Missing or zero reference: {name}')
        result['change_vs_'+name+'_percent'] = 100*(result[value]/denominator-1.)
    return result


def real_tables(run, cfg, out, selection):
    methods = CONTROLS+list(cfg['variants'])
    metrics, risks = [], []
    for stage in ['train', 'evaluate']:
        for destination, suffix in [(metrics, 'metrics'), (risks, 'risk_metrics')]:
            frame = scope(pd.read_csv(run/f'{stage}_{suffix}.csv'))
            expected = set(methods if stage == 'train' else methods+ALIASES)
            for _, part in frame.groupby(['fold', 'group', 'scenario', 'regime']):
                if set(part.model) != expected:
                    raise ValueError('Unmatched real model sets')
            if stage == 'train':
                aliases = []
                for fold, part in frame.groupby('fold'):
                    for alias, name in [('selected', selection['folds'][fold]['model']),
                                        ('common', selection['common']['model'])]:
                        selected = part[part.model == name].copy()
                        selected['model'] = alias
                        aliases.append(selected)
                frame = pd.concat([frame, *aliases], ignore_index=True)
            destination.append(frame)
    real, risk = pd.concat(metrics, ignore_index=True), pd.concat(risks, ignore_index=True)
    keys = ['scope', 'fold', 'group', 'scenario', 'extrapolation', 'model', 'regime']
    check_horizons(real, keys, 'mae_deg')
    real.to_csv(out/'real_metrics_all.csv', index=False)
    records = real.groupby(keys, as_index=False).mae_deg.mean()
    records = compare(records, [key for key in keys if key != 'model'])
    records.to_csv(out/'real_record_means.csv', index=False)
    horizon = real.groupby(['scope', 'fold', 'model', 'regime', 'horizon_s'], as_index=False).agg(
        mae_deg=('mae_deg', 'mean'), scenarios=('scenario', 'nunique'))
    horizon.to_csv(out/'real_horizons.csv', index=False)
    summary = horizon.groupby(['scope', 'fold', 'model', 'regime'], as_index=False).agg(
        mae_deg=('mae_deg', 'mean'), scenarios=('scenarios', 'min'))
    summary = compare(summary, ['scope', 'fold', 'regime'])
    summary.to_csv(out/'real_summary_all_scopes.csv', index=False)
    for name, part in summary.groupby('scope'):
        part.to_csv(out/f'{name}_summary.csv', index=False)
    if risk.duplicated(keys).any() or not np.isfinite(risk[RISK_COLUMNS]).all().all():
        raise ValueError('Invalid risk cells')
    risk.to_csv(out/'risk_record_metrics.csv', index=False)
    risk_summary = risk.groupby(['scope', 'fold', 'model', 'regime'], as_index=False)[RISK_COLUMNS].mean()
    risk_summary = risk_summary.rename(columns={'packet_mae_p95_deg': 'mean_of_record_packet_mae_p95_deg'})
    risk_summary.to_csv(out/'risk_summary.csv', index=False)
    return methods, real, summary, horizon, records, risk_summary


def fit_tables(run, cfg, out):
    rows = json.loads((run/'fits.json').read_text())
    expected = {(f, n, s) for f in cfg['folds'] for n in cfg['variants'] for s in cfg['seeds']}
    if len(rows) != len(expected) or {(r['fold'], r['model'], r['seed']) for r in rows} != expected:
        raise ValueError('Fit metadata does not match configuration')
    fields = ['fold', 'model', 'seed', 'selected_epoch', 'epochs_executed', 'trainable_parameter_count',
              'total_parameter_count', 'optimizer', 'optimizer_steps', 'backward_passes',
              'optimizer_closure_evaluations', 'fit_seconds', 'elapsed_seconds', 'selection_score']
    fits = pd.DataFrame(rows)[fields].copy()
    fits['family'] = fits.model.map(lambda name: cfg['variants'][name]['family'])
    fits['mode'] = fits.model.map(lambda name: cfg['variants'][name].get('mode', 'residual'))
    fits['retained_starting_checkpoint'] = fits.selected_epoch == -1
    fits['frozen_parameter_count'] = fits.total_parameter_count-fits.trainable_parameter_count
    fits['fallback_equals_soft'] = fits.retained_starting_checkpoint & (fits['mode'] != 'standalone')
    fits.to_csv(out/'fits_flat.csv', index=False)
    aggregate = fits.groupby('model', as_index=False).agg(
        fits=('seed', 'size'), epoch_minus_one=('retained_starting_checkpoint', 'sum'),
        unchanged_soft_fallbacks=('fallback_equals_soft', 'sum'),
        selected_epoch_min=('selected_epoch', 'min'), selected_epoch_median=('selected_epoch', 'median'),
        selected_epoch_max=('selected_epoch', 'max'), epochs_executed=('epochs_executed', 'sum'),
        trainable_min=('trainable_parameter_count', 'min'), trainable_max=('trainable_parameter_count', 'max'),
        frozen_parameters=('frozen_parameter_count', 'max'),
        optimizer_steps=('optimizer_steps', 'sum'), backward_passes=('backward_passes', 'sum'),
        fit_seconds=('fit_seconds', 'sum'), runner_elapsed_seconds=('elapsed_seconds', 'sum'))
    aggregate.to_csv(out/'fit_summary.csv', index=False)
    fits.groupby(['fold', 'model'], as_index=False).agg(
        epoch_minus_one=('retained_starting_checkpoint', 'sum'),
        best_epoch_mean=('selected_epoch', 'mean'), fit_seconds=('fit_seconds', 'sum')).to_csv(
            out/'fit_fold_summary.csv', index=False)
    return fits, aggregate


def seed_tables(root, run, cfg, out, real, summary):
    # This previously audited routine only aggregates arbitrary configured seed
    # arrays; no previous strategy counts, controls, or selection enter it.
    dispersion = retained_seed_tables(root, run, cfg, out, real, summary)
    write(out/'seed_scope.json', {'methods': list(cfg['variants']), 'seeds': cfg['seeds'],
        'note': SEED_NOTE, 'seed_mae_mean_is_not_ensemble_mae': True,
        'aggregation': 'Equal records at each horizon, then equal horizons, then descriptive seed statistics'})
    return dispersion


def acceptance(summary, cfg, selection, out):
    cells = []
    reference = cfg['acceptance']['reference']
    if reference not in REFERENCES:
        raise ValueError('Acceptance reference missing from comparison tables')
    scopes = [('evaluation_extrapolation', fold, cfg['acceptance'][fold+'_max_ratio']) for fold in cfg['folds']]
    scopes.append(('load_stress', 'primary', cfg['acceptance']['load_max_ratio']))
    for alias in ALIASES:
        for scope_name, fold, maximum in scopes:
            for regime in ['all', 'active']:
                rows = summary[(summary.scope == scope_name) & (summary.fold == fold)
                               & (summary.model == alias) & (summary.regime == regime)]
                if len(rows) != 1:
                    raise ValueError('Missing acceptance cell')
                row = rows.iloc[0]
                ratio = float(row.mae_deg/row[reference+'_mae_deg'])
                resolved = selection['folds'][fold]['model'] if alias == 'selected' else selection['common']['model']
                cells.append(dict(rule=alias, resolved_model=resolved, scope=scope_name, fold=fold,
                    regime=regime, mae_deg=float(row.mae_deg), reference=reference,
                    reference_mae_deg=float(row[reference+'_mae_deg']), ratio=ratio,
                    maximum_ratio=maximum, passed=ratio <= maximum))
    result = {'rules': {alias: {'passed': all(c['passed'] for c in cells if c['rule'] == alias),
                               'cells': [c for c in cells if c['rule'] == alias]} for alias in ALIASES},
              'common_model': selection['common']['model'],
              'operational_thresholds_not_statistical_bounds': True}
    write(out/'acceptance.json', result)
    pd.DataFrame(cells).to_csv(out/'acceptance.csv', index=False)
    return result


def simulation_tables(run, cfg, out, methods):
    frame = pd.read_csv(run/'simulation_metrics.csv')
    casekeys = ['fold', 'model', 'force_law', 'drive_amplitude_m', 'future_acceleration_gain',
                'measurement', 'omega_ratio', 'damping_per_s', 'length_m', 'amplitude_deg',
                'phase_rad', 'trajectory_id']
    check_horizons(frame, casekeys, 'absolute_error_deg')
    expected = set(methods+ALIASES)
    expected_cases = int(np.prod([len(value) for value in cfg['simulation'].values() if isinstance(value, list)]))
    counts = {}
    for fold, part in frame.groupby('fold'):
        if set(part.model) != expected:
            raise ValueError('Unmatched simulation methods')
        cases = part[[key for key in casekeys if key not in ['fold', 'model']]].drop_duplicates()
        if len(cases) != expected_cases:
            raise ValueError('Simulation case count differs from fixed factorial configuration')
        counts[fold] = len(cases)
    if set(counts) != set(cfg['folds']):
        raise ValueError('Missing simulation fold')
    cases = frame.groupby(casekeys, as_index=False).absolute_error_deg.mean().rename(columns={'absolute_error_deg': 'mae_deg'})
    if not (cases.groupby([key for key in casekeys if key != 'model']).model.nunique() == len(expected)).all():
        raise ValueError('Incomplete per-case model comparison')
    cases.to_csv(out/'simulation_case_means.csv', index=False)
    groups = ['fold', 'model', 'force_law', 'drive_amplitude_m', 'future_acceleration_gain', 'measurement']
    summary = cases.groupby(groups, as_index=False).agg(mae_deg=('mae_deg', 'mean'),
        case_median_mae_deg=('mae_deg', 'median'), case_p95_mae_deg=('mae_deg', lambda x: float(x.quantile(.95))),
        worst_case_mae_deg=('mae_deg', 'max'), cases=('mae_deg', 'size'))
    summary = compare(summary, [key for key in groups if key != 'model'])
    summary.to_csv(out/'simulation_groups.csv', index=False)
    overall = cases.groupby(['fold', 'model', 'measurement'], as_index=False).agg(
        mae_deg=('mae_deg', 'mean'), cases=('mae_deg', 'size'))
    overall.to_csv(out/'simulation_summary.csv', index=False)
    for suffix, extras in [('frequency_damping', ['omega_ratio', 'damping_per_s']), ('lengths', ['length_m'])]:
        cases.groupby(groups+extras, as_index=False).mae_deg.mean().to_csv(out/f'simulation_{suffix}.csv', index=False)
    frame.groupby(groups+['horizon_s'], as_index=False).absolute_error_deg.mean().to_csv(out/'simulation_horizons.csv', index=False)
    return counts, overall


def figures(out, summary, horizon, methods, cfg, selection):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, len(cfg['folds']), figsize=(15, 7.5), sharey=True, squeeze=False)
    for ax, fold in zip(axes[0], cfg['folds']):
        part = summary[(summary.scope == 'evaluation_extrapolation') & (summary.fold == fold)
                       & (summary.regime == 'all')].set_index('model').loc[methods]
        colors = ['#788390' if name in CONTROLS else
                  '#18847c' if cfg['variants'][name]['family'] == 'adapter' else '#5067b4' for name in methods]
        bars = ax.barh(np.arange(len(methods)), part.mae_deg, color=colors)
        chosen = selection['folds'][fold]['model']
        if chosen in methods:
            bars[methods.index(chosen)].set_edgecolor('#15212b')
            bars[methods.index(chosen)].set_linewidth(2.)
        ax.set(title=fold, xlabel='MAE угла, °')
        ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    axes[0][0].set_yticks(np.arange(len(methods)), methods)
    axes[0][0].invert_yaxis()
    fig.suptitle(f'Все {len(methods)} методов: ретроспективная экстраполяция по длине', fontsize=13)
    fig.tight_layout(); fig.savefig(out/'all_methods_extrapolation.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, len(cfg['folds']), figsize=(15, 8), squeeze=False)
    for column, fold in enumerate(cfg['folds']):
        for row, regime in enumerate(['all', 'active']):
            ax = axes[row, column]
            local = horizon[(horizon.scope == 'evaluation_extrapolation') & (horizon.fold == fold)
                            & (horizon.regime == regime)]
            names = list(dict.fromkeys(['incumbent', 'soft_symmetry', 'previous_local', selection['folds'][fold]['model']]))
            for name in names:
                y = local[local.model == name].sort_values('horizon_s')
                ax.plot(y.horizon_s, y.mae_deg, 'o-', ms=3, label=name)
            ax.set(title=f'{fold} / {regime}', xlabel='Горизонт, с', ylabel='MAE, °')
            ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=7)
    fig.suptitle('Выбор по validation: горизонты общего и активного режимов', fontsize=12)
    fig.tight_layout(); fig.savefig(out/'selected_horizons.png', dpi=180); plt.close(fig)


def external_verification(out, expected_models):
    path = out/'replay_verification.json'
    replay = json.loads(path.read_text())
    if replay.get('passed') is not True:
        raise ValueError('Full structured replay must pass before the report')
    parameters = replay['parameter_audit']
    if parameters.get('passed') is not True or parameters['models_checked'] != expected_models:
        raise ValueError('Complete parameter audit required')
    if replay['original_files'].get('passed') is not True:
        raise ValueError('Original main/APHYNITY preservation failed')
    latency_path = out/'inference_latency.json'
    latency = json.loads(latency_path.read_text()) if latency_path.exists() else None
    if latency is not None and latency.get('passed') is not True:
        raise ValueError('Failed latency/replay benchmark')
    return replay, latency


def audit_tables(out, replay):
    rows = []
    for model in replay['parameter_audit']['models']:
        row = {key: model[key] for key in ['fold', 'model', 'seed', 'family', 'selected_epoch',
            'additive_frozen_base', 'total_parameter_count', 'trainable_parameter_count',
            'frozen_tensors_checked', 'fallback_tensors_checked', 'buffers_checked', 'changed_elements']}
        row.update({key: value for key, value in model['stability'].items() if key != 'omega_values'})
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'parameter_and_stability_audit.csv', index=False)
    if not frame.passed.all() or frame.plant_or_controller_certificate.any():
        raise ValueError('Incorrect stability interpretation or failed audit')
    return frame


def unit_test_record(out):
    path = out/'structured_models_tests.log'
    if not path.exists():
        return None
    content = path.read_text()
    matches = re.findall(r'Ran (\d+) tests? in ([\d.]+)s', content)
    if not matches or not re.search(r'^OK\s*$', content, flags=re.MULTILINE):
        raise ValueError('Retained unit-test log does not demonstrate passing tests')
    count, seconds = matches[-1]
    return {'tests': int(count), 'seconds': float(seconds), 'passed': True, 'sha256': sha256(path)}


def document(path, run, out, cfg, methods, summary, horizon, records, risks,
             fits, fit_summary, seeds, selection, verdict, simulation_counts,
             simulation, hash_counts, replay, latency, audits, unit_tests):
    headline = summary[summary.scope == 'evaluation_extrapolation']
    all_active = headline[headline.regime.isin(['all', 'active'])].pivot(
        index='model', columns=['fold', 'regime'], values='mae_deg')
    all_active.columns = [' / '.join(c) for c in all_active.columns]
    quiet = headline[headline.regime == 'quiet'].pivot(index='model', columns='fold', values='mae_deg')
    chosen = headline[(headline.model == 'selected') & headline.regime.isin(['all', 'active'])][
        ['fold', 'regime', 'mae_deg', 'change_vs_previous_local_percent', 'change_vs_soft_symmetry_percent']]
    load = summary[(summary.scope == 'load_stress') & (summary.fold == 'primary')].pivot(
        index='model', columns='regime', values='mae_deg')
    record_table = records[(records.scope == 'evaluation_extrapolation') & (records.model == 'selected')
                          & records.regime.isin(['all', 'active'])][
        ['fold', 'scenario', 'regime', 'mae_deg', 'previous_local_mae_deg', 'change_vs_previous_local_percent']]
    risk_table = risks[(risks.scope == 'evaluation_extrapolation') & (risks.regime == 'all')
                       & risks.model.isin(['soft_symmetry', 'previous_local', 'selected', 'common'])]
    seed_table = seeds[(seeds.scope == 'evaluation_extrapolation') & (seeds.regime == 'all')][
        ['fold', 'model', 'seed_mae_mean_deg', 'seed_mae_std_deg', 'seed_mae_min_deg',
         'seed_mae_max_deg', 'ensemble_mae_deg']]
    simulation_table = simulation.pivot(index='model', columns=['fold', 'measurement'], values='mae_deg')
    simulation_table.columns = [' / '.join(c) for c in simulation_table.columns]
    gates = pd.DataFrame([cell for rule in verdict['rules'].values() for cell in rule['cells']])
    status = '; '.join(f"{rule}: {'выполнен' if value['passed'] else 'не выполнен'}" for rule, value in verdict['rules'].items())
    additive_fallbacks = int(fits.fallback_equals_soft.sum())
    standalone_fallbacks = int((fits.retained_starting_checkpoint & (fits['mode'] == 'standalone')).sum())
    latitude = 'Отдельное измерение задержки отсутствует.'
    if latency is not None:
        latency_rows = pd.DataFrame(latency.get('summary', []))
        wanted = [key for key in ['fold', 'model', 'measurements', 'median_ms', 'p95_ms', 'max_ms', 'fraction_over_50ms'] if key in latency_rows]
        latitude = (table(latency_rows[wanted])+'\n\nПрогретый CPU, batch=1, актор + observer + ансамбль трёх угловых моделей. '
            'Измерение не включает загрузку весов, sensor I/O и оптимизацию MPC. '
            'Сравнение с 50 мс описывает этот запуск и не является гарантией real-time или WCET.')
    stability = replay['parameter_audit']
    stability_compact = {key: value for key, value in stability.items()
                         if key != 'models' and not isinstance(value, (list, dict))}
    stability_types = audits.groupby('kind').size().to_dict()
    unit_note = (f"Существующие и новые unit tests: {unit_tests['tests']} PASS; "
                 'лог `structured_models_tests.log`. ' if unit_tests else
                 'Полный unit-test лог не включён в эту копию отчёта. ')
    integration_note = ('Аудит перехода от прогнозов к реальному оптимизатору описан отдельно в '
        '[CONTROL_INTEGRATION_AUDIT](CONTROL_INTEGRATION_AUDIT_20261107.md).'
        if (path.parent/'CONTROL_INTEGRATION_AUDIT_20261107.md').exists() else '')
    bundle_note = ('Переносимый комплект прогнозирования описан в '
        '[RESEARCH_BUNDLE](RESEARCH_BUNDLE_20261107.md). Он разделяет сохранённый default '
        'и новые исследовательские candidates; экспорт не меняет результаты gate '
        'и не объявляет выбранный candidate принятым для эксплуатации.'
        if (path.parent/'RESEARCH_BUNDLE_20261107.md').exists() else '')
    text = f'''# Структурированные модели CranePilot — 7 ноября 2026

Завершены {len(fits)} обучений: {len(cfg['variants'])} новых вариантов × {len(cfg['folds'])} разбиения × {len(cfg['seeds'])} seed. Выбор общей стратегии по validation — `{selection['common']['model']}`. Заранее заданные критерии принятия: **{status}**. Локальные решения: {', '.join(f"{fold}: `{value['model']}`" for fold, value in selection['folds'].items())}.

В {int(fits.retained_starting_checkpoint.sum())} fit выбран checkpoint эпохи −1: {additive_fallbacks} добавочных моделей точно сохраняют исходный soft, {standalone_fallbacks} самостоятельных моделей сохраняют собственную начальную динамику. Это не улучшение от обучения. Даже успешный gate строки `previous_local` означает сохранение ранее выбранных моделей, а не новое архитектурное достижение.

## Результат предварительного выбора

{table(chosen)}

Все реальные записи уже участвовали в исследовании. Новый этап фиксирует архитектуры и выбор checkpoints до evaluation/simulation, но не создаёт новый независимый test. Результат ретроспективный; нет аппаратного подтверждения и основания для заявления SOTA. Существующий LinearMPC не использует этот forecaster: sway, tracking, effort, ограничения и стоимость замкнутого управления здесь не измерены.

{integration_note}

`previous_local` — победители предыдущего этапа: phase-space fine-tune в primary, LPV actor ridge .01 в short-transfer, incumbent в long-transfer. Это разное устройство моделей между folds. `selected` и `common` — псевдонимы замороженного выбора, а не дополнительные fit. Любой более удачный evaluation-only кандидат в таблицах ниже остаётся исследовательской гипотезой и не заменяет сохранённый выбор.

## Что проверяли

Четыре adapter: ширина8/16 с ограниченной силой через осциллятор; ширина8 с прямой поправкой угла; force8 с фазовым surrogate. База soft заморожена; новый выход инициализирован нулём, параметры базы сохранены в каждом checkpoint. Обучается321 или641 новых параметров. Для силы bound .05 rad и zeta .05; для прямого углового выхода bound .02 rad. Поэтому direct/force — архитектурная ablation с разным физическим смыслом и масштабом ограничения.

Четыре latent-модели: самостоятельные размерности6/12, добавочная размерности6 и её phase-space вариант. Глобальный генератор задан как кососимметричная часть минус положительная диагональ затухания; переход использует преобразование Cayley с шагом omega(L)×.05. Это обучаемая модель вход–выход, не доказанное Koopman-представление. Устойчивость относится только к автономному скрытому переходу. Она не доказывает устойчивость базовой сети, крана или контроллера и не ограничивает общее усиление форсирующих/выходных матриц.

Вдохновение и ограничения переноса разобраны в [обзоре первоисточников](STRUCTURED_MODELS_LITERATURE_20261107.md); исходный [протокол](STRUCTURED_MODELS_DESIGN_20261107.md) зафиксирован до обучения. Добавление отдельной сети к замороженной модели связано с [KNODE online, §3.2–4](https://proceedings.mlr.press/v205/jiahao23a/jiahao23a.pdf), но здесь используется offline TRAIN, без онлайн-очереди и управления роботом. Условия связи скрытой и физической устойчивости обсуждаются в [Fan et al., §4–5](https://arxiv.org/html/2401.08153v2); у нашего идентификационного блока эти условия не установлены.

Входы:41 наблюдение при100Гц, текущая длина и30 известных будущих команд при20Гц. Future measured theta/x используются только как метки. Актор и TRAIN-калибровка длины неизменны. Потеря — SmoothL1((prediction−target)/.1,beta=.1) на30 шагах; phase добавляет .25 ошибки нормированных приращений и .25 ошибки конечного угла. Это surrogate, не фактическая угловая скорость или замкнутый cost. Adapter имеет дополнительный штраф .01 квадрата собственного bounded head; latent — без него. Семейства отличаются архитектурой и регуляризацией.

Headline: evaluation-сценарии с длиной вне TRAIN-диапазона. Равные веса целых сценариев и горизонтов {', '.join(f'{h:g}' for h in HORIZONS_S)}с; interpolation и нагрузка отдельно. Перекрывающиеся окна не считаются независимыми повторениями.

## Все методы: экстраполяция, MAE в градусах

{table(all_active.reset_index())}

Тихий режим, те же extrapolation-сценарии:

{table(quiet.reset_index())}

## Нагрузка в основном разбиении

{table(load.reset_index())}

## Выбранные прогнозы по отдельным extrapolation-записям

{table(record_table)}

Все методы и все режимы каждого сценария сохранены в `real_record_means.csv`; горизонты — `real_horizons.csv`, полные ячейки — `real_metrics_all.csv`. Графики `all_methods_extrapolation.png` и `selected_horizons.png` показывают общую MAE и зависимость от горизонта. Validation и interpolation присутствуют в отдельных summary-файлах и не смешиваются с headline.

## Предварительно заданные критерии принятия

{table(gates)}

И локальный, и общий выбор сравниваются с последним `previous_local`: primary all/active без ухудшения; short, long и primary load с допуском1%. Это практические пороги, не статистические границы. Они не используются для повторного выбора по evaluation. Сохранены все отрицательные результаты, неудачные обновления и fallback.

## Фактическое обучение

{table(fit_summary)}

Всего {int(fits.epochs_executed.sum())} эпох, {int(fits.optimizer_steps.sum())} optimizer updates, {int(fits.backward_passes.sum())} backward; суммарный fit {float(fits.fit_seconds.sum()):.3f}с. AdamW lr={cfg['training']['learning_rate']:g}, decay={cfg['training']['weight_decay']:g}, максимум{cfg['training']['epochs']}эпох, patience{cfg['training']['patience']}, batch{cfg['training']['batch_size']}, gradient clip{cfg['training']['gradient_clip']:g}. Adapter кеширует замороженные признаки; latent обновляет динамику. Одинаковый предел эпох не означает равные FLOPs. `fit_seconds` включает внутреннюю подготовку и validation; `runner_elapsed_seconds` дополнительно содержит сохранение/перезагрузку и проверку.

Полные fit и selected epochs — `fits_flat.csv`; обучаемые/замороженные количества приведены отдельно. Для standalone эпоха−1 — собственная физически ориентированная инициализация, для residual — точный soft; эти fallback нельзя объединять в утверждение «сохранены прежние прогнозы».

## Разброс seed

{table(seed_table)}

Показаны mean/std/min/max MAE отдельных seed и отдельная MAE ансамбля. Последняя сначала усредняет прогнозы, поэтому не равна средней seed MAE. Для добавочных моделей seeds включают разные frozen bases; у standalone это инициализации новой модели. Три seed не дают доверительный интервал переноса на неизвестную длину. Контролям без отдельных сохранённых seed-массивов не приписывается нулевой разброс.

## Крупные ошибки, поздний горизонт и приращения

{table(risk_table)}

`mean_of_record_packet_mae_p95_deg` — среднее отдельных p95 записей, **не pooled p95** окон. Peak — средний максимум абсолютной ошибки30 шагов пакета; tail — MAE последних10шагов; increment — ошибка приращения, делённая на omega×.05. Это диагностики прогноза, не нарушения физических ограничений. Все методы/режимы сохранены в `risk_summary.csv` и `risk_record_metrics.csv`.

## Новые точки симуляции

{table(simulation_table.reset_index())}

Число случаев по folds: {json.dumps(simulation_counts, ensure_ascii=False)}, всего{sum(simulation_counts.values())}. Все {len(methods)} методов и два псевдонима получают одно заданное ускорение; актор обойдён только для этой угловой проверки. `simulation_groups.csv` раздельно показывает force-law, drive, future-gain и measurement, включая p95/max ошибок случаев; дополнительные файлы разлагают частоту, затухание, длину и горизонт. Это новые точки прежнего семейства симулятора, без независимого аппаратного испытания.

## Replay, заморозка и сохранность

Полный replay из сохранённых весов: {replay['prediction_comparisons']} сравнений ({replay['ensemble_prediction_comparisons']} ансамблевых/контрольных и{replay['individual_seed_comparisons']} отдельных seed) на{replay['packet_count']} пакетах из{replay['real_record_files']} записей. Максимальное расхождение{replay['max_error_deg']:.6g}°. Входы заново собраны только из истории и известного плана; метки и средние ансамблей проверены отдельно.

{unit_note}Проверены{replay['parameter_audit']['models_checked']} сохранённых state_dict. Виды проверок: `{json.dumps(stability_types, ensure_ascii=False)}`. Learned Cayley, fixed damped filter и bounded direct head — разные проверки; у direct head нет рекуррентного оператора. Сводка аудита: `{json.dumps(stability_compact, ensure_ascii=False)}`. Полные матричные показатели и проверки замороженных bases/epoch−1 находятся в `replay_verification.json` и `parameter_and_stability_audit.csv`; численный аудит дополняет структурное доказательство и не является сертификатом управления. Исходный snapshot основной версии и APHYNITY: {replay['original_files']['checked_files']} совпадающих файлов. Этапные SHA/научные проверки: `{json.dumps(hash_counts, ensure_ascii=False)}`.

Runner также проверил reload, родительские прогнозы, подмену targets, независимую сборку history-only, сходимость интегратора и инвариантность к скрытому будущему gain. Проверки не используют evaluation для градиентов или выбора модели.

## Задержка прогнозирования

{latitude}

Run: `{run}`. Результаты: `{out}`. Архив создаётся отдельно после отчёта с SHA/CRC; полный replay требует сохранённых upstream данных и весов. Публикация в GitHub не выполняется генератором отчёта или архиватором и проверяется отдельно от научных результатов.

{bundle_note}
'''
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('artifacts/runs/structured_models_20261107'))
    parser.add_argument('--output', type=Path, default=Path('results/structured_models_20261107'))
    parser.add_argument('--document', type=Path, default=Path('docs/STRUCTURED_MODELS_2026_11_07_RU.md'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run, out, doc = root/args.run, root/args.output, root/args.document
    provenance, counts = verify_evidence(root, run)
    cfg = provenance['config']
    if (out/'report.json').exists() or doc.exists():
        raise FileExistsError('Preserve completed structured-model reports')
    selection = json.loads((run/'selection.json').read_text())
    if selection['uses_evaluation_this_stage'] or selection['uses_simulation']:
        raise ValueError('Selection used evaluation or simulation')
    expected_fits = len(cfg['folds'])*len(cfg['variants'])*len(cfg['seeds'])
    replay, latency = external_verification(out, expected_fits)
    unit_tests = unit_test_record(out)
    out.mkdir(parents=True, exist_ok=True)
    audits = audit_tables(out, replay)
    methods, real, summary, horizon, records, risks = real_tables(run, cfg, out, selection)
    fits, fit_summary = fit_tables(run, cfg, out)
    seeds = seed_tables(root, run, cfg, out, real, summary)
    verdict = acceptance(summary, cfg, selection, out)
    simulation_counts, simulation = simulation_tables(run, cfg, out, methods)
    for name in ['selection.json', 'fits.json', 'training.csv', 'train_verification.json',
                 'evaluate_verification.json', 'train_timings.csv', 'evaluate_timings.csv']:
        shutil.copy2(run/name, out/name)
    figures(out, summary, horizon, methods, cfg, selection)
    document(doc, args.run, args.output, cfg, methods, summary, horizon, records, risks,
        fits, fit_summary, seeds, selection, verdict, simulation_counts, simulation, counts,
        replay, latency, audits, unit_tests)
    report = dict(source_provenance_sha256=sha256(run/'provenance.json'),
        report_source_sha256=sha256(Path(__file__)),
        aggregation_helper_sha256=sha256(root/'scripts/report_finetune_campaign.py'),
        document=str(doc.relative_to(root)), document_sha256=sha256(doc), verified_hash_counts=counts,
        replay_verification_sha256=sha256(out/'replay_verification.json'),
        unit_tests=unit_tests,
        new_fits=len(fits), methods=methods, selection=selection, acceptance=verdict,
        epoch_minus_one_count=int(fits.retained_starting_checkpoint.sum()),
        exact_soft_fallback_count=int(fits.fallback_equals_soft.sum()),
        fit_seconds=float(fits.fit_seconds.sum()), epochs_executed=int(fits.epochs_executed.sum()),
        optimizer_steps=int(fits.optimizer_steps.sum()), backward_passes=int(fits.backward_passes.sum()),
        simulation_cases_per_fold=simulation_counts, seed_note=SEED_NOTE,
        risk_quantile_aggregation='Mean of record-wise packet MAE p95; not pooled p95',
        real_records_retrospective=True, closed_loop_control_evaluated=False,
        headline_scope='evaluation_extrapolation')
    if latency is not None:
        report['inference_latency_sha256'] = sha256(out/'inference_latency.json')
    write(out/'report.json', report)
    print(summary[(summary.scope == 'evaluation_extrapolation') & summary.model.isin(ALIASES)].to_string(index=False))
    print('ACCEPTANCE', json.dumps(verdict, indent=2))
    print('FITS', len(fits), 'EPOCH_MINUS_ONE', report['epoch_minus_one_count'])


if __name__ == '__main__':
    main()
