"""Verify and report the frozen global-model study without fitting or selection."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from cranepilot.rollout import HORIZONS_S

CONTROLS = ['incumbent', 'soft_symmetry', 'combined', 'physical']
REGIMES = ['all', 'active', 'quiet']
ALIASES = ['selected', 'common']
HORIZONS = list(HORIZONS_S)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def verify_evidence(root, run):
    """Verify retained source, fitted models, frozen selection, and complete results."""
    prov = json.loads((run/'provenance.json').read_text())
    if prov['status'] != 'complete':
        raise ValueError('Completed evaluation required; do not report a partial run')
    counts = {}
    for field, base in [('guarded_hashes', root), ('checkpoint_hashes', run), ('result_hashes', run)]:
        hashes = prov[field]
        if not hashes:
            raise ValueError(f'Empty evidence manifest: {field}')
        for name, digest in hashes.items():
            path = base/name.replace('\\', '/')
            if sha256(path) != digest:
                raise ValueError(f'Changed {field}: {name}')
        counts[field] = len(hashes)
    actual = {str(p.relative_to(run)) for p in run.rglob('*') if p.is_file() and p.name != 'provenance.json'}
    if actual != {name.replace('\\', '/') for name in prov['result_hashes']}:
        raise ValueError('Result manifest does not cover exactly the completed run')
    if sha256(run/'selection.json') != prov['selection_sha256']:
        raise ValueError('Frozen selection changed')
    for key in ['source_run', 'broad_run']:
        folder = root/prov['config'][key]
        upstream = json.loads((folder/'provenance.json').read_text())
        for name, digest in upstream['checkpoint_hashes'].items():
            if sha256(folder/name.replace('\\', '/')) != digest:
                raise ValueError(f'Changed upstream checkpoint: {key}/{name}')
        counts[key+'_checkpoints'] = len(upstream['checkpoint_hashes'])
    for stage in ['train', 'evaluate']:
        verification = json.loads((run/f'{stage}_verification.json').read_text())
        if verification['passed'] is not True:
            raise ValueError(f'{stage} verification failed')
    return prov, counts


def bool_column(series):
    result = series.astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})
    if result.isna().any():
        raise ValueError('Invalid extrapolation flag')
    return result.astype(bool)


def complete_horizons(frame, keys, value):
    if frame.empty or frame.duplicated(keys+['horizon_s']).any():
        raise ValueError('Empty or duplicate metric cells')
    if not np.isfinite(frame[value]).all() or (frame[value] < 0).any():
        raise ValueError('Finite nonnegative errors required')
    for _, group in frame.groupby(keys, dropna=False):
        if len(group) != len(HORIZONS) or not np.allclose(sorted(group.horizon_s), HORIZONS, atol=1e-8):
            raise ValueError('Exactly the declared seven horizons required per record/case')


def compare(frame, keys, references=CONTROLS):
    result = frame.copy()
    if result.empty:
        return result
    for name in references:
        reference = frame[frame.model == name][keys+['mae_deg']].rename(columns={'mae_deg': name+'_mae_deg'})
        result = result.merge(reference, on=keys, how='left', validate='many_to_one')
        if result[name+'_mae_deg'].isna().any() or (result[name+'_mae_deg'] <= 0).any():
            raise ValueError(f'Missing or zero comparison denominator: {name}')
        result['change_vs_'+name+'_percent'] = 100*(result.mae_deg/result[name+'_mae_deg']-1)
    return result


def read_model_metadata(run, cfg, out):
    weak, delay, all_metadata, manifests = [], [], [], {}
    for fold in cfg['folds']:
        manifest = json.loads((run/fold/'models.json').read_text())
        manifests[fold] = manifest
        if len(manifest) != 15:
            raise ValueError('Expected 15 new fits per fold')
        for name, spec in manifest.items():
            path = run/fold/spec['file']
            if spec['family'] == 'weak':
                metadata = json.loads(path.read_text())
                weak.append({'fold': fold, 'model': name, **{key: metadata[key] for key in
                    ['ridge', 'c', 'kscale', 'bscale', 'rows', 'data_rank', 'identified_full_rank',
                     'scaled_data_mse', 'ridge_penalty']},
                    'training_sids': ','.join(map(str, metadata['training_sids']))})
            else:
                with np.load(path, allow_pickle=False) as saved:
                    metadata = json.loads(str(saved['metadata_json'].item()))
                    if saved['coef'].shape != (metadata['feature_dim'], 30):
                        raise ValueError('Model dimensions disagree with saved coefficients')
                delay.append({'fold': fold, 'model': name, **{key: metadata[key] for key in
                    ['basis', 'input_mode', 'alpha', 'feature_dim', 'base_dim', 'training_rows',
                     'positive_weight_rows', 'numerical_rank', 'svd_driver', 'outputs']}})
            all_metadata.append({'fold': fold, 'model': name, 'spec': spec, 'metadata': metadata})
    if len(weak) != 9 or len(delay) != 36:
        raise ValueError('Expected 9 weak and 36 direct saved fits across three folds')
    methods = CONTROLS+list(manifests[cfg['folds'][0]])
    if len(methods) != 19 or any(set(CONTROLS+list(m)) != set(methods) for m in manifests.values()):
        raise ValueError('Fixed 19-method set differs across folds')
    pd.DataFrame(weak).to_csv(out/'weak_parameters.csv', index=False)
    pd.DataFrame(delay).to_csv(out/'direct_fit_metadata.csv', index=False)
    write(out/'all_fit_metadata.json', all_metadata)
    return methods, manifests, pd.DataFrame(weak), pd.DataFrame(delay)


def fit_timings(run, manifests, out):
    source = json.loads((run/'fitting.json').read_text())
    if len(source) != 45 or len({(r['fold'], r['model']) for r in source}) != 45:
        raise ValueError('Expected 45 unique fit entries')
    grouped = {}
    for row in source:
        spec = manifests[row['fold']][row['model']]
        if spec['family'] == 'weak':
            key = (row['fold'], 'weak', row['model'], '')
            seconds = row['seconds']
        else:
            key = (row['fold'], spec['basis'], spec['basis'], spec['input_mode'])
            seconds = row['path_fit_seconds']
        if key in grouped and grouped[key]['seconds'] != seconds:
            raise ValueError('Shared SVD path timing differs between alpha entries')
        group = grouped.setdefault(key, {'fold': key[0], 'family': key[1], 'fit_unit': key[2],
                                        'input_mode': key[3], 'seconds': seconds, 'models': []})
        group['models'].append(row['model'])
    rows = [{**v, 'fitted_models': len(v['models']), 'models': ','.join(v['models'])} for v in grouped.values()]
    frame = pd.DataFrame(rows)
    if len(frame) != 21:
        raise ValueError('Expected 9 separate weak fits and 12 shared SVD paths')
    frame.to_csv(out/'fitting_time_units.csv', index=False)
    totals = frame.groupby(['fold', 'family'], as_index=False).agg(seconds=('seconds', 'sum'),
        fitted_models=('fitted_models', 'sum'), timed_units=('fit_unit', 'size'))
    totals.to_csv(out/'fitting_time_summary.csv', index=False)
    return totals


def real_tables(run, out, methods, selection):
    frames = []
    for stage in ['train', 'evaluate']:
        frame = pd.read_csv(run/f'{stage}_metrics.csv')
        frame['extrapolation'] = bool_column(frame.extrapolation)
        expected = set(methods if stage == 'train' else methods+ALIASES)
        for _, part in frame.groupby(['fold', 'group', 'scenario', 'regime']):
            if set(part.model) != expected:
                raise ValueError('Missing methods in a matched real-record cell')
        if stage == 'train':
            aliases = []
            for fold, part in frame.groupby('fold'):
                for alias, chosen in [('selected', selection['folds'][fold]['model']), ('common', selection['common']['model'])]:
                    copy = part[part.model == chosen].copy(); copy['model'] = alias; aliases.append(copy)
            frame = pd.concat([frame, *aliases], ignore_index=True)
        frames.append(frame)
    real = pd.concat(frames, ignore_index=True)
    real['scope'] = np.where(real.group == 'evaluation',
        np.where(real.extrapolation, 'evaluation_extrapolation', 'evaluation_interpolation'), real.group)
    keys = ['scope', 'fold', 'group', 'scenario', 'extrapolation', 'model', 'regime']
    complete_horizons(real, keys, 'mae_deg')
    real.to_csv(out/'real_metrics_all.csv', index=False)
    records = real.groupby(keys, as_index=False).mae_deg.mean()
    records.to_csv(out/'real_record_means.csv', index=False)
    # Equal scenario weight at every horizon, then equal weight across seven horizons.
    horizon = real.groupby(['scope', 'fold', 'model', 'regime', 'horizon_s'], as_index=False).agg(
        mae_deg=('mae_deg', 'mean'), scenarios=('scenario', 'nunique'))
    horizon.to_csv(out/'real_horizons.csv', index=False)
    summary = horizon.groupby(['scope', 'fold', 'model', 'regime'], as_index=False).agg(
        mae_deg=('mae_deg', 'mean'), scenarios=('scenarios', 'min'))
    summary = compare(summary, ['scope', 'fold', 'regime'])
    summary.to_csv(out/'real_summary_all_scopes.csv', index=False)
    for scope in summary.scope.unique():
        summary[summary.scope == scope].to_csv(out/f'{scope}_summary.csv', index=False)
    return summary, horizon


def acceptance(summary, cfg, selection, out):
    cells = []
    for fold in cfg['folds']:
        reference = 'soft_symmetry' if fold == 'primary' else 'incumbent'
        threshold = cfg['acceptance']['primary_all_active_vs_soft' if fold == 'primary' else 'transfer_all_active_vs_incumbent']
        for regime in ['all', 'active']:
            row = summary[(summary.scope == 'evaluation_extrapolation') & (summary.fold == fold)
                          & (summary.model == 'common') & (summary.regime == regime)]
            if len(row) != 1: raise ValueError('Missing extrapolation acceptance cell')
            value = row.iloc[0]; ratio = float(value.mae_deg/value[reference+'_mae_deg'])
            cells.append({'scope': 'evaluation_extrapolation', 'fold': fold, 'regime': regime,
                          'reference': reference, 'ratio': ratio, 'maximum_ratio': threshold, 'passed': ratio <= threshold})
    for regime in ['all', 'active']:
        row = summary[(summary.scope == 'load_stress') & (summary.fold == 'primary')
                      & (summary.model == 'common') & (summary.regime == regime)]
        if len(row) != 1: raise ValueError('Missing primary load acceptance cell')
        value = row.iloc[0]; ratio = float(value.mae_deg/value.incumbent_mae_deg)
        threshold = cfg['acceptance']['load_all_active_vs_incumbent']
        cells.append({'scope': 'load_stress', 'fold': 'primary', 'regime': regime, 'reference': 'incumbent',
                      'ratio': ratio, 'maximum_ratio': threshold, 'passed': ratio <= threshold})
    verdict = {'common_model': selection['common']['model'], 'cells': cells,
               'passed': all(row['passed'] for row in cells), 'operational_thresholds_not_statistical_bounds': True}
    write(out/'acceptance.json', verdict); pd.DataFrame(cells).to_csv(out/'acceptance.csv', index=False)
    return verdict


def simulation_tables(run, out, manifests, cfg):
    simulation = pd.read_csv(run/'simulation_metrics.csv')
    case_keys = ['fold', 'model', 'force_law', 'drive_amplitude_m', 'future_acceleration_gain', 'measurement',
                 'omega_ratio', 'damping_per_s', 'length_m', 'amplitude_deg', 'phase_rad', 'trajectory_id']
    complete_horizons(simulation, case_keys, 'absolute_error_deg')
    scopes = {}
    for fold in cfg['folds']:
        scope = json.loads((run/fold/'simulation_scope.json').read_text())
        excluded = {name for name, spec in manifests[fold].items() if spec.get('input_mode') == 'command'}
        allowed = set(CONTROLS+list(manifests[fold]))-excluded
        if set(scope['excluded_command_models']) != excluded or set(scope['angular_methods']) != allowed:
            raise ValueError('Simulation scope disagrees with saved model types')
        part = simulation[simulation.fold == fold]
        if set(part.model) != allowed:
            raise ValueError('Simulation includes undeclared methods or omits declared methods')
        for _, cell in part.groupby([key for key in case_keys if key not in ['model', 'fold']]):
            if set(cell.model) != allowed: raise ValueError('Unmatched simulation method cases')
        scopes[fold] = scope
    cases = simulation.groupby(case_keys, as_index=False).absolute_error_deg.mean().rename(columns={'absolute_error_deg': 'mae_deg'})
    cases.to_csv(out/'simulation_case_means.csv', index=False)
    groups = ['fold', 'model', 'force_law', 'drive_amplitude_m', 'future_acceleration_gain', 'measurement']
    grouped = cases.groupby(groups, as_index=False).agg(mae_deg=('mae_deg', 'mean'),
        median_case_mae_deg=('mae_deg', 'median'), case_p95_mae_deg=('mae_deg', lambda x: float(x.quantile(.95))),
        worst_case_mae_deg=('mae_deg', 'max'), cases=('mae_deg', 'size'))
    grouped = compare(grouped, [k for k in groups if k != 'model'])
    grouped.to_csv(out/'simulation_groups.csv', index=False)
    for suffix, extras in [('frequency_damping', ['omega_ratio', 'damping_per_s']), ('lengths', ['length_m'])]:
        cases.groupby(groups+extras, as_index=False).agg(mae_deg=('mae_deg', 'mean'), cases=('mae_deg', 'size')).to_csv(out/f'simulation_{suffix}.csv', index=False)
    simulation.groupby(groups+['horizon_s'], as_index=False).absolute_error_deg.mean().to_csv(out/'simulation_horizons.csv', index=False)
    counts = {fold: int(len(part[case_keys[2:]].drop_duplicates())) for fold, part in cases.groupby('fold')}
    write(out/'simulation_scope.json', scopes)
    return grouped, counts, scopes


def figures(out, summary, horizon, methods, selection):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    subset = summary[(summary.scope == 'evaluation_extrapolation') & (summary.fold == 'primary')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 9), sharey=True)
    colors = ['#64748b' if n in CONTROLS else '#bd7a35' if n.startswith('weak_') else
              '#268a86' if n.startswith('lpv_') else '#85549a' for n in methods]
    for ax, regime in zip(axes, REGIMES):
        part = subset[subset.regime == regime].set_index('model').loc[methods]
        ax.barh(np.arange(len(methods)), part.mae_deg, color=colors)
        ax.set_title({'all': 'Все окна', 'active': 'Движение', 'quiet': 'Покой'}[regime])
        ax.set_xlabel('MAE угла, °'); ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    axes[0].set_yticks(np.arange(len(methods)), [n+(' *' if n == selection['common']['model'] else '') for n in methods])
    axes[0].invert_yaxis()
    fig.suptitle('Основной набор: все 19 методов на ретроспективной экстраполяции', fontsize=13)
    fig.text(.02, .012, '* Общий метод выбран только по валидации; одинаковый вес записей и семи горизонтов.', fontsize=9)
    fig.tight_layout(rect=[0, .035, 1, .96]); fig.savefig(out/'primary_all_methods.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    names = list(dict.fromkeys(['incumbent', 'soft_symmetry', 'physical', 'common']))
    for ax, fold in zip(axes, selection['folds']):
        part = horizon[(horizon.scope == 'evaluation_extrapolation') & (horizon.fold == fold) & (horizon.regime == 'all')]
        for name in names:
            line = part[part.model == name].sort_values('horizon_s')
            ax.plot(line.horizon_s, line.mae_deg, marker='o', ms=3, label=name)
        ax.set_title(fold); ax.set_xlabel('Горизонт, с'); ax.grid(alpha=.2)
    axes[0].set_ylabel('MAE угла, °'); axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle('Общий выбор: '+selection['common']['model']+'; ретроспективная экстраполяция', fontsize=12)
    fig.tight_layout(); fig.savefig(out/'common_horizons.png', dpi=180); plt.close(fig)


def markdown_table(frame):
    columns = [str(c) for c in frame.columns]
    def cell(value):
        if isinstance(value, (float, np.floating)): return f'{value:.6g}'
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |',
                      *['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False, name=None)]])


def short_transfer_tables(out, summary, horizon, selection):
    chosen = selection['folds']['short_transfer']['model']
    records = pd.read_csv(out/'real_record_means.csv')
    records = records[(records.scope == 'evaluation_extrapolation') & (records.fold == 'short_transfer')
                      & records.model.isin(['incumbent', chosen])]
    record_table = records.pivot(index=['scenario', 'regime'], columns='model', values='mae_deg').reset_index()
    record_table['change_percent'] = 100*(record_table[chosen]/record_table.incumbent-1)
    record_table.to_csv(out/'short_transfer_record_deltas.csv', index=False)
    short = horizon[(horizon.scope == 'evaluation_extrapolation') & (horizon.fold == 'short_transfer')
                    & horizon.model.isin(['incumbent', chosen])]
    table = short.pivot(index=['regime', 'horizon_s'], columns='model', values='mae_deg').reset_index()
    table['change_percent'] = 100*(table[chosen]/table.incumbent-1)
    table.to_csv(out/'short_transfer_horizon_deltas.csv', index=False)
    return record_table, table


def document(path, run, out, summary, weak, delay, times, selection, verdict, counts, hash_counts, short_records, short_horizons):
    primary = summary[(summary.scope == 'evaluation_extrapolation') & (summary.fold == 'primary')]
    primary_table = primary.pivot(index='model', columns='regime', values='mae_deg').reset_index()
    primary_table = primary_table[['model', 'all', 'active', 'quiet']]
    transfers = summary[(summary.scope == 'evaluation_extrapolation') & (summary.fold != 'primary') & (summary.regime.isin(['all', 'active']))]
    transfer_table = transfers.pivot(index='model', columns=['fold', 'regime'], values='mae_deg')
    transfer_table.columns = [' / '.join(c) for c in transfer_table.columns]; transfer_table = transfer_table.reset_index()
    load = summary[(summary.scope == 'load_stress') & (summary.fold == 'primary')]
    load_table = load.pivot(index='model', columns='regime', values='mae_deg').reset_index()
    ranks = delay.groupby(['fold', 'basis', 'input_mode'], as_index=False).agg(
        D=('feature_dim', 'first'), rank=('numerical_rank', 'first'), windows=('training_rows', 'first'), fits=('model', 'size'))
    verdict_text = 'выполнены' if verdict['passed'] else 'не выполнены'
    failed = [f"{r['fold']}/{r['scope']}/{r['regime']}: {r['ratio']:.6f} > {r['maximum_ratio']:.6f}" for r in verdict['cells'] if not r['passed']]
    chosen = selection['folds']['short_transfer']['model']
    short_rows = summary[(summary.scope == 'evaluation_extrapolation') & (summary.fold == 'short_transfer') & (summary.model == 'selected')].set_index('regime')
    all_row, active_row = short_rows.loc['all'], short_rows.loc['active']
    all_horizons = short_horizons[short_horizons.regime == 'all']
    active_horizons = short_horizons[short_horizons.regime == 'active']
    worse_all = ', '.join(f'{v:g}' for v in all_horizons[all_horizons.change_percent > 0].horizon_s)
    worse_active = ', '.join(f'{v:g}' for v in active_horizons[active_horizons.change_percent > 0].horizon_s)
    all_records = short_records[short_records.regime == 'all']
    text = f'''# Глобальные физические и полиномиальные модели — 7 ноября 2026

Новый локальный результат: выбранная по валидации модель `{chosen}` улучшила перенос на короткие тросы. Средняя MAE всех окон: {all_row.incumbent_mae_deg:.6f} → {all_row.mae_deg:.6f}° ({all_row.change_vs_incumbent_percent:+.2f}%); при движении: {active_row.incumbent_mae_deg:.6f} → {active_row.mae_deg:.6f}° ({active_row.change_vs_incumbent_percent:+.2f}%). Улучшение среднего всех окон есть в {int((all_records.change_percent < 0).sum())} из {len(all_records)} записей. Это локальный выигрыш, а не универсальная замена модели.

По горизонтам выигрыш неоднороден: для всех окон улучшены {int((all_horizons.change_percent < 0).sum())} из 7 горизонтов, ухудшены {worse_all} с; при движении улучшены {int((active_horizons.change_percent < 0).sum())} из 7, ухудшены {worse_active} с. Полные значения ниже и в `short_transfer_horizon_deltas.csv`; изменение среднего нельзя читать как выигрыш каждого краткосрочного прогноза.

Общий вариант `{selection['common']['model']}` выбран по валидации до оценки. Условия перехода на общий вариант **{verdict_text}**. Локальный выбор: {', '.join(f"{fold}: `{v['model']}`" for fold, v in selection['folds'].items())}.

Это ретроспективное сравнение на ранее использованных реальных записях. Здесь нет нового нетронутого теста, независимого аппаратного испытания или основания для заявления SOTA. Новая симуляция — угловая диагностика с заданным будущим ускорением; она не проверяет актор.

## Протокол и область вывода

Обучены 45 новых детерминированных моделей: 9 интегральных физических, 18 линейных LPV и 18 квадратичных моделей задержек. Это 15 вариантов на каждом из трёх разбиений, плюс четыре сохранённых контроля: `incumbent`, `soft_symmetry`, `combined`, `physical`. Сети и актор заново не обучались. Слабая форма — ограниченный линейный МНК с фиксированными интегральными строками, без EIV/IRLS; это не воспроизведение WENDy. Квадратичная модель — прямой прогноз остатка по задержкам, без гарантии рекурсивной устойчивости или интерпретации оператора Koopman.

Параметры, масштабы признаков и коэффициенты оценены только на TRAIN. Валидация выбирает вариант; оценка и симуляция не используются для выбора на этом этапе. При прогнозе доступны 41 прошлый отсчёт и план команды. Прямая модель использует весь разрешённый план, поэтому причинность отдельного горизонта относительно более поздних команд не заявляется. Управляющая модель `command` исключает актор только из входов поправки; номинальная физическая база по-прежнему использует актор. Входной режим `actor` использует его прогноз ускорения.

MAE усредняется с одинаковым весом сценариев на каждом из семи горизонтов {', '.join(f'{h:g}' for h in HORIZONS)} с, затем с одинаковым весом горизонтов. Заголовочные таблицы и условия перехода используют **evaluation/extrapolation**. Интерполяция сохранена отдельно и не смешана с переносом. `selected` и `common` — псевдонимы уже замороженных решений, не дополнительные модели.

## Короткие тросы: локальный выбор по записям и горизонтам

{markdown_table(short_records)}

{markdown_table(short_horizons)}

## Основной набор: MAE, градусы

{markdown_table(primary_table)}

## Перенос за пределы обучающих длин: MAE, градусы

{markdown_table(transfer_table)}

## Нагрузка, основной набор: MAE, градусы

{markdown_table(load_table)}

## Условия перехода для общего варианта

{markdown_table(pd.DataFrame(verdict['cells']))}

Порог — практическое правило, не статистическая граница. На основном наборе нужны all/active не хуже `soft_symmetry`; на экстраполяционном переносе и нагрузке — не хуже `incumbent` более чем на 1%. {'Нарушения: '+ '; '.join(failed)+'.' if failed else 'Все восемь проверок выполнены.'}

## Физические коэффициенты

Уравнение: θ″ + c θ′ + kscale·ω(L)² θ = −bscale·ω(L)²/g·x″. Коэффициенты глобальные для разбиения, при прогнозе не переоцениваются. Ridge направлен к (c, kscale, bscale) = (0, 1, 1).

{markdown_table(weak[['fold', 'model', 'ridge', 'c', 'kscale', 'bscale', 'data_rank', 'identified_full_rank']])}

## Фактические размеры и ранги прямых моделей

{markdown_table(ranks)}

Это данные из сохранённых метаданных и размеров коэффициентов, а не номинальная оценка сложности. Три alpha на каждой строке используют одну SVD. Штраф действует на коэффициенты признаков, нормированных по TRAIN RMS, с усреднением ошибки по 30 выходам. Перезапуск по разным seed не выполнялся: решение детерминировано.

## Время оценивания параметров

{markdown_table(times)}

Время каждой общей SVD-траектории учтено **один раз**, хотя она сохраняет три alpha. Сумма отдельных физических fit и общих SVD: {times.seconds.sum():.3f} с. Это измеренное время блоков fit, без загрузки данных и полного времени запуска. Время пакетных прогнозов всех 19 методов сохранено отдельно; это не задержка одного развёрнутого метода.

## Свежая угловая симуляция

Число случаев на разбиение: {json.dumps(counts, ensure_ascii=False)}. Оценены только 13 разрешённых вариантов: четыре контроля, три физических и шесть прямых с ускорением. Шесть моделей с командами исключены: симулятор не задаёт актор и реалистичную опорную команду. `selected`/`common` не добавлены в эту таблицу как новые методы. Все угловые методы получают одно и то же заданное будущее ускорение. Скрытое изменение будущего коэффициента действует на цель и не доступно прогнозу.

`simulation_case_means.csv` хранит среднее семи горизонтов каждого случая. `simulation_groups.csv` сохраняет отдельно движение/нулевое движение, закон связи (effective/geometric), коэффициент будущего ускорения и режим измерения (exact/noisy_quantized). Там же приведены медиана, 95-й процентиль и максимум ошибок случаев. Это описательные показатели этой сетки; p-value и перенос на реальное распределение не заявляются. Результаты по длине, частоте, затуханию и горизонту сохранены в отдельных CSV. Разбиения используют свои калибровки, поэтому их симуляционные цели не обязаны совпадать.

## Проверка и файлы

Проверены SHA-256 исходников и защищённых входов, всех моделей, замороженного выбора, полного набора результатов и родительских нейросетевых весов: {json.dumps(hash_counts, ensure_ascii=False)}. Сохранённые проверки включают совпадение четырёх контролей с прошлым этапом, независимый вход из 41 отсчёта, инвариантность к подмене будущих целей, сходимость интегратора и недоступность скрытого будущего коэффициента. Они относятся к исполняемому протоколу и не заменяют независимую оценку качества.

Артефакты запуска: `{run}`. Таблицы и графики: `{out}`. Полные таблицы содержат все 19 методов и оба псевдонима для реальных данных, включая валидацию, интерполяцию, экстраполяцию и нагрузку. Архив этого этапа создаётся отдельно после отчёта, с проверкой CRC и SHA-256; полное воспроизведение также требует сохранённых данных и весов предыдущих этапов.
'''
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('artifacts/runs/global_research_20261107'))
    parser.add_argument('--output', type=Path, default=Path('results/global_research_20261107'))
    parser.add_argument('--document', type=Path, default=Path('docs/GLOBAL_RESEARCH_2026_11_07_RU.md'))
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    run = root/args.run; out = root/args.output; doc = root/args.document
    prov, hash_counts = verify_evidence(root, run); cfg = prov['config']
    if (out/'report.json').exists() or doc.exists():
        raise FileExistsError('Preserve completed report and research document')
    out.mkdir(parents=True, exist_ok=True)
    selection = json.loads((run/'selection.json').read_text())
    if selection['uses_evaluation_this_stage'] or selection['uses_simulation']:
        raise ValueError('Selection must be locked independently of evaluation and simulation')
    methods, manifests, weak, delay = read_model_metadata(run, cfg, out)
    times = fit_timings(run, manifests, out)
    summary, horizon = real_tables(run, out, methods, selection)
    verdict = acceptance(summary, cfg, selection, out)
    short_records, short_horizons = short_transfer_tables(out, summary, horizon, selection)
    _, counts, sim_scopes = simulation_tables(run, out, manifests, cfg)
    for name in ['selection.json', 'train_verification.json', 'evaluate_verification.json', 'train_timings.csv', 'evaluate_timings.csv', 'fitting.json']:
        shutil.copy2(run/name, out/name)
    figures(out, summary, horizon, methods, selection)
    document(doc, args.run, args.output, summary, weak, delay, times, selection, verdict, counts, hash_counts, short_records, short_horizons)
    report = {'source_provenance_sha256': sha256(run/'provenance.json'), 'report_source_sha256': sha256(Path(__file__)),
              'document': str(doc.relative_to(root)), 'document_sha256': sha256(doc), 'verified_hash_counts': hash_counts,
              'methods': methods, 'new_fit_models': 45, 'fit_families': {'weak': 9, 'lpv_linear': 18, 'nvar_quadratic': 18},
              'selection': selection, 'acceptance': verdict, 'history_samples': 41,
              'real_records_retrospective': True, 'neural_training': False, 'simulation_actor_bypassed': True,
              'simulation_cases_per_fold': counts, 'simulation_scopes': sim_scopes,
              'fit_seconds_without_duplicate_svd': float(times.seconds.sum()),
              'aggregation': 'equal scenarios at each horizon, then equal seven horizons; simulation equal cases within declared groups',
              'headline_scope': 'evaluation_extrapolation', 'hardware_evidence': False}
    write(out/'report.json', report)
    print(summary[(summary.scope == 'evaluation_extrapolation') & summary.model.isin(['selected', 'common'])].to_string(index=False))
    print('ACCEPTANCE', json.dumps(verdict, indent=2))


if __name__ == '__main__':
    main()
