"""Post-hoc descriptive horizon audit of frozen dense-reference CSV metrics.

No data/model fitting, prediction generation, gating, routing, or selection.
All declared ages, noises, horizons and physical records remain in the output.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'results/dense_reference_20261108'
OUT = ROOT/'results/dense_horizons_20261109'
DOC = ROOT/'docs/DENSE_HORIZONS_2026_11_09_RU.md'
METHODS = ['selected', 'dense_matched_arx', 'matched_arx', 'parent_quad', 'bank_simplex_1']
HORIZONS = [.1, .2, .35, .5, .75, 1., 1.5]
TOLERANCE = 1e-12


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def table(frame):
    def cell(value):
        return f'{value:.6g}' if isinstance(value, (float, np.floating)) else str(value)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |',
        '| '+' | '.join(['---']*len(frame.columns))+' |',
        *['| '+' | '.join(map(cell, row))+' |' for row in frame.itertuples(index=False, name=None)]])


def main():
    if OUT.exists() or DOC.exists():
        raise FileExistsError('Preserve previous diagnostic; never overwrite')
    provenance_path = PARENT/'frozen_replay/provenance.json'
    metrics_path = PARENT/'frozen_replay/metrics.csv'
    development_path = PARENT/'frozen_development/development.json'
    provenance = json.loads(provenance_path.read_text())
    development = json.loads(development_path.read_text())
    verification = json.loads((PARENT/'verification.json').read_text())
    previous_report = json.loads((PARENT/'report.json').read_text())
    if provenance['status'] != 'complete' or verification['passed'] is not True:
        raise ValueError('Completed and verified frozen replay required')
    if digest(metrics_path) != provenance['result_hashes']['metrics.csv']:
        raise ValueError('Archived metrics digest mismatch')
    if digest(provenance_path) != verification['retrospective_sha256']:
        raise ValueError('Archived replay provenance differs from verification')
    if digest(development_path) != verification['development_sha256']:
        raise ValueError('Archived development differs from verification')
    if digest(PARENT/'verification.json') != previous_report['verification_sha256']:
        raise ValueError('Parent report verification changed')
    cfg, task = development['config'], development['task']
    ages = sorted(a*10 for a in cfg['evaluation_ages_raw'])
    noises = sorted(cfg['noise_quantum_multipliers'])
    if len(ages) != 9 or noises != [0, 1]:
        raise ValueError('Expected complete frozen nine-age/two-noise contract')
    pairs = {(f'{family}_{axis}', int(sid)) for family, fold in task['families'].items()
             for axis in fold['axes'] for sid in fold['confirmation']}
    scenarios = sorted({sid for _, sid in pairs})
    if len(scenarios) != 7:
        raise ValueError('Expected seven retained physical records')
    source = pd.read_csv(metrics_path)
    frame = source[(source['group'] == 'all') & source.method.isin(METHODS)].copy()
    if not (frame.role == 'old_evaluation').all():
        raise ValueError('Unexpected roles in retrospective metrics')
    frame['horizon_s'] = frame.horizon_s.round(12)
    keys = ['age_ms', 'noise_quantum', 'case', 'scenario', 'method', 'horizon_s']
    expected = {(age, noise, case, sid, method, horizon)
                for age, noise, (case, sid), method, horizon in
                itertools.product(ages, noises, pairs, METHODS, HORIZONS)}
    observed = set(frame[keys].itertuples(index=False, name=None))
    if frame.duplicated(keys).any() or observed != expected or len(frame) != len(expected):
        raise ValueError('Incomplete or duplicate paired age/noise/axis/horizon cells')
    if not np.isfinite(frame.mae_deg).all() or (frame.mae_deg < 0).any() or (frame.windows <= 0).any():
        raise ValueError('Invalid metric cells; no dropping permitted')
    # Every method and horizon must use the same original origins for each axis.
    if not (frame.groupby(['age_ms', 'noise_quantum', 'case', 'scenario']).windows.nunique() == 1).all():
        raise ValueError('Unmatched method/horizon window support')
    if not (frame.groupby(['case', 'scenario']).windows.nunique() == 1).all():
        raise ValueError('Age/noise changed the scored window count')
    records = frame.groupby(['age_ms', 'noise_quantum', 'scenario', 'method', 'horizon_s'], as_index=False).agg(
        mae_deg=('mae_deg', 'mean'), axes=('case', 'nunique'), windows_per_axis=('windows', 'first'))
    summary = records.groupby(['age_ms', 'noise_quantum', 'method', 'horizon_s'], as_index=False).agg(
        mae_deg=('mae_deg', 'mean'), physical_records=('scenario', 'nunique'))
    if not (summary.physical_records == len(scenarios)).all():
        raise ValueError('Physical record coverage changed')
    paired = summary.pivot(index=['age_ms', 'noise_quantum', 'horizon_s'], columns='method', values='mae_deg').reset_index()
    paired_records = records.pivot(index=['age_ms', 'noise_quantum', 'scenario', 'horizon_s'], columns='method', values='mae_deg').reset_index()
    for comparator in METHODS[1:]:
        paired['selected_minus_'+comparator] = paired.selected-paired[comparator]
        paired_records['selected_minus_'+comparator] = paired_records.selected-paired_records[comparator]
        for label, predicate in [('wins', lambda v: v < -TOLERANCE), ('losses', lambda v: v > TOLERANCE),
                                 ('ties', lambda v: v.abs() <= TOLERANCE)]:
            grouped = paired_records.assign(count=predicate(paired_records['selected_minus_'+comparator])).groupby(
                ['age_ms', 'noise_quantum', 'horizon_s'], as_index=False)['count'].sum()
            grouped = grouped.rename(columns={'count': label+'_vs_'+comparator})
            paired = paired.merge(grouped, on=['age_ms', 'noise_quantum', 'horizon_s'], validate='one_to_one')
    recombined = summary.groupby(['age_ms', 'noise_quantum', 'method'], as_index=False).mae_deg.mean()
    old = pd.read_csv(PARENT/'summary.csv')
    old = old[(old['group'] == 'all') & old.method.isin(METHODS)]
    joined = recombined.merge(old, on=['age_ms', 'noise_quantum', 'method'], suffixes=('_new', '_parent'), validate='one_to_one')
    if len(joined) != len(ages)*len(noises)*len(METHODS):
        raise ValueError('Incomplete parent recombination coverage')
    delta = float((joined.mae_deg_new-joined.mae_deg_parent).abs().max())
    if delta > TOLERANCE:
        raise ValueError('Recombined horizon means differ from previous report')
    for method in METHODS:
        clean = recombined[(recombined.noise_quantum == 0) & (recombined.method == method)].set_index('age_ms')
        for age in ages:
            if abs(clean.loc[age, 'mae_deg']-previous_report['clean_old_curve'][method][str(age)]) > TOLERANCE:
                raise ValueError('Parent JSON clean curve differs')
    OUT.mkdir(parents=True)
    frame.to_csv(OUT/'axis_horizon_metrics.csv', index=False)
    records.to_csv(OUT/'record_horizon_metrics.csv', index=False)
    summary.to_csv(OUT/'horizon_summary.csv', index=False)
    paired.to_csv(OUT/'paired_horizon_summary.csv', index=False)
    paired_records.to_csv(OUT/'paired_record_horizons.csv', index=False)
    recombined.to_csv(OUT/'recombined_age_summary.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), layout='constrained')
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    for row, (comparator, label) in enumerate([('dense_matched_arx', 'VAL-selected dense ARX'),
                                              ('bank_simplex_1', 'Direct bank alone')]):
        column = 'selected_minus_'+comparator
        bound = float(paired[column].abs().max())
        for col, noise in enumerate(noises):
            ax = axes[row, col]
            matrix = paired[paired.noise_quantum == noise].pivot(index='age_ms', columns='horizon_s', values=column).reindex(index=ages, columns=HORIZONS)
            im = ax.imshow(matrix.to_numpy(), cmap='RdBu_r', vmin=-bound, vmax=bound, aspect='auto')
            for i, age in enumerate(ages):
                for j, horizon in enumerate(HORIZONS):
                    value = matrix.loc[age, horizon]
                    if value > TOLERANCE:
                        ax.add_patch(Rectangle((j-.48, i-.48), .96, .96, fill=False, edgecolor='#c31230', linewidth=1.6))
                    ax.text(j, i, f'{value:+.3f}' if abs(value) > 5e-4 else '0.000', ha='center', va='center', fontsize=9,
                            color='white' if abs(value) > .6*bound else 'black')
            ax.set_xticks(range(len(HORIZONS)), [f'{v:g}' for v in HORIZONS])
            ax.set_yticks(range(len(ages)), ages)
            ax.set(xlabel='Forecast horizon after current time (s)', ylabel='Known angle age (ms)',
                   title=f'{label} | '+('clean' if noise == 0 else 'synthetic jitter ±q'))
            fig.colorbar(im, ax=ax, shrink=.8, label='Selected MAE − comparator MAE (degrees)')
    fig.suptitle('Frozen models: horizon tradeoffs on seven previously opened records\nNegative = selected better; red outline = selected worse', fontsize=14)
    fig.savefig(OUT/'dense_horizons.png', dpi=170)
    plt.close(fig)
    counterexample = paired[(paired.age_ms == 250) & (paired.horizon_s == .1)][
        ['noise_quantum', 'selected', 'dense_matched_arx', 'selected_minus_dense_matched_arx',
         'wins_vs_dense_matched_arx', 'losses_vs_dense_matched_arx']]
    counts = paired.groupby('noise_quantum').agg(cells=('horizon_s', 'size'),
        better_than_dense=('selected_minus_dense_matched_arx', lambda x: int((x < -TOLERANCE).sum())),
        worse_than_dense=('selected_minus_dense_matched_arx', lambda x: int((x > TOLERANCE).sum())),
        better_than_bank=('selected_minus_bank_simplex_1', lambda x: int((x < -TOLERANCE).sum())),
        worse_than_bank=('selected_minus_bank_simplex_1', lambda x: int((x > TOLERANCE).sum())),
        ties_with_bank=('selected_minus_bank_simplex_1', lambda x: int((x.abs() <= TOLERANCE).sum()))).reset_index()
    counts.to_csv(OUT/'descriptive_cell_counts.csv', index=False)
    clean250 = paired[(paired.age_ms == 250) & (paired.noise_quantum == 0)][
        ['horizon_s', 'selected', 'dense_matched_arx', 'bank_simplex_1', 'selected_minus_bank_simplex_1']]
    DOC.write_text(f'''# Где средняя ошибка скрывает проигрыш: горизонты и возраст угла

9 ноября 2026. Это новая описательная разбивка **уже замороженных** метрик
`dense_reference_20261108`. Новых измерений, прогнозов, fit, выбора весов или
изменения gate нет. Диагностика предложена после просмотра прежних результатов,
поэтому не является предзарегистрированным подтверждением.

## Обнаруженный контрпример

Среднее по всем семи горизонтам скрывает проигрыш самого короткого прогноза.
При известном возрасте угла250мс и прогнозе на100мс после текущего момента:

{table(counterexample)}

В обоих уровнях шума primary хуже плотного ARX по этой общей MAE и проигрывает
в6из7 физических записей. Это совместимо с прежним выигрышем среднего по семи
горизонтам: у плотного ARX сильнее растёт ошибка на длинных горизонтах.
Результат не доказывает причину ошибки и не разрешает перевыбрать маршрутизацию
на age250мс по этим же записям. Прогноз100мс после текущего момента находится
на350мс позже последнего полученного угла; support-каналы имеют иной доступный
clock согласно прежнему контракту.

## Полная фиксированная сетка

Все9ages {ages}мс × 2шума {noises} × 7горизонтов {HORIZONS}с.
Сохранены все5методов: `selected`, `dense_matched_arx`, `matched_arx`,
`parent_quad`, `bank_simplex_1`. Ничего не отфильтровано по ошибке, амплитуде
или режиму. Сначала равные оси внутри физической записи на каждом горизонте,
затем равные7записей. Возраст, шум и горизонт здесь не усредняются между собой.
Это исходная equal-physical-scenario постановка, **не** quiet/drive-balanced score.

{table(counts)}

Эти числа — количество описательных ячеек, а не независимые испытания,
вероятности успеха или биномиальная значимость. Одни и те же7записей повторяются
по шуму, возрастам и горизонтам. Age0 для selected и прямого банка совпадает
по конструкции, поэтому7таких ties не являются новыми успешными исправлениями.

При250мс без добавленного шума:

{table(clean250)}

Смесь проигрывает прямому банку на части коротких и на последнем горизонте,
но выигрывает на .5 и1с. Средний score не означает равномерного улучшения.
Все остальные ages/noises, выигрыши и проигрыши остаются в CSV, включая
попарные результаты каждого сценария. График `dense_horizons.png` показывает
две полные карты сравнений; положительная разность обведена красным.

## Ограничения и назначение

Ошибка измеряется относительно прежнего encoder, а не независимого точного
угла. Абсолютная разность меньше одного encoder quantum сама по себе не
доказывает физически разрешимое улучшение; в низкоамплитудных фрагментах
относительные проценты могут вводить в заблуждение. Здесь используются градусы
и сохраняются все окна. Пиковая ошибка или риск пропуска опасного угла из этих
усреднённых CSV не выводятся. Для них потребовались бы исходные per-window
ошибки и отдельно зафиксированные диагностические определения.

Временной возраст и jitter синтетические, управление взято из прежней записи,
замкнутый контур не пересчитывается. Результат не является проверкой safety,
устойчивости, deadline, реального sensor delay или новой аппаратной победой.
Возможное назначение — включить отдельную проверку короткого горизонта в
будущий заранее замороженный протокол на новых данных. Текущий primary,
comparator bundle, их выборы и существующие prospective gates не изменены.

## Проверка расчёта

Проверен SHA256 архивного `frozen_replay/metrics.csv` против его завершённого
manifest, связь manifest с прежним verification и report. Проверены все
{len(frame)} axis×record×method×age×noise×horizon ячеек, отсутствие дублей,
совпадающее положительное число окон у методов и горизонтов, полный paired
набор. Сохранены {len(records)} record-horizon ячеек и{len(paired_records)}
попарных record/age/noise/horizon строк. Повторное усреднение семи горизонтов
совпало с прежним equal-record отчётом во всех{len(joined)}ячейках;
максимальная разность{delta:.3g}°. Отдельно сверена чистая кривая прежнего JSON.
Это пересчёт статистики сохранённых ошибок; код обучения/inference не запускался.

Полные файлы: `results/dense_horizons_20261109/`. Независимое подтверждение
по-прежнему отсутствует; исходная исследовательская цель не объявляется выполненной.
''')
    result = dict(passed=True, metrics_sha256=digest(metrics_path),
        frozen_provenance_sha256=digest(provenance_path), development_sha256=digest(development_path),
        parent_report_sha256=digest(PARENT/'report.json'), parent_summary_sha256=digest(PARENT/'summary.csv'),
        source_sha256=digest(Path(__file__)), document_sha256=digest(DOC),
        methods=METHODS, ages_ms=ages, noises=noises, horizons_s=HORIZONS,
        physical_scenarios=scenarios, axis_record_pairs=len(pairs), axis_cells=len(frame),
        record_horizon_cells=len(records), paired_record_horizon_rows=len(paired_records),
        recombined_cells=len(joined), recombination_max_abs_difference_deg=delta,
        paired_coverage_complete=True, common_window_counts_verified=True,
        cell_counts=counts.to_dict('records'), counterexample=counterexample.to_dict('records'),
        exploratory_posthoc_diagnostic=True, fitting_performed=False, inference_performed=False,
        primary_unchanged=True, selection_changed=False, independent_confirmation=False,
        closed_loop_control_evaluated=False, final_goal_achieved=False)
    write(OUT/'verification.json', result)
    write(OUT/'report.json', {**result, 'result_hashes': {
        str(p.relative_to(OUT)): digest(p) for p in OUT.rglob('*') if p.is_file()}})
    print(json.dumps({'passed': True, 'counterexample': result['counterexample'],
                      'recombined_cells': len(joined), 'max_difference_deg': delta}, indent=2))


if __name__ == '__main__':
    main()
