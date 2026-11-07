"""Report a completed fixed-composition stress run, preserving every comparison."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from cranepilot.data import sha256


METHODS = ['incumbent', 'soft_symmetry', 'incumbent_gate', 'combined',
           'fixed_half', 'physical', 'adaptive_physics']


def comparisons(frame, keys, references):
    result = frame.copy()
    for name in references:
        reference = frame[frame.model == name][keys + ['mae_deg']].rename(
            columns={'mae_deg': f'{name}_mae_deg'})
        result = result.merge(reference, on=keys, how='left', validate='many_to_one')
        result[f'change_vs_{name}_percent'] = 100 * (
            result.mae_deg / result[f'{name}_mae_deg'] - 1)
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def figures(out, real, simulation, cfg):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.6),
                             gridspec_kw={'width_ratios': [1, 1.65]})
    columns = [(fold, regime) for fold in cfg['folds'] for regime in ['all', 'active']]
    base_y = np.arange(len(columns))
    for offset, name, label, color in [(-.24, 'soft_symmetry', 'Частичная симметрия', '#c8793e'),
                                      (0., 'incumbent_gate', 'Исходная + gate', '#577b97'),
                                      (.24, 'combined', 'Симметрия + gate', '#297b70')]:
        values = [float(real[(real.fold == fold) & (real.regime == regime)
                             & (real.model == name)].change_vs_incumbent_percent.iloc[0])
                  for fold, regime in columns]
        axes[0].barh(base_y + offset, values, height=.21, color=color, label=label)
        for y, value in zip(base_y + offset, values):
            axes[0].annotate(f'{value:+.2f}%', (value, y),
                             xytext=(3 if value >= 0 else -3, 0), textcoords='offset points',
                             ha='left' if value >= 0 else 'right', va='center', fontsize=8)
    fold_names = {'primary': 'Основное', 'short_transfer': 'Короткие', 'long_transfer': 'Длинные'}
    axes[0].set_yticks(base_y, [fold_names.get(fold, fold) + (' — всё' if regime == 'all' else ' — активное')
                               for fold, regime in columns])
    axes[0].invert_yaxis()
    axes[0].axvline(0, color='#63717b', lw=.8)
    axes[0].grid(axis='x', color='#e5e9eb')
    axes[0].set_axisbelow(True)
    axes[0].margins(x=.33)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='lower left',
               bbox_to_anchor=(.11, .045), ncol=3, fontsize=9)
    axes[0].set_title('Реальные записи: ретроспективная оценка', pad=15)
    axes[0].set_xlabel('Изменение MAE к исходной модели, %\nОтрицательное значение — улучшение')

    primary = simulation[simulation.fold == 'primary']
    groups = [(c['name'], mode) for c in cfg['simulation']['conditions']
              for mode in cfg['simulation']['measurement']]
    table = primary.pivot(index=['condition', 'measurement'], columns='model', values='mae_deg')
    matrix = table.reindex(index=pd.MultiIndex.from_tuples(groups), columns=METHODS).to_numpy()
    if not np.isfinite(matrix).all():
        raise ValueError('Complete seven-method primary simulation matrix required')
    positive = matrix[matrix > 0]
    vmin, vmax = max(float(positive.min()), 1e-8), max(float(matrix.max()), 1e-7)
    norm = LogNorm(vmin=vmin, vmax=max(vmin * 1.01, vmax))
    im = axes[1].imshow(np.maximum(matrix, vmin), cmap='magma_r', norm=norm, aspect='auto')
    for i in range(len(groups)):
        for j in range(len(METHODS)):
            color = 'white' if norm(max(matrix[i, j], vmin)) > .58 else '#202b33'
            axes[1].text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center', fontsize=8, color=color)
    condition_labels = {'nominal': 'Номинал', 'slow': 'ω × 0,85', 'fast': 'ω × 1,15',
                        'damped': 'Затухание', 'hidden_impulse': 'Скрытый импульс'}
    mode_labels = {'exact': 'точно', 'quantized': 'квант.', 'noisy_quantized': 'шум',
                   'biased_quantized': 'смещение'}
    axes[1].set_yticks(range(len(groups)), [f'{condition_labels.get(c,c)} / {mode_labels.get(m,m)}' for c,m in groups])
    axes[1].set_xticks(range(7), ['Исходная', 'Симметрия', 'Исходная\n+ gate', 'Комбинация',
                                 'Смесь ½', 'Физика', 'Адапт.\nфизика'], fontsize=8)
    for i in range(4, len(groups), 4):
        axes[1].axhline(i - .5, color='white', lw=1.5)
    axes[1].set_title('Новая симуляция: основной набор весов', pad=15)
    fig.colorbar(im, ax=axes[1], fraction=.035, pad=.02,
                 label='MAE угла, °; логарифмическая шкала цвета')
    fig.suptitle('Фиксированная комбинация: точность и устойчивость к несоответствию модели', fontsize=14)
    fig.text(.02, .015, 'Симуляция не является аппаратным испытанием. Скрытый будущий импульс не наблюдается ни одним методом.', fontsize=9)
    fig.tight_layout(rect=[0, .12, 1, .96], w_pad=3)
    fig.savefig(out / 'combined_findings.png', dpi=170)
    fig.savefig(out / 'combined_findings.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run, args.output
    prov = json.loads((run / 'provenance.json').read_text())
    if prov['status'] != 'complete':
        raise ValueError('A completed run is required')
    for relative, expected in prov['result_hashes'].items():
        if sha256(run / relative) != expected:
            raise ValueError(f'Changed result: {relative}')
    if (out / 'report.json').exists():
        raise FileExistsError('Preserve the completed report; choose a new output directory')
    cfg = prov['config']
    real = pd.read_csv(run / 'real_metrics.csv')
    sim = pd.read_csv(run / 'simulation_metrics.csv')
    if real.duplicated(['fold', 'group', 'scenario', 'model', 'regime', 'horizon_s']).any():
        raise ValueError('Duplicate real metric rows')
    real_horizon_counts = real.groupby(['fold', 'group', 'scenario', 'model', 'regime']).horizon_s.nunique()
    if not real_horizon_counts.eq(7).all():
        raise ValueError('Every real scenario/regime must contain all seven horizons')
    # Average the seven horizons within each complete scenario first, then
    # average scenarios equally. Overlapping windows are not sample weights.
    evaluation = real[(real.group == 'evaluation') & real.extrapolation.astype(bool)]
    records = evaluation.groupby(['fold', 'scenario', 'model', 'regime'], as_index=False).mae_deg.mean()
    summary = records.groupby(['fold', 'model', 'regime'], as_index=False).agg(
        mae_deg=('mae_deg', 'mean'), scenarios=('scenario', 'nunique'))
    summary = comparisons(summary, ['fold', 'regime'], ['incumbent', 'soft_symmetry', 'incumbent_gate', 'physical'])
    horizons = evaluation.groupby(['fold', 'model', 'regime', 'horizon_s'], as_index=False).mae_deg.mean()
    stress_records = real[real.group == 'load_stress'].groupby(
        ['fold', 'scenario', 'model', 'regime'], as_index=False).mae_deg.mean()
    load_stress = stress_records.groupby(['fold', 'model', 'regime'], as_index=False).mae_deg.mean()
    load_stress = comparisons(load_stress, ['fold', 'regime'], ['incumbent', 'soft_symmetry'])

    case_keys = ['fold', 'condition', 'measurement', 'length_m', 'trajectory_id', 'model']
    if sim.duplicated(case_keys + ['horizon_s']).any():
        raise ValueError('Duplicate simulation metric rows')
    counts = sim.groupby(case_keys).horizon_s.nunique()
    if not counts.eq(7).all():
        raise ValueError('Every simulated case must contain all seven horizons')
    cases = sim.groupby(case_keys, as_index=False).absolute_error_deg.mean().rename(columns={'absolute_error_deg': 'mae_deg'})
    group_keys = ['fold', 'condition', 'measurement', 'model']
    stress = cases.groupby(group_keys, as_index=False).agg(
        mae_deg=('mae_deg', 'mean'), case_p95_mae_deg=('mae_deg', lambda x: float(x.quantile(.95))),
        worst_case_mae_deg=('mae_deg', 'max'), trajectories=('trajectory_id', 'nunique'))
    stress = comparisons(stress, ['fold', 'condition', 'measurement'],
                         ['incumbent', 'soft_symmetry', 'incumbent_gate', 'physical', 'adaptive_physics'])
    lengths = cases.groupby(group_keys + ['length_m'], as_index=False).agg(
        mae_deg=('mae_deg', 'mean'), case_p95_mae_deg=('mae_deg', lambda x: float(x.quantile(.95))))
    sim_horizons = sim.groupby(group_keys + ['horizon_s'], as_index=False).absolute_error_deg.mean()

    # The configuration threshold is operational; these are not confidence bounds.
    real_gates, simulation_gates = [], []
    for fold in cfg['folds']:
        limit = cfg['acceptance']['primary_all_active_max_ratio' if fold == 'primary' else 'transfer_all_active_max_ratio']
        for regime in ['all', 'active']:
            row = summary[(summary.fold == fold) & (summary.regime == regime) & (summary.model == 'combined')]
            if len(row) != 1:
                raise ValueError('One combined real metric per acceptance cell required')
            value = row.iloc[0]
            ratio = float(value.mae_deg / value.incumbent_mae_deg)
            real_gates.append({'fold': fold, 'regime': regime, 'ratio_to_incumbent': ratio,
                               'maximum_ratio': limit, 'passed': bool(ratio <= limit)})
        local = stress[(stress.fold == fold) & (stress.model == 'combined')]
        expected_groups = {(c['name'], m) for c in cfg['simulation']['conditions'] for m in cfg['simulation']['measurement']}
        actual_groups = set(zip(local.condition, local.measurement))
        if len(local) != 20 or len(expected_groups) != 20 or actual_groups != expected_groups:
            raise ValueError('Exactly the 20 predefined simulation groups per fold are required')
        limit = cfg['acceptance']['each_simulation_group_max_ratio']
        for row in local.itertuples():
            ratio = float(row.mae_deg / row.incumbent_mae_deg)
            simulation_gates.append({'fold': fold, 'condition': row.condition, 'measurement': row.measurement,
                                     'ratio_to_incumbent': ratio, 'maximum_ratio': limit, 'passed': bool(ratio <= limit)})
    acceptance = {'real': real_gates, 'simulation': simulation_gates,
                  'real_passed': all(x['passed'] for x in real_gates),
                  'simulation_passed': all(x['passed'] for x in simulation_gates),
                  'simulation_groups_per_fold': {fold: sum(x['fold'] == fold for x in simulation_gates) for fold in cfg['folds']},
                  'statistical_noninferiority_claim': False}
    acceptance['overall_passed'] = acceptance['real_passed'] and acceptance['simulation_passed']
    out.mkdir(parents=True, exist_ok=True)
    real.to_csv(out / 'real_metrics_all.csv', index=False)
    records.to_csv(out / 'real_scenario_means.csv', index=False)
    summary.to_csv(out / 'real_summary.csv', index=False)
    horizons.to_csv(out / 'real_horizons.csv', index=False)
    load_stress.to_csv(out / 'load_stress.csv', index=False)
    cases.to_csv(out / 'simulation_case_means.csv', index=False)
    stress.to_csv(out / 'simulation_groups.csv', index=False)
    lengths.to_csv(out / 'simulation_lengths.csv', index=False)
    sim_horizons.to_csv(out / 'simulation_horizons.csv', index=False)
    stress.sort_values('change_vs_incumbent_percent', ascending=False).to_csv(out / 'simulation_worst_groups.csv', index=False)
    pd.concat([summary.assign(domain='retrospective_real'), stress.assign(domain='simulation')],
              ignore_index=True, sort=False).to_csv(out / 'results.csv', index=False)
    for name in ['gate_statistics.csv', 'verification.json']:
        shutil.copy2(run / name, out / name)
    write_json(out / 'acceptance.json', acceptance)
    figures(out, summary, stress, cfg)
    write_json(out / 'report.json', {
        'source_run': str(run), 'source_provenance_sha256': sha256(run / 'provenance.json'),
        'report_source_sha256': sha256(Path(__file__)), 'source_results_verified': len(prov['result_hashes']),
        'acceptance_overall_passed': acceptance['overall_passed'], 'real_evaluation_retrospective': True,
        'new_hardware_evidence': False, 'training_performed': False, 'selection_performed': False,
        'model_default_changed': False, 'case_p95_definition': '95th percentile across trajectory seven-horizon MAEs within each condition/measurement/fold/method',
        'real_weighting': 'Equal scenarios after equal seven horizons; evaluation extrapolation only',
        'simulation_fold_targets_identical': False,
        'fold_simulation_case_counts': {fold: int(len(cases[(cases.fold == fold) & (cases.model == 'incumbent')])) for fold in cfg['folds']},
        'all_negative_results_retained': True})
    print(summary[(summary.model == 'combined') & summary.regime.isin(['all', 'active'])].to_string(index=False))
    print('Acceptance:', json.dumps({key: acceptance[key] for key in ['real_passed', 'simulation_passed', 'overall_passed']}))


if __name__ == '__main__':
    main()
