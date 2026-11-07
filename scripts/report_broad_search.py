"""Report both frozen screens, all failures and record-level uncertainty."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from cranepilot.metrics import paired_bootstrap


def main():
    out = Path('results/broad_search_20261107')
    if (out / 'report.json').exists(): raise FileExistsError('Preserve earlier report')
    out.mkdir(parents=True, exist_ok=True)
    runs = {name: Path('artifacts/runs') / f'{name}_20261107' for name in ['broad_search', 'reliability_blend']}
    metric_parts, confirmation_parts, selections, provenance = [], [], {}, {}
    for family, path in runs.items():
        p = json.loads((path / 'provenance.json').read_text())
        if p['status'] != 'complete': raise ValueError('Both completed screens required')
        for name, expected in p['result_hashes'].items():
            if sha256(path / name) != expected: raise ValueError(f'Changed result: {path / name}')
        provenance[family] = p
        selections[family] = json.loads((path / 'selection.json').read_text())
        d = pd.read_csv(path / 'metrics.csv'); d['family'] = family; metric_parts.append(d)
        c = pd.read_csv(path / 'confirmation_metrics.csv'); c['family'] = family; confirmation_parts.append(c)
    with np.load(runs['broad_search'] / 'confirmation_predictions.npz') as a, np.load(runs['reliability_blend'] / 'confirmation_predictions.npz') as b:
        np.testing.assert_array_equal(a['theta'], b['theta'])
        np.testing.assert_array_equal(a['target'], b['target'])
    d = pd.concat(metric_parts, ignore_index=True); c = pd.concat(confirmation_parts, ignore_index=True)
    evaluation = d[(d.group == 'evaluation') & d.extrapolation]
    summary = evaluation.groupby(['family', 'model', 'fold', 'regime']).mae_deg.mean().reset_index()
    base = summary[summary.model == 'incumbent'].drop(columns='model').rename(columns={'mae_deg': 'incumbent_mae_deg'})
    summary = summary.merge(base, on=['family', 'fold', 'regime'])
    summary['change_percent'] = (summary.mae_deg / summary.incumbent_mae_deg - 1) * 100
    summary.to_csv(out / 'summary.csv', index=False)
    evaluation.groupby(['family', 'model', 'fold', 'regime', 'horizon_s']).mae_deg.mean().to_csv(out / 'horizons.csv')
    d[d.group == 'load_stress'].groupby(['family', 'model', 'fold', 'regime']).mae_deg.mean().to_csv(out / 'load_stress.csv')
    confirmation = c.groupby(['family', 'model', 'measurement', 'damping_per_s']).absolute_error_deg.mean().reset_index()
    confirmation.to_csv(out / 'confirmation_summary.csv', index=False)
    c.groupby(['family', 'model']).absolute_error_deg.mean().to_csv(out / 'confirmation_all.csv')
    pairs, intervals = [], []
    for family in runs:
        part = evaluation[(evaluation.family == family) & (evaluation.regime == 'all')]
        for fold in ['primary', 'short_transfer', 'long_transfer']:
            table = part[part.fold == fold].groupby(['scenario', 'model']).mae_deg.mean().unstack('model')
            for model in ['selected_candidate', 'global_strategy']:
                difference = table[model] - table.incumbent
                for sid, value in difference.items():
                    pairs.append({'family': family, 'fold': fold, 'policy': model, 'scenario': sid, 'delta_mae_deg': value})
                intervals.append({'family': family, 'fold': fold, 'policy': model, **paired_bootstrap(difference.to_numpy(), seed=711)})
    pd.DataFrame(pairs).to_csv(out / 'paired_scenario_differences.csv', index=False)
    pd.DataFrame(intervals).to_csv(out / 'descriptive_bootstrap.csv', index=False)
    fits = pd.read_csv(runs['broad_search'] / 'fits.csv')
    fits.to_csv(out / 'fits.csv', index=False)
    fits.groupby(['fold', 'model']).agg(selected_initial=('selected_epoch', lambda x: int((x == -1).sum())),
        epochs=('epochs_executed', 'sum'), seconds=('fit_seconds', 'sum')).to_csv(out / 'training_budget.csv')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    variants = list(provenance['broad_search']['config']['variants'])
    gates = [f'gate_{x:g}' for x in provenance['reliability_blend']['config']['tau_bins']]
    rows = [('broad_search', name) for name in variants] + [('reliability_blend', name) for name in gates]
    columns = [(fold, regime) for fold in ['primary', 'short_transfer', 'long_transfer'] for regime in ['all', 'active']]
    matrix = np.array([[float(summary[(summary.family == family) & (summary.model == name) & (summary.fold == fold) & (summary.regime == regime)].change_percent.iloc[0]) for fold, regime in columns] for family, name in rows])
    fig, ax = plt.subplots(figsize=(11.6, 9.0))
    im = ax.imshow(matrix, cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10), aspect='auto')
    for i, (family, name) in enumerate(rows):
        for j, (fold, regime) in enumerate(columns):
            star = '*' if selections[family]['folds'][fold]['model'] == name else ''
            color = 'white' if abs(matrix[i, j]) >= 7 else '#182d3b'
            ax.text(j, i, f'{matrix[i,j]:+.2f}%{star}', ha='center', va='center', color=color, fontsize=10)
    ax.set_yticks(range(len(rows)), [name for _, name in rows])
    ax.set_xticks(range(6), ['Основное\nвсё', 'Основное\nактивное', 'Короткие\nвсё', 'Короткие\nактивное', 'Длинные\nвсё', 'Длинные\nактивное'])
    ax.axhline(8.5, color='white', lw=3)
    fig.colorbar(im, ax=ax, shrink=.7, label='Изменение MAE относительно incumbent, %\nцвет ограничен диапазоном ±10%')
    ax.set_title('81 обучение и шесть фиксированных смесей: все результаты переноса', pad=18, fontsize=13)
    fig.text(.02, .02, 'Отрицательное значение — улучшение. * Выбрано по валидации до теста. Записи уже использовались в исследовании.', fontsize=9)
    fig.tight_layout(rect=[0, .05, 1, 1]); fig.savefig(out / 'strategy_matrix.png', dpi=180); fig.savefig(out / 'strategy_matrix.svg'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    primary = evaluation[(evaluation.family == 'broad_search') & (evaluation.fold == 'primary') & (evaluation.regime == 'all')]
    for name, label, color in [('incumbent', 'Исходная модель', '#506879'), ('soft_symmetry', 'Частичная симметрия', '#c8793e')]:
        series = primary[primary.model == name].groupby('horizon_s').mae_deg.mean()
        axes[0].plot(series.index, series, marker='o', color=color, label=label)
    axes[0].set_title('Основной протокол: все горизонты'); axes[0].set_xlabel('Горизонт, с'); axes[0].set_ylabel('MAE угла, °'); axes[0].legend(frameon=False)
    labels = ['Исходная\nмодель', 'Частичная\nсимметрия', 'Смесь\ngate_0.5', 'Физический\nпрогноз']
    for i, (mode, color) in enumerate([('exact', '#347d78'), ('quantized', '#bdb7a7')]):
        vals = []
        for family, name in [('broad_search', 'incumbent'), ('broad_search', 'soft_symmetry'), ('reliability_blend', 'gate_0.5'), ('reliability_blend', 'physical')]:
            vals.append(float(confirmation[(confirmation.family == family) & (confirmation.model == name) & (confirmation.measurement == mode) & (confirmation.damping_per_s == 0)].absolute_error_deg.iloc[0]))
        x = np.arange(4) + (i - .5) * .36
        axes[1].bar(x, vals, width=.34, color=color, label='Точная история' if mode == 'exact' else 'Квантованная история')
        for xx, value in zip(x, vals): axes[1].text(xx, value + .005, f'{value:.3f}', ha='center', fontsize=9)
    axes[1].set_xticks(range(4), labels); axes[1].set_ylim(0, .31); axes[1].set_ylabel('MAE угла, °')
    axes[1].set_title('Новая сетка: свободное движение без затухания'); axes[1].legend(frameon=False, fontsize=9)
    for ax in axes: ax.grid(axis='y', color='#e5e9eb'); ax.set_axisbelow(True)
    fig.suptitle('Два разных результата: точность основного прогноза и физический предел', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(out / 'main_findings.png', dpi=180); fig.savefig(out / 'main_findings.svg'); plt.close(fig)
    info = {'date': '2026-11-07', 'training_runs': len(fits), 'initial_checkpoints_selected': int((fits.selected_epoch == -1).sum()),
            'epochs_executed': int(fits.epochs_executed.sum()), 'sum_fit_seconds': float(fits.fit_seconds.sum()),
            'selection': selections, 'source_provenance_sha256': {k: sha256(p / 'provenance.json') for k, p in runs.items()},
            'descriptive_intervals': intervals, 'new_hardware_evidence': False, 'source_data_reused': True,
            'model_default_changed': False, 'confirmation_cases_per_family': 432, 'confirmation_targets_identical': True}
    (out / 'report.json').write_text(json.dumps(info, indent=2))
    print(summary[summary.model.isin(['incumbent', 'selected_candidate', 'global_strategy'])].to_string(index=False))
    print('\nBootstrap\n', pd.DataFrame(intervals).to_string(index=False))


if __name__ == '__main__': main()
