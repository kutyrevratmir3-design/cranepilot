"""Summarize the completed frozen-weight audit without selecting a new model."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cranepilot.data import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path('artifacts/runs/limit_audit_20261107_float64'))
    p.add_argument('--output', type=Path, default=Path('results/limit_audit_20261107'))
    a = p.parse_args()
    provenance = json.loads((a.run / 'provenance.json').read_text())
    if provenance['status'] != 'complete':
        raise ValueError('Completed audit required')
    for name, digest in provenance['output_sha256'].items():
        if sha256(a.run / name) != digest:
            raise ValueError(f'Changed audit output: {name}')
    if a.output.exists():
        raise FileExistsError('Preserve previous report')
    a.output.mkdir(parents=True)
    raw = pd.read_csv(a.run / 'metrics.csv')
    ensemble = raw[raw.seed == 'ensemble']
    columns = {}
    for name, group, subset in [
        ('free_undamped_exact_deg', 'synthetic_development', 'exact_damping_0'),
        ('free_undamped_quantized_deg', 'synthetic_development', 'quantized_damping_0'),
        ('synthetic_all_deg', 'synthetic_development', 'all'),
        ('validation_all_deg', 'validation_s4', 'all'),
        ('validation_active_deg', 'validation_s4', 'active'),
        ('validation_quiet_deg', 'validation_s4', 'quiet')]:
        part = ensemble[(ensemble.group == group) & (ensemble.subset == subset)]
        columns[name] = part.groupby('variant').mae_deg.mean()
    train = ensemble[ensemble.group.str.startswith('train_')]
    for subset in ['all', 'active', 'quiet']:
        columns[f'train_{subset}_deg'] = train[train.subset == subset].groupby(['variant', 'group']).mae_deg.mean().groupby('variant').mean()
    summary = pd.DataFrame(columns).loc[list(provenance['config']['variants'])]
    summary.index.name = 'variant'
    summary.to_csv(a.output / 'summary.csv')
    changes = (summary / summary.loc['incumbent'] - 1) * 100
    changes.to_csv(a.output / 'relative_error_change_percent.csv')
    raw.groupby(['group', 'subset', 'variant', 'seed']).mae_deg.mean().to_csv(a.output / 'per_seed_summary.csv')
    per_case = pd.read_csv(a.run / 'synthetic_case_metrics.csv')
    per_case.groupby(['variant', 'measurement', 'damping_per_s', 'length_m']).mae_deg.mean().to_csv(a.output / 'synthetic_length_breakdown.csv')
    counts = ensemble[ensemble.variant == 'incumbent'].groupby(['group', 'subset']).n.first()
    counts.to_csv(a.output / 'case_counts.csv')
    labels = ['Исходная модель', 'Без сдвига угла', 'Без поправки скорости',
              'Без поправки частоты', 'Без начальной памяти', 'Без пассивной динамики',
              'Без нейронного воздействия', 'Только поправки состояния',
              'Только обученная динамика', 'Физический прогноз']
    colors = ['#31546a', '#cf7140'] + ['#aab5bd'] * 7 + ['#528676']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    for ax, column, title in zip(axes,
        ['free_undamped_exact_deg', 'validation_all_deg', 'validation_active_deg'],
        ['Свободное движение\nбез затухания, точная история',
         'Запись 4: вся валидация', 'Запись 4: активные участки']):
        values = summary[column].to_numpy()
        ax.barh(np.arange(len(values)), values, color=colors, height=.66)
        ax.axvline(values[0], color='#31546a', lw=.9, ls='--', alpha=.6)
        ax.set_title(title, fontsize=11, pad=13)
        ax.set_xlabel('MAE угла, ° — меньше лучше')
        ax.set_xlim(0, values.max() * 1.22)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#e6e9eb', lw=.7)
        for i, v in enumerate(values):
            ax.text(v + values.max() * .02, i, f'{v:.4f}', va='center', fontsize=9)
    axes[0].set_yticks(np.arange(len(labels)), labels)
    axes[0].invert_yaxis()
    fig.suptitle('CranePilot: что дают отдельные обученные поправки', fontsize=16, x=.02, ha='left')
    fig.text(.02, .02, 'Сохранённые веса · 3 seed · без обучения · только development/validation · действующая модель не заменена', fontsize=10, color='#52616b')
    fig.tight_layout(rect=[0, .055, 1, .94])
    fig.savefig(a.output / 'limit_audit.png', dpi=180, facecolor='white')
    fig.savefig(a.output / 'limit_audit.svg', facecolor='white')
    plt.close(fig)
    info = {'source_run': str(a.run), 'source_provenance_sha256': sha256(a.run / 'provenance.json'),
            'aggregation': 'Equal horizons, then equal scenarios for train; three-seed ensemble predictions averaged before scoring.',
            'model_replaced': False, 'test_records_scored': [],
            'summary': summary.to_dict('index'), 'relative_change_percent': changes.to_dict('index')}
    (a.output / 'report.json').write_text(json.dumps(info, indent=2))
    print(summary.to_string())
    print('\nCounts\n', counts.to_string())


if __name__ == '__main__':
    main()
