"""Generate all-variant tables, diagnostics and a compact research figure."""
from pathlib import Path
import json
import re
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


LABELS = {'forced_driven_huber': 'Incumbent', 'dense_horizon': 'Dense / 7 horizons',
          'temporal_uniform': 'Temporal / all steps', 'temporal_horizon': 'Temporal / 7 horizons',
          'tcn_reference': 'TCN reference', 'forced_physics': 'Physics',
          'selected_candidate': 'Validation-selected'}


def markdown(frame, digits=5):
    columns = list(frame.columns)
    rows = ['| '+' | '.join(map(str, columns))+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for row in frame.itertuples(index=False, name=None):
        values = [f'{value:.{digits}f}' if isinstance(value, (float, np.floating)) else str(value) for value in row]
        rows.append('| '+' | '.join(values)+' |')
    return '\n'.join(rows)


def main():
    run = Path('artifacts/runs/temporal_20261106')
    output = Path('results/latest')
    p = json.loads((run/'provenance.json').read_text())
    if p['status'] != 'complete':
        raise ValueError('Complete evaluation first')
    selections = json.loads((run/'selection.json').read_text())
    metrics = pd.read_csv(run/'metrics.csv')
    extrap = metrics.query("group == 'evaluation' and regime == 'all' and extrapolation")
    summary = extrap.groupby(['fold', 'model']).mae_deg.mean().unstack('fold')
    horizons = extrap.query("fold == 'primary'").groupby(['horizon_s', 'model']).mae_deg.mean().unstack()
    active = metrics.query("group == 'evaluation' and regime == 'active_at_origin' and extrapolation")
    active = active.groupby(['fold', 'model']).mae_deg.mean().unstack('fold')
    stress = metrics.query("group == 'load_stress' and regime == 'all'").groupby('model').mae_deg.mean()
    clean = pd.read_csv(run/'cleanroom/metrics.csv')
    clean_summary = clean.groupby(['model', 'measurement']).absolute_error_deg.mean().unstack()
    clean_damping = clean.query("measurement == 'exact'").groupby(['model', 'damping_per_s']).absolute_error_deg.mean().unstack()
    seeds = pd.read_csv(run/'validation_seeds.csv')
    logs = pd.read_csv(run/'training.csv')
    checks = (output/'scientific_checks_20261106.txt').read_text(encoding='utf-8')
    if not checks.rstrip().endswith('OK'):
        raise ValueError('Scientific test suite must pass before reporting')
    test_count = int(re.search(r'Ran (\d+) tests', checks).group(1))
    gains = {fold: 100*(1-summary.loc['selected_candidate', fold]/summary.loc['forced_driven_huber', fold])
             for fold in summary.columns}
    # Paired differences are summarized at scenario level, never as independent windows.
    records = extrap.groupby(['fold', 'scenario', 'model']).mae_deg.mean().unstack('model')
    differences = records['selected_candidate']-records['forced_driven_huber']
    rng = np.random.default_rng(6102026)
    intervals = {}
    for fold in summary.columns:
        d = differences.loc[fold].to_numpy()
        boot = rng.choice(d, size=(10000, len(d)), replace=True).mean(1)
        intervals[fold] = {'per_record_deg': d.tolist(),
                           'descriptive_95pct_deg': np.percentile(boot, [2.5, 97.5]).tolist()}
    report = {'date': '2026-11-06', 'new_angular_fits': int(len(seeds)), 'new_actuator_fits': 0,
        'tests_passed': test_count,
        'selected_by_validation': selections, 'extrapolation_macro_mae_deg': summary.to_dict(),
        'selected_gain_percent': gains, 'primary_horizon_mae_deg': horizons.to_dict(),
        'active_origin_macro_mae_deg': active.to_dict(), 'load_stress_macro_mae_deg': stress.to_dict(),
        'cleanroom_mean_mae_deg': clean_summary.to_dict(), 'record_level_differences': intervals,
        'cleanroom_exact_by_damping_mae_deg': clean_damping.to_dict(),
        'fit_seconds': float(logs.groupby(['fold', 'model', 'seed']).elapsed_s.max().sum()),
        'scope': 'Retrospective real-data comparisons and frozen synthetic angular-only falsification; no SOTA or hardware claim'}
    (output/'research_20261106_report.json').write_text(json.dumps(report, indent=2))
    summary.to_csv(output/'research_20261106_summary.csv')
    horizons.to_csv(output/'research_20261106_horizons.csv')
    clean_summary.to_csv(output/'research_20261106_cleanroom_summary.csv')
    for name in ('metrics.csv', 'training.csv', 'validation_seeds.csv', 'selection.json', 'provenance.json'):
        shutil.copy2(run/name, output/f'temporal_20261106_{name}')
    for name in ('metrics.csv', 'cases.csv', 'provenance.json'):
        shutil.copy2(run/'cleanroom'/name, output/f'cleanroom_20261106_{name}')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7), constrained_layout=True)
    colors = dict(zip(LABELS, ['#455A64', '#1976D2', '#E69F00', '#009E73', '#9C27B0', '#999999', '#D55E00']))
    for name in ['forced_driven_huber', 'dense_horizon', 'temporal_uniform', 'temporal_horizon', 'tcn_reference']:
        axes[0].plot(horizons.index, horizons[name], marker='o', ms=3, lw=1.6, color=colors[name], label=LABELS[name])
    axes[0].set(xlabel='Forecast horizon (s)', ylabel='Scenario-mean MAE (degrees)',
                title='Recorded lengths: primary split')
    axes[0].legend(fontsize=7, loc='upper left')
    show = ['forced_driven_huber', 'selected_candidate', 'tcn_reference', 'forced_physics']
    x = np.arange(3)
    for i, name in enumerate(show):
        values = summary.loc[name, ['primary', 'short_transfer', 'long_transfer']]
        axes[1].bar(x+(i-1.5)*.2, values, width=.19, color=colors[name], label=LABELS[name])
    axes[1].set(xticks=x, xticklabels=['Primary', 'Shorter', 'Longer'],
                ylabel='Macro MAE (degrees)', title='Extrapolation only; repeated records')
    axes[1].legend(fontsize=7)
    names = ['forced_physics', 'forced_driven_huber', 'dense_horizon', 'temporal_uniform', 'temporal_horizon', 'tcn_reference']
    ypos = np.arange(len(names))
    for i, measurement in enumerate(['exact', 'quantized']):
        axes[2].barh(ypos+(i-.5)*.34, clean_summary.loc[names, measurement], height=.32,
                     color=['#455A64', '#E69F00'][i], label=measurement)
    axes[2].set(yticks=ypos, yticklabels=[LABELS[n] for n in names],
                xlabel='Mean MAE (degrees)', title='Synthetic free sway; actor bypassed')
    axes[2].invert_yaxis()
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.2, axis='y' if ax is not axes[2] else 'x')
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig('paper/figures/research_20261106.png', dpi=180)
    fig.savefig('paper/figures/research_20261106.svg')
    plt.close(fig)
    selection_table = pd.DataFrame([{'Fold': fold, 'Selected model': value['model'],
        'Validation MAE': value['ensemble_validation_mae_deg'][value['model']]} for fold, value in selections.items()])
    text = f'''# Research checkpoint, 6 November 2026

Local only. This stage completed {len(seeds)} new angular fits: four variants,
three fixed seeds, three previously used splits. It reused each split's verified
actuator and train-only length calibration; no actuator was retrained. The
configuration, synthetic check and source were committed before training in
`678b754`. Validation selected checkpoints and the candidate; all variants are
reported, including failures to improve.

## Locked validation choices

{markdown(selection_table)}

Selection is among the three new physics candidates and the retained incumbent;
the TCN reference is reported separately and is not in this pool. A selected incumbent means the new
candidate did not justify replacement on that validation record. Selection
does not guarantee better held-out performance.

## Recorded-data results

Mean scenario/horizon MAE in degrees; lower is better. Each additional fold's
interpolation record is excluded from this table and retained in the full CSV.

{markdown(summary.reset_index())}

Gain of the validation-selected candidate relative to the incumbent, percent:
{json.dumps(gains, sort_keys=True)}. These values are descriptive and signed;
a negative value is a regression. Different folds may select different models.

Primary horizon details:

{markdown(horizons.reset_index())}

Active-at-origin records, same seven-horizon macro average:

{markdown(active.reset_index())}

Primary alternate-load/shape stress group (10/11/12), kept separate:

{markdown(stress.reset_index())}

## Synthetic falsification

576 predetermined cases from 288 nonlinear free-sway trajectories, each with
exact and quantized past angles. All angular models receive zero future
acceleration. The learned actuator is deliberately bypassed. Effective length
matches the incumbent's train-only calibration. This is an interpretable
matched-physics limit check, not another real-world performance score.

{markdown(clean_summary.reset_index())}

Exact measurements, separated by damping coefficient (1/s):

{markdown(clean_damping.reset_index())}

The undamped limit is especially informative: the incumbent averages about
0.23573 deg versus 0.01213 deg for the physical forecast. The temporal/uniform
model reduces this to 0.04522 deg, but loses the main recorded-data comparison.
Better behavior on this simple synthetic limit therefore does not by itself
justify replacing the current model. Physics-preserving residual constraints
and more reliable validation across regimes are concrete next hypotheses.

No weights, architecture, blend or selection were tuned against these cases.
The case-by-case errors and RK4 step-halving audit are retained. Large errors
here indicate that learned dataset corrections need not vanish in free sway.

## Interpretation limits

The real records have already informed previous development. The experiment
remains retrospective, despite locking all settings before this stage's
evaluation. Overlapping windows, neural seeds and repeated folds are not
independent physical trials. The scenario-level paired differences and
descriptive bootstrap intervals are in research_20261106_report.json; they
cannot repair this small, reused dataset or certify population superiority.

Temporal convolutions follow the established TCN idea (Bai, Kolter and Koltun,
2018, https://arxiv.org/abs/1803.01271). This task adaptation uses channel
dropout and no weight normalization; it is not an exact reproduction of the
paper's benchmarks. The two physics variants retain the same passive core
and extra learned forcing port. No new passivity or closed-loop guarantee is
claimed. ReSMILE was inspected but not reproduced or included as a beaten
baseline; see TEMPORAL_DESIGN.md.

## Reproduction and inference

{test_count} numerical, information-contract and causal-alignment tests passed.
The new Aalto event tests cover future-value changes, source/server availability
and missing history; they do not imply model accuracy on the new dataset.

Run from the project root with the project environment:

```sh
python -m scripts.run_temporal --output artifacts/runs/NEW_RUN
python -m scripts.run_temporal --output artifacts/runs/NEW_RUN --evaluate
python -m scripts.run_cleanroom --run artifacts/runs/NEW_RUN
python -m scripts.predict_temporal --input examples/history_packet.npz --output examples/NEW_FORECAST.csv
```

Training refuses to overwrite retained runs. Evaluation verifies code, data,
source checkpoints and locked selection hashes. The inference command uses
history-only inputs and verifies its checkpoint hashes. It uses the selected
primary variant by default; choose `--variant` and `--fold` explicitly for
another retained model. Seed spread is descriptive, not a confidence interval.

## Separate industrial data acquisition

Downloaded and verified Aalto's Ilmatar dataset, DOI 10.5281/zenodo.6907569
(CC BY 4.0): eight cycles, 31,303 exported rows, 0/120/500/1000 kg payloads.
It has a 250 ms exported grid, unlike the current 10 ms model contract.
An audit found many exported angles matching later raw events rather than the
last available event at the printed grid time. A separate importer now builds
ten causal signals from GOOD finite raw events using both source/server times,
retaining missing initial values and log ages. No model was trained or scored
on these data. Geometry, sensing and future-command semantics remain unresolved.
See AALTO_DATA_AUDIT.md and the retained per-cycle checks and examples.

All artifacts remain local; GitHub and Overleaf were not updated in this stage.
'''
    Path('docs/RESEARCH_2026_11_06.md').write_text(text, encoding='utf-8')
    print(summary.to_string())
    print(clean_summary.to_string())


if __name__ == '__main__':
    main()
