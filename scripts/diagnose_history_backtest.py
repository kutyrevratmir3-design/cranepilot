"""Post-hoc matched-window diagnosis of frozen historical backtest choices.

No fitting, parameter selection, or target-dependent changes are performed.
History scores and predictions are read from the complete retained run.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from cranepilot.rollout import HORIZON_INDEX

MODELS = ['bt21', 'bt31', 'bt_multi', 'bt_guarded']


def number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def flags(series):
    values = series.astype(str).str.lower()
    if not values.isin(['true', 'false']).all():
        raise ValueError('Expected complete boolean diagnostics')
    return values.eq('true').to_numpy()


def run_diagnostic(run, out):
    if out.exists(): raise FileExistsError('Preserve earlier diagnostic')
    provenance = json.loads((run/'provenance.json').read_text())
    if provenance['status'] != 'complete': raise ValueError('Completed experiment required')
    for mapping, base in [('guarded_hashes', Path('.')), ('result_hashes', run)]:
        for name, digest in provenance[mapping].items():
            if sha256(base/name.replace('\\', '/')) != digest:
                raise ValueError(f'Changed retained dependency or result: {name}')
    estimates = pd.concat([pd.read_csv(run/f'{stage}_parameter_estimates.csv') for stage in ['select', 'evaluate']], ignore_index=True)
    estimates = estimates[estimates.model.isin(MODELS)]
    if estimates.duplicated(['fold', 'group', 'scenario', 'model', 'k']).any():
        raise ValueError('Duplicate retained parameter estimates')
    metric = pd.concat([pd.read_csv(run/f'{stage}_metrics.csv') for stage in ['select', 'evaluate']], ignore_index=True)
    records = []
    for (fold, group, sid), part in estimates.groupby(['fold', 'group', 'scenario'], sort=True):
        sid = int(sid)
        extra_values = metric[(metric.fold==fold) & (metric.group==group) & (metric.scenario==sid)].extrapolation.unique()
        if len(extra_values) != 1: raise ValueError('Ambiguous record scope')
        extra = bool(extra_values[0])
        scope = ('evaluation_extrapolation' if extra else 'evaluation_interpolation') if group=='evaluation' else group
        with np.load(run/f'{fold}/predictions_s{sid}.npz') as retained:
            origins = retained['k']; target = retained['target'][:, HORIZON_INDEX]
            physical = np.abs(np.rad2deg(retained['physical'][:, HORIZON_INDEX]-target)).mean(1)
            old = np.abs(np.rad2deg(retained['adaptive'][:, HORIZON_INDEX]-target)).mean(1)
            spacing = np.diff(origins)*.01
            for model, local in part.groupby('model', sort=True):
                local = local.sort_values('k')
                np.testing.assert_array_equal(local.k.to_numpy(), origins)
                bt = local.backtest_mse_rad2.to_numpy(dtype=float)
                nominal_bt = local.nominal_backtest_mse_rad2.to_numpy(dtype=float)
                if not np.isfinite(bt).all() or not np.isfinite(nominal_bt).all() or (bt<0).any() or (nominal_bt<0).any():
                    raise ValueError('Invalid retained MSE')
                error = np.abs(np.rad2deg(retained[model][:, HORIZON_INDEX]-target)).mean(1)
                lower_bt = bt < nominal_bt
                denominator = float(nominal_bt.mean())
                record = dict(fold=fold, group=group, scope=scope, scenario=sid, extrapolation=extra, model=model,
                    packets=len(origins), history_samples=41,
                    origin_spacing_min_s=number(spacing.min()) if len(spacing) else None,
                    origin_spacing_max_s=number(spacing.max()) if len(spacing) else None,
                    backtest_mse_mean_rad2=number(bt.mean()), nominal_backtest_mse_mean_rad2=denominator,
                    backtest_mse_ratio=number(bt.mean()/denominator) if denominator>0 else None,
                    ratio_defined=denominator>0,
                    accepted_adaptation_fraction=number(flags(local.accepted_adaptation).mean()),
                    guard_rejected_candidate_fraction=number(flags(local.guard_rejected_candidate).mean()),
                    adjacent_frequency_switch_fraction=number(np.mean(np.diff(local.frequency_ratio)!=0)) if len(origins)>1 else None,
                    adjacent_damping_switch_fraction=number(np.mean(np.diff(local.damping_per_s)!=0)) if len(origins)>1 else None,
                    future_mae_deg=number(error.mean()), physical_future_mae_deg=number(physical.mean()),
                    old_adaptive_future_mae_deg=number(old.mean()),
                    delta_vs_physical_deg=number((error-physical).mean()), delta_vs_old_adaptive_deg=number((error-old).mean()),
                    lower_backtest_fraction=number(lower_bt.mean()))
                for name, reference in [('physical', physical), ('old_adaptive', old)]:
                    worse = error>reference; better = error<reference
                    record[f'future_worse_than_{name}_fraction'] = number(worse.mean())
                    record[f'future_better_than_{name}_fraction'] = number(better.mean())
                    record[f'lower_backtest_and_worse_than_{name}_fraction'] = number((lower_bt & worse).mean())
                    record[f'lower_backtest_and_better_than_{name}_fraction'] = number((lower_bt & better).mean())
                records.append(record)
    records = pd.DataFrame(records)
    keys = ['fold', 'scope', 'model']
    columns = [c for c in records.columns if c.endswith('_fraction') or c.endswith('_deg') or c.endswith('_rad2') or c=='backtest_mse_ratio']
    summary = records.groupby(keys)[columns].mean().reset_index()
    counts = records.groupby(keys).agg(records=('scenario','size'), ratio_defined_records=('ratio_defined','sum')).reset_index()
    summary = summary.merge(counts, on=keys)
    out.mkdir(parents=True)
    records.to_csv(out/'per_record.csv', index=False)
    summary.to_csv(out/'equal_record_summary.csv', index=False)
    clean_summary = json.loads(summary.to_json(orient='records'))
    (out/'equal_record_summary.json').write_text(json.dumps(clean_summary, indent=2, allow_nan=False)+'\n')
    report = {'scope':'Post-hoc descriptive matched-window diagnosis; no fitting or new selection.',
        'source_run':str(run), 'source_provenance_sha256':sha256(run/'provenance.json'),
        'diagnostic_source_sha256':sha256(Path(__file__)),
        'guarded_dependencies_verified':len(provenance['guarded_hashes']),
        'retained_results_verified':len(provenance['result_hashes']),
        'record_model_rows':len(records), 'models':MODELS,
        'summary_weighting':'Equal whole-record weights within fold/scope/model. Fractions first computed within records.',
        'backtest_ratio_definition':'Mean final-choice historical holdout MSE divided by mean nominal holdout MSE within a record; no floor. Undefined if denominator is zero, with defined-record count retained.',
        'guarded_score_semantics':'Final selected historical MSE is nominal after fallback; no pre-guard score is substituted.',
        'comparisons':'Strict lower/greater comparisons without inferential significance; seven horizon absolute errors averaged per window.',
        'interpretation_limits':['All windows overlap; no independent-observation claim or p-values.',
            'Backtest scores with different prefixes do not represent identical held-back samples.',
            'Lower historical error and worse future error co-occurrence is descriptive, not causal attribution.',
            'Accepted adaptation includes damping-only changes; frequency switching alone omits them.',
            'No candidate or threshold changed after these diagnostics.']}
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(summary[(summary.fold=='primary') & (summary.scope=='evaluation_extrapolation')][
        ['model','backtest_mse_ratio','accepted_adaptation_fraction','guard_rejected_candidate_fraction',
         'future_mae_deg','physical_future_mae_deg','old_adaptive_future_mae_deg',
         'lower_backtest_and_worse_than_physical_fraction']].to_string(index=False))
    print('WROTE', out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('artifacts/runs/history_backtest_20261107'))
    parser.add_argument('--output', type=Path, default=Path('results/history_backtest_20261107/diagnostics'))
    args = parser.parse_args(); run_diagnostic(args.run, args.output)


if __name__=='__main__': main()
