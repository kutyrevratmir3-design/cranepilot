"""Post-hoc descriptive fit/prediction diagnostics; no new predictor selection.

Recompute only the frozen nominal observer residual. Read all adaptive forecasts
and parameter estimates from the completed retained run. Overlapping windows
are not independent observations; summaries give each whole record equal weight.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import load_dataset, sha256, QUANTIZATION_RAD
from cranepilot.physics import LengthCalibration
from cranepilot.rollout import command_windows, HORIZON_INDEX
from scripts.forced_adaptive_physics import forced_fit


def number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def run_diagnostic(run, out):
    if out.exists():
        raise FileExistsError('Preserve earlier diagnostic')
    provenance = json.loads((run/'provenance.json').read_text())
    if provenance['status'] != 'complete':
        raise ValueError('Completed experiment required')
    for mapping, base in [('guarded_hashes', Path('.')), ('result_hashes', run)]:
        for relative, digest in provenance[mapping].items():
            if sha256(base/relative.replace('\\','/')) != digest:
                raise ValueError(f'Changed retained dependency/result: {relative}')
    source = Path(provenance['config']['source_run'])
    original = json.loads((source/'provenance.json').read_text())
    data = load_dataset(Path('data/processed'))
    estimates = pd.concat([pd.read_csv(run/'select_parameter_estimates.csv'),
                           pd.read_csv(run/'evaluate_parameter_estimates.csv')], ignore_index=True)
    if estimates.duplicated(['fold','group','scenario','model','k']).any():
        raise ValueError('Duplicate parameter estimates')
    records, subsets = [], []
    for (fold, group, sid), part in estimates.groupby(['fold','group','scenario'], sort=True):
        sid = int(sid)
        calibration = LengthCalibration(**original['inputs'][fold]['calibration'])
        split = original['config']['folds'][fold]
        train_lengths = [float(data[s].signals['L'][0]) for s in split['train']]
        length = float(data[sid].signals['L'][0])
        extra = length < min(train_lengths) or length > max(train_lengths)
        scope = ('evaluation_extrapolation' if extra else 'evaluation_interpolation') if group == 'evaluation' else group
        with np.load(run/f'{fold}/predictions_s{sid}.npz') as retained:
            origins = retained['k']
            raw = command_windows(data[sid], calibration, origins)
            np.testing.assert_array_equal(raw['target_q'], retained['target'])
            theta = raw['angular_context'][:,:41]*.1
            x = raw['actuator_context'][:,:41]*.1
            _, _, residual = forced_fit(theta,x,raw['omega'],0.,velocity=raw['initial_velocity'])
            nominal_mse = np.mean(residual**2,axis=1)
            nominal_error = np.abs(np.rad2deg(retained['physical'][:,HORIZON_INDEX]-retained['target'][:,HORIZON_INDEX])).mean(1)
            spacing = np.diff(origins)*.01
            for model, local in part.groupby('model',sort=True):
                local = local.sort_values('k')
                np.testing.assert_array_equal(local.k.to_numpy(),origins)
                error = np.abs(np.rad2deg(retained[model][:,HORIZON_INDEX]-retained['target'][:,HORIZON_INDEX])).mean(1)
                changed = local.frequency_ratio.to_numpy()!=1.
                boundary = local.ratio_boundary_hit.to_numpy(dtype=bool)
                frequency_grid = provenance['config']['variants'][model]['ratios']
                frequency_boundary_applicable = len(set(frequency_grid))>1
                keys = dict(fold=fold,group=group,scope=scope,scenario=sid,length_m=length,extrapolation=extra,model=model)
                records.append(dict(**keys,packets=len(origins),history_samples=41,
                    origin_spacing_min_s=number(spacing.min()),origin_spacing_max_s=number(spacing.max()),
                    nominal_history_rms_median_bins=number(np.median(np.sqrt(nominal_mse))/QUANTIZATION_RAD),
                    adaptive_history_rms_median_bins=number(local.history_rms_rad.median()/QUANTIZATION_RAD),
                    history_mse_ratio=number(local.history_mse_rad2.mean()/nominal_mse.mean()),
                    future_mae_deg=number(error.mean()),nominal_future_mae_deg=number(nominal_error.mean()),
                    future_delta_mae_deg=number((error-nominal_error).mean()),future_worse_packet_fraction=number(np.mean(error>nominal_error)),
                    frequency_boundary_applicable=frequency_boundary_applicable,
                    frequency_boundary_fraction=number(boundary.mean()) if frequency_boundary_applicable else None,
                    frequency_changed_fraction=number(changed.mean()),
                    damping_max_fraction=number(np.mean(local.damping_per_s==max(provenance['config']['variants'][model]['dampings']))),
                    adjacent_frequency_switch_fraction=number(np.mean(np.diff(local.frequency_ratio)!=0)),
                    adjacent_damping_switch_fraction=number(np.mean(np.diff(local.damping_per_s)!=0))))
                masks = {'nominal_frequency':~changed,'changed_frequency':changed,
                         'zero_damping':local.damping_per_s.to_numpy()==0,
                         'positive_damping':local.damping_per_s.to_numpy()>0}
                if frequency_boundary_applicable:
                    masks.update(frequency_interior=~boundary,frequency_boundary=boundary)
                for subset, mask in masks.items():
                    if not mask.any():
                        subsets.append(dict(**keys,subset=subset,packets=0,future_mae_deg=None,nominal_future_mae_deg=None,delta_mae_deg=None))
                    else:
                        subsets.append(dict(**keys,subset=subset,packets=int(mask.sum()),future_mae_deg=number(error[mask].mean()),
                            nominal_future_mae_deg=number(nominal_error[mask].mean()),delta_mae_deg=number((error[mask]-nominal_error[mask]).mean())))
    records = pd.DataFrame(records); subsets = pd.DataFrame(subsets)
    summary_columns = ['history_mse_ratio','future_mae_deg','nominal_future_mae_deg','future_delta_mae_deg',
                       'future_worse_packet_fraction','frequency_boundary_fraction','frequency_changed_fraction','damping_max_fraction',
                       'adjacent_frequency_switch_fraction','adjacent_damping_switch_fraction',
                       'nominal_history_rms_median_bins','adaptive_history_rms_median_bins']
    summary = records.groupby(['fold','scope','model'])[summary_columns].mean().reset_index()
    count = records.groupby(['fold','scope','model']).size().rename('records').reset_index()
    summary = summary.merge(count,on=['fold','scope','model'])
    # Compare only residual scales, never treat synthetic cases as measured records.
    sim_rows = []
    for fold in provenance['config']['folds']:
        meta = pd.read_csv(run/f'{fold}/simulation_cases.csv')
        for model in provenance['config']['variants']:
            fitted = pd.read_csv(run/f'{fold}/simulation_estimates_{model}.csv').sort_values('case')
            np.testing.assert_array_equal(fitted['case'],np.arange(len(meta)))
            together = meta.copy(); together['history_rms_bins']=fitted.history_rms_rad.to_numpy()/QUANTIZATION_RAD
            for key, values in together.groupby(['force_law','drive_amplitude_m','future_acceleration_gain','measurement']):
                sim_rows.append(dict(fold=fold,model=model,force_law=key[0],drive_amplitude_m=key[1],future_acceleration_gain=key[2],
                    measurement=key[3],cases=len(values),history_rms_median_bins=number(values.history_rms_bins.median())))
    out.mkdir(parents=True)
    records.to_csv(out/'per_record.csv',index=False)
    subsets.to_csv(out/'parameter_subsets.csv',index=False)
    summary.to_csv(out/'equal_record_summary.csv',index=False)
    pd.DataFrame(sim_rows).to_csv(out/'simulation_history_residuals.csv',index=False)
    example = subsets[(subsets.fold=='primary') & (subsets.scenario==2) & (subsets.model=='adaptive') & (subsets.subset=='changed_frequency')]
    report = {'scope':'Post-hoc descriptive diagnosis, no new training or strategy selection. Only fixed nominal observer recomputed.',
        'overlapping_windows_independent':False,'summary_weighting':'Equal whole-record weights within fold/scope/model; all available records retained.',
        'source_run':str(run),'source_provenance_sha256':sha256(run/'provenance.json'),
        'diagnostic_source_sha256':sha256(Path(__file__)),'guarded_dependencies_verified':len(provenance['guarded_hashes']),
        'retained_results_verified':len(provenance['result_hashes']),'record_model_rows':len(records),
        'origin_spacing_s':sorted(records.origin_spacing_min_s.unique().tolist()),
        'scenario2_example_posthoc':example.to_dict('records'),
        'interpretation_limits':['A lower historical fit error does not establish predictive validity.',
          'Switching over overlapping histories is a sensitivity diagnostic, not proof of a particular noise source.',
          'Observer residual includes model mismatch and measurement effects; it is not a direct sensor-noise estimate.',
          'Singleton frequency grids are excluded from frequency-boundary rate interpretation.',
          'No inferential confidence intervals or causal attribution are claimed.']}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(summary[(summary.fold=='primary') & (summary.scope=='evaluation_extrapolation')].to_string(index=False))
    print('WROTE',out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('artifacts/runs/forced_adaptive_20261107'))
    parser.add_argument('--output',type=Path,default=Path('results/forced_adaptive_20261107/diagnostics'))
    args=parser.parse_args();run_diagnostic(args.run,args.output)


if __name__=='__main__':main()
