"""Complete local report; never select variants by evaluation performance."""
from pathlib import Path
import itertools
import json
import re
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def complete(path):
    p=json.loads((path/'provenance.json').read_text())
    if p['status']!='complete':raise RuntimeError(f'Incomplete campaign: {path}')
    return p


def markdown_table(frame,columns):
    rows=['| Model | '+' | '.join(f'{h:g} s' for h in columns)+' |',
          '|---|'+'---:|'*len(columns)]
    for name,row in frame.iterrows():rows.append('| '+str(name)+' | '+' | '.join(f'{row[h]:.4f}' for h in columns)+' |')
    return '\n'.join(rows)


def main():
    out=Path('results/latest');runs=Path('artifacts/runs')
    force=runs/'forceaware_campaign_v2';phase=runs/'phase_normalized_campaign'
    measurement=runs/'measurement_consistent_campaign';cross=runs/'cross_length_campaign';zero=runs/'zero_prior_gp_campaign'
    provenance={p.name:complete(p) for p in [force,phase,measurement,cross,zero]}
    primary=pd.read_csv(out/'forceaware_metrics.csv')
    parts=[primary]
    for run in [phase,measurement]:
        for sid in [1,2,7,8,10,11,12]:
            original=np.load(force/f'predictions_s{sid}.npz');new=np.load(run/f'predictions_s{sid}.npz')
            np.testing.assert_array_equal(original['k'],new['k']);np.testing.assert_array_equal(original['target'],new['target'])
        parts.append(pd.read_csv(run/'metrics.csv'))
    z=pd.read_csv(zero/'metrics.csv');parts.append(z.query("fold == 'primary'").drop(columns=['fold','transfer']))
    metrics=pd.concat(parts,ignore_index=True)
    summary=metrics.groupby(['group','model','regime','horizon_s'],as_index=False)[['mae_deg','rmse_deg']].mean()
    metrics.to_csv(out/'research_20261103_metrics.csv',index=False);summary.to_csv(out/'research_20261103_summary.csv',index=False)
    x=pd.read_csv(cross/'metrics.csv')
    x=pd.concat([x,z.query("fold != 'primary'")],ignore_index=True)
    x.to_csv(out/'cross_length_metrics.csv',index=False)
    cross_summary=x.groupby(['fold','group','transfer','model','regime','horizon_s'],as_index=False)[['mae_deg','rmse_deg']].mean()
    cross_summary.to_csv(out/'cross_length_summary.csv',index=False)
    evaluation=summary.query("group == 'evaluation' and regime == 'all'").pivot(index='model',columns='horizon_s',values='mae_deg')
    val=summary.query("group == 'validation' and regime == 'all'").groupby('model').mae_deg.mean()
    candidates=list(provenance[force.name]['config']['variants'])+['phase_passive','phase_driven','measurement_driven']
    candidate=val.loc[candidates].idxmin()
    extra=x.query("group == 'evaluation' and regime == 'all' and transfer != 'interpolation'")
    macro=extra.groupby(['fold','model'],as_index=False).mae_deg.mean()
    score=evaluation.mean(axis=1)
    per_record=metrics.query("group == 'evaluation' and regime == 'all'").groupby(['scenario','model']).mae_deg.mean().unstack()
    delta=(per_record[candidate]-per_record.gp_action_residual).to_numpy()
    draws=np.array([np.mean(delta[list(index)]) for index in itertools.product(range(4),repeat=4)])
    checks=Path('.local/research_20261103_checks.txt').read_text(encoding='utf-8')
    test_count=int(re.search(r'Ran (\d+) tests',checks).group(1))
    report={'date':'2026-11-03','scope':'retrospective fixed-length conditional forecast; no claim of best-in-literature or new hardware control validation',
        'new_angular_neural_fits':36,'new_fold_actor_fits':2,'new_kernel_candidate_fits':56,
        'failed_before_training':1,'numerical_and_information_tests_passed':test_count,
        'best_new_neural_by_validation':candidate,'primary_macro_mae_deg':score.to_dict(),
        'improvement_vs_previous_driven_percent':float(100*(1-score[candidate]/score.driven)),
        'improvement_vs_first_pepm_percent':float(100*(1-score[candidate]/score.passive_m4)),
        'primary_candidate_minus_old_gp_per_record_deg':delta.tolist(),
        'four_record_descriptive_bootstrap_95pct_deg':np.quantile(draws,[.025,.975]).tolist(),
        'cross_length_extrapolation_macro_mae_deg':macro.to_dict('records'),
        'unique_length_evaluation_records_including_cross_ranges':sorted(set(x.query("group == 'evaluation'").scenario)|{1,2,7,8}),
        'stress_records':[10,11,12],
        'latency':json.loads((out/'inference_latency.json').read_text()),
        'inference_parity':json.loads(Path('.local/inference_parity.json').read_text()),
        'publication':'All work in this stage kept local; no GitHub writes'}
    (out/'research_20261103_report.json').write_text(json.dumps(report,indent=2))
    for run in [phase,measurement,cross,zero]:
        for name in ['provenance.json','validation_seeds.csv','training.csv','kernel_validation.csv','diagnostics.csv']:
            if (run/name).exists():shutil.copyfile(run/name,out/(run.name+'_'+name))
    shutil.copyfile('.local/research_20261103_checks.txt',out/'scientific_checks_20261103.txt')
    shutil.copyfile('.local/quantization_audit.json',out/'encoder_quantization_audit.json')
    shutil.copyfile('.local/inference_parity.json',out/'inference_parity.json')
    names={'driven':'Previous driven NN','gp_action_residual':'Previous GP (512)',
        'full_kernel':'Full GP (1024)','zero_prior_gp':'Zero-prior GP (1024)',
        'gru_reference':'Direct GRU reference',candidate:'Force-aware robust NN'}
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name,label in names.items():
        part=summary.query("group == 'evaluation' and regime == 'all' and model == @name")
        axes[0].plot(part.horizon_s,part.mae_deg,'o-',label=label,ms=3,lw=1.2)
    axes[0].set(title='Primary cable-length extrapolation',xlabel='Forecast horizon (s)',ylabel='Mean scenario MAE (degrees)')
    axes[0].grid(alpha=.2);axes[0].legend(fontsize=7)
    models=['forced_physics','forced_driven_huber','gru_reference','gp_reference','zero_prior_gp']
    for j,model in enumerate(models):
        values=[float(macro.query('fold == @fold and model == @model').mae_deg.iloc[0]) for fold in ['short_transfer','long_transfer']]
        axes[1].bar(np.arange(2)+(j-2)*.15,values,width=.14,label=model)
    axes[1].set(xticks=[0,1],xticklabels=['Shorter lengths','Longer lengths'],ylabel='Mean extrapolation MAE (degrees)',title='Additional ranges: extrapolation subset only')
    axes[1].legend(fontsize=7);axes[1].grid(axis='y',alpha=.2)
    for ext in ['png','svg']:fig.savefig(f'paper/figures/research_20261103.{ext}',dpi=190)
    plt.close(fig)
    folds_table=macro.pivot(index='fold',columns='model',values='mae_deg')
    text=f'''# Research stage completed 3 November 2026

This stage remains local under the user's publication preference. The separately
compiled 11-page manuscript predates these experiments.

## Retained work

36 angular neural fits, two freshly trained fold-specific actuators and 56 GP
candidate fits completed. All neural variants use three fixed seeds. One first
force-aware run failed before training because a reversed NumPy view could not
be converted to a tensor; its failed provenance is preserved and the corrected
run uses a fresh directory. All {test_count} numerical/information tests pass.

## Scientific changes

1. A forced oscillator observer uses past trolley positions through an integrated
   position response, avoiding a noisy second derivative of measured position.
2. Matched ablations compare raw/forced initialization and MSE/robust loss.
3. A phase-normalized angular model and a conventional recurrent GRU reference
   are both trained. Exact angular phase-similarity checks pass, but phase
   normalization does not improve the measured development result.
4. A nominal encoder-bin consistency ablation limits initial angle correction.
   TRAIN quantization grids were audited, including a constant offset in record 5.
   This restriction does not improve the best model and is retained as a negative
   result; it does not disprove the sensor's quantization or certify true-state bins.
5. Stronger GP references use 1024 training points, RBF/Matern kernels and a fixed
   validation grid. A further zero-residual-prior reference is added after observing
   extrapolation instability. This is retrospective baseline hardening, not a
   previously untouched confirmatory experiment. All scores remain in the report.
6. Two additional training ranges refit calibration AND the actuator independently
   in each fold. Interpolation is excluded from the extrapolation-only headline.

## Primary results

Best new neural variant by validation: **{candidate}**. No evaluation-based seed,
horizon mixture or threshold selection is used. Mean MAE over four evaluation
records and seven horizons is {score[candidate]:.5f} degrees versus {score.driven:.5f}
for the previous driven model, {score.passive_m4:.5f} for original PEPM,
{score.gp_action_residual:.5f} for the earlier GP and {score.zero_prior_gp:.5f}
for the new zero-prior GP. The gain over the previous driven model is
{report['improvement_vs_previous_driven_percent']:.2f} percent. Intermediate and
short horizons still have different winners; retain the complete table.

{markdown_table(evaluation,[.1,.2,.35,.5,.75,1.,1.5])}

The descriptive four-record paired bootstrap interval for candidate minus the
earlier GP is {report['four_record_descriptive_bootstrap_95pct_deg']}. These few
reused records cannot support a strong population-level confidence claim.

## Additional extrapolation ranges

Short transfer trains on records 4/6/8 and validates on 5. True short-length
extrapolation is records 1/2/3; record 7 is interpolation and reported separately.
Long transfer trains on 1/3/5 and validates on 4. True long-length extrapolation
is 6/7/8; record 2 is interpolation. Every compared model shares its fold's
fresh actuator and training-only length calibration. These are overlapping
reuses of eight physical records, not new independent physical trials.

'''
    text+='| Range | '+' | '.join(folds_table.columns)+' |\n|---|'+'---:|'*len(folds_table.columns)+'\n'
    for fold,row in folds_table.iterrows():text+='| '+fold+' | '+' | '.join(f'{v:.5f}' for v in row)+' |\n'
    text+='''
## Stress tests, computation and reuse

Additional load/shape records 10/11/12 at approximately the same length are
reported separately in research_20261103_summary.csv. They also change trajectory;
errors must not be attributed only to payload mass. The learned predictor is
still not evaluated inside MPC or on a newly controlled physical crane.

Inference has a target-free NPZ interface in scripts/predict_cranepilot.py.
It requires only the 41-sample history, current fixed length and prescribed
future reference. Its output matches the benchmark and records descriptive
seed spread, not calibrated uncertainty. The --fast option precomputes the
fixed Cayley/rotation transition and its input port once per window. It is
checked against the reference solver, including nonzero learned forcing and
encoder-bin anchoring. No weights are changed by that optimization.

Warm, single-packet CPU measurements include preprocessing, the shared actuator
and all three neural seeds; model loading and sensor acquisition are excluded.
See inference_latency.json for median/p95, processor and thread settings. The
GP remains a useful speed comparator. These timings are not a hard real-time
or closed-loop control guarantee.

## Evidence and limits

All run configurations, source/data/checkpoint hashes, calibration, training
logs, predictions and every evaluation scenario are retained. Source snapshots
are versioned locally. Full GP joblib checkpoints remain under artifacts/runs;
the compact handoff archive prioritizes neural weights and prediction evidence.

The benchmark card defines this exact niche. No best-in-literature claim is
made: literature comparison, released-method reproduction and stronger
independent evaluation remain necessary. Recent related work already includes
crane LPV identification with length dependence, physics-informed forecasting,
and GP-corrected MPC; see BENCHMARK_CARD.md for primary-source links.
'''
    Path('docs/RESEARCH_2026_11_03.md').write_text(text,encoding='utf-8')
    print(json.dumps(report,indent=2));print(folds_table.round(5).to_string())


if __name__=='__main__':main()
