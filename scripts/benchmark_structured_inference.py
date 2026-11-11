"""Batch-one CPU latency for frozen structured angular ensembles plus actuator.

Only execute after the campaign is complete. Cold imports/loading, evidence
hashing and physical packet assembly are excluded. Timed work includes actor,
forced observer, tensor conversion, all three angular seeds and ensemble mean.
This is a descriptive local timing experiment, not a real-time guarantee.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from cranepilot.data import load_dataset,sha256
from scripts.verify_finetune_campaign import PROJECT_ROOT,verify_mapping

WARMUPS=3
REPEATS=10
THREADS=4
ORDER_SEED=20261107
TOLERANCE_DEG=5e-5
REFERENCE_INTERVAL_MS=50.


def utc_now():return datetime.now(timezone.utc).isoformat()


def benchmark(run,provenance,selection,report,rows):
    import torch
    from cranepilot.benchmark import environment_record
    from scripts.structured_candidate import StructuredCandidate
    from scripts.predict_cranepilot import build_packet
    from scripts.run_broad_search import inputs,predict
    torch.set_num_threads(THREADS)
    cfg=provenance['config'];rng=np.random.default_rng(ORDER_SEED)
    if len(cfg['seeds'])!=3 or len(set(cfg['seeds']))!=3:raise ValueError('Three frozen ensemble seeds required')
    source=json.loads((PROJECT_ROOT/cfg['source_run']/'provenance.json').read_text())
    if source['status']!='complete':raise ValueError('Completed split source required')
    data=load_dataset(PROJECT_ROOT/'data/processed')
    report.update(environment=environment_record(),cpu_threads=torch.get_num_threads(),
                  cpu_interop_threads=torch.get_num_interop_threads(),batch_size=1,ensemble_seeds=list(cfg['seeds']),
                  warmups_per_packet_method=WARMUPS,repeats_per_packet_method=REPEATS,order_seed=ORDER_SEED,
                  selection=[],verification_checks=[],packets=[])
    for fold in cfg['folds']:
        # Loading the complete parent/candidate is outside the timed region.
        # Only the explicitly selected three-model lists are evaluated below.
        candidate=StructuredCandidate(cfg,fold,run/fold)
        methods={'soft_symmetry':candidate.parent.models['soft_symmetry']}
        methods.update(candidate.ensembles)
        selected=selection['folds'][fold]['model']
        if selected in candidate.ensembles:
            methods[selected]=candidate.ensembles[selected]
            selected_status=dict(fold=fold,selected_model=selected,measured=True,
                                 timing_key=selected,reason='Selected new structured ensemble')
        else:
            selected_status=dict(fold=fold,selected_model=selected,measured=False,
                reason='Selected method is a retained control or previous_local; no full-candidate timing is substituted')
        report['selection'].append(selected_status)
        candidate.actor.cpu().eval()
        for models in methods.values():
            if len(models)!=3:raise AssertionError('Exactly three angular models required')
            for model in models:model.cpu().eval()
        if any(p.device.type!='cpu' for p in candidate.actor.parameters()):raise AssertionError('CPU actor required')
        if any(p.dtype!=torch.float32 for models in methods.values() for model in models for p in model.parameters()):
            raise AssertionError('Frozen float32 angular models required')

        @torch.no_grad()
        def forward(packet,models):
            prepared=inputs(packet,candidate.actor)
            return np.stack([predict(model,prepared) for model in models]).mean(0)

        for sid in source['config']['folds'][fold]['validation']:
            scenario=data[sid];path=run/fold/f'predictions_s{sid}.npz'
            with np.load(path,allow_pickle=False) as saved:
                origins=saved['k']
                np.testing.assert_array_equal(origins,np.arange(40,len(scenario.time)-150,cfg['evaluation_stride']))
                indices=np.array([0,len(origins)//2,len(origins)-1])
                if len(np.unique(indices))!=3:raise AssertionError('Three distinct validation packets required')
                for packet_number,index in enumerate(indices):
                    k=int(origins[index]);past=k-np.arange(41);future=k+np.arange(30)*5
                    # Packet construction is deliberately excluded from latency.
                    packet=build_packet(scenario.signals['theta'][past][None],scenario.signals['x'][past][None],
                        scenario.signals['xr'][past][None],np.array([scenario.signals['L'][k]]),
                        scenario.signals['xr'][future][None],candidate.calibration)
                    report['packets'].append(dict(fold=fold,scenario=int(sid),origin_k=k,retained_index=int(index)))
                    references={name:np.asarray(saved[name][index:index+1]) for name in methods}

                    def check(name,prediction,phase,repeat):
                        if prediction.shape!=(1,30) or not np.isfinite(prediction).all():raise AssertionError('Finite batch-one thirty-step forecast required')
                        reference=references[name]
                        if reference.shape!=(1,30) or not np.isfinite(reference).all():raise AssertionError('Invalid retained validation forecast')
                        delta=float(np.max(np.abs(np.rad2deg(prediction-reference))))
                        if delta>TOLERANCE_DEG:raise AssertionError(f'Batch-one replay differs: {fold}/{sid}/{name}: {delta} deg')
                        report['verification_checks'].append(dict(fold=fold,scenario=int(sid),origin_k=k,
                            model=name,phase=phase,repeat=repeat,max_error_deg=delta))
                        return delta

                    for name,models in methods.items():
                        for warmup in range(WARMUPS):check(name,forward(packet,models),'warmup',warmup)
                    # Interleaved fixed-seed order reduces systematic first-method
                    # advantage; CPU work is synchronous so no GPU sync is needed.
                    for repeat in range(REPEATS):
                        for name in rng.permutation(list(methods)):
                            started=time.perf_counter_ns();prediction=forward(packet,methods[name])
                            elapsed_ms=(time.perf_counter_ns()-started)/1e6
                            delta=check(name,prediction,'measured',repeat)
                            rows.append(dict(fold=fold,scenario=int(sid),packet=packet_number,origin_k=k,
                                model=str(name),repeat=repeat,elapsed_ms=elapsed_ms,prediction_max_error_deg=delta))
            print(f'LATENCY {fold} scenario={sid} methods={list(methods)} packets=3 repeats={REPEATS}',flush=True)
    frame=pd.DataFrame(rows);summary=[]
    for (fold,method),part in frame.groupby(['fold','model'],sort=True):
        timings=part.elapsed_ms.to_numpy();packet_count=len(part[['scenario','origin_k']].drop_duplicates())
        if len(timings)!=packet_count*REPEATS:raise AssertionError('Unexpected timing count')
        summary.append(dict(fold=fold,model=method,measurements=len(timings),packets=packet_count,
            median_ms=float(np.median(timings)),p95_ms=float(np.quantile(timings,.95)),max_ms=float(timings.max()),
            min_ms=float(timings.min()),mean_ms=float(timings.mean()),
            fraction_over_50ms=float(np.mean(timings>REFERENCE_INTERVAL_MS)),
            p95_over_reference_interval=float(np.quantile(timings,.95)/REFERENCE_INTERVAL_MS)))
    report.update(summary=summary,timed_forecasts=len(rows),
        max_prediction_error_deg=max(check['max_error_deg'] for check in report['verification_checks']))


def main():
    os.chdir(PROJECT_ROOT)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('artifacts/runs/structured_models_20261107'))
    parser.add_argument('--output',type=Path,default=Path('results/structured_models_20261107'))
    args=parser.parse_args();run=args.run.resolve();out=args.output.resolve()
    json_path=out/'inference_latency.json';csv_path=out/'inference_latency.csv'
    if json_path.exists() or csv_path.exists():raise FileExistsError('Preserve previous timing evidence; choose a fresh output directory')
    path=run/'provenance.json';provenance=json.loads(path.read_text())
    if provenance['status']!='complete':raise ValueError('Only benchmark a completed frozen campaign')
    report=dict(passed=False,run=str(run),started_utc=utc_now(),benchmark_source_sha256=sha256(Path(__file__)),
        hash_helper_source_sha256=sha256(PROJECT_ROOT/'scripts/verify_finetune_campaign.py'),provenance_sha256=sha256(path),
        fitting_performed=False,reference_interval_ms=REFERENCE_INTERVAL_MS,real_time_guarantee=False,
        timing_includes=['actor forward','forced observer refinement','feature/tensor conversion',
                        'three angular seed forwards','numpy conversion and ensemble mean','no_grad/eval wrapper overhead'],
        timing_excludes=['cold Python imports','model/checkpoint loading','filesystem and SHA checks',
                        'physical packet assembly','forecast verification against retained output',
                        'sensor I/O','controller optimization','scheduling and deployment integration'],
        interpretation='Descriptive warmed batch-one CPU latency on three validation packets per record; 50 ms is a comparison reference, not a deadline guarantee.',
        replay_tolerance_deg=TOLERANCE_DEG)
    rows=[]
    try:
        integrity={key:verify_mapping(provenance[key],base) for key,base in
                   [('guarded_hashes',PROJECT_ROOT),('checkpoint_hashes',run),('result_hashes',run)]}
        integrity['selection']=verify_mapping({'selection.json':provenance['selection_sha256']},run)
        report['integrity']=integrity
        if not all(item['passed'] for item in integrity.values()):raise AssertionError('Retained campaign evidence changed')
        selection=json.loads((run/'selection.json').read_text());benchmark(run,provenance,selection,report,rows)
        report['passed']=True
    except Exception as exc:
        report['failure']={'type':type(exc).__name__,'message':str(exc)};raise
    finally:
        report['finished_utc']=utc_now();out.mkdir(parents=True,exist_ok=True)
        with csv_path.open('x') as handle:
            pd.DataFrame(rows,columns=['fold','scenario','packet','origin_k','model','repeat','elapsed_ms','prediction_max_error_deg']).to_csv(handle,index=False)
        with json_path.open('x') as handle:json.dump(report,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps({'passed':report['passed'],'timed_forecasts':report['timed_forecasts'],
                      'max_prediction_error_deg':report['max_prediction_error_deg'],'summary':report['summary']},indent=2),flush=True)


if __name__=='__main__':main()
