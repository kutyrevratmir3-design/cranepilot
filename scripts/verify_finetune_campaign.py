"""Replay all frozen real forecasts, seeds and ensembles from history-only inputs.

No fitting occurs. The complete run, checkpoints, selection and original-file
snapshot are verified before models are loaded. Preserve every verification
attempt by choosing a fresh output path.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import numpy as np
from cranepilot.data import load_dataset, sha256

PROJECT_ROOT=Path(__file__).resolve().parents[1]
TRANSFER_ROOT=PROJECT_ROOT.parent.parent
TOLERANCE_DEG=5e-5
ACCELERATION_TOLERANCE=1e-7
ORIGINAL_COUNTS={'main':1111,'aphynity':170}


def utc_now(): return datetime.now(timezone.utc).isoformat()


def verify_mapping(mapping,base):
    if not isinstance(mapping,dict) or not mapping:raise ValueError('Nonempty retained hash mapping required')
    missing=[];changed=[];errors=[];checked=0
    for name,expected in mapping.items():
        path=base/name.replace('\\','/')
        if not path.is_file():missing.append(name);continue
        try:actual=sha256(path)
        except OSError as exc:errors.append({'path':name,'error':str(exc)});continue
        checked+=1
        if actual!=expected:changed.append({'path':name,'expected':expected,'actual':actual})
    return dict(passed=not(missing or changed or errors),expected_files=len(mapping),checked_files=checked,
                missing=missing,changed=changed,read_errors=errors)


def original_files():
    path=TRANSFER_ROOT/'handoff/restoration_20261107/original_file_hashes.json'
    snapshot=json.loads(path.read_text())
    if set(snapshot)!=set(ORIGINAL_COUNTS):raise ValueError('Unexpected original snapshot sections')
    roots={'main':PROJECT_ROOT,'aphynity':TRANSFER_ROOT/'aphynity_pilot/cranepilot'}
    result=dict(manifest=str(path),manifest_sha256=sha256(path),projects={})
    for name,count in ORIGINAL_COUNTS.items():
        if len(snapshot[name])!=count:raise ValueError(f'Original {name} snapshot count changed')
        result['projects'][name]=verify_mapping(snapshot[name],roots[name])
    result['passed']=all(v['passed'] for v in result['projects'].values())
    result['checked_files']=sum(v['checked_files'] for v in result['projects'].values())
    return result


def finite(value,shape,label):
    array=np.asarray(value)
    if array.shape!=shape or not np.isfinite(array).all():raise AssertionError(f'{label}: finite {shape} array required')
    return array


def audit_saved_parameters(run,provenance):
    """Check saved tensors against each corresponding frozen soft warm start."""
    import torch
    cfg=provenance['config'];broad=PROJECT_ROOT/cfg['broad_run']
    broad_path=broad/'provenance.json';broad_prov=json.loads(broad_path.read_text())
    if broad_prov['status']!='complete':raise ValueError('Completed warm-start research required')
    expected_keys={(fold,name,int(seed)) for fold in cfg['folds'] for name in cfg['variants'] for seed in cfg['seeds']}
    if len(expected_keys)!=90:raise ValueError('Expected ninety fitted models for parameter audit')
    fits=json.loads((run/'fits.json').read_text());fit_by_key={}
    for row in fits:
        key=(row['fold'],row['model'],int(row['seed']))
        if key in fit_by_key:raise ValueError('Duplicate fit metadata')
        fit_by_key[key]=row
    if set(fit_by_key)!=expected_keys:raise ValueError('Fit metadata does not cover all saved models')
    warm_paths={f'{fold}/soft_symmetry_seed{seed}.pt' for fold in cfg['folds'] for seed in cfg['seeds']}
    warm_hashes={name:broad_prov['checkpoint_hashes'][name] for name in warm_paths}
    warm_integrity=verify_mapping(warm_hashes,broad)
    if not warm_integrity['passed']:raise AssertionError('Warm-start checkpoint hashes changed')
    source_states={}
    for fold in cfg['folds']:
        for seed in cfg['seeds']:
            source_states[(fold,int(seed))]=torch.load(broad/fold/f'soft_symmetry_seed{seed}.pt',
                                                    map_location='cpu',weights_only=True)
    prefixes={'observer_only':('initial.','observer.'),
              'innovation_only':('hidden_initial.','innovation.','force_output.')}
    models=[];frozen_checks=0;fallback_checks=0
    for fold,name,seed in sorted(expected_keys):
        meta=fit_by_key[(fold,name,seed)];strategy=cfg['variants'][name]['strategy']
        if meta['strategy']!=strategy:raise AssertionError('Fit strategy disagrees with frozen config')
        selected_epoch=int(meta['selected_epoch'])
        if selected_epoch < -1 or ('best_epoch' in meta and int(meta['best_epoch'])!=selected_epoch):
            raise AssertionError('Inconsistent selected-epoch metadata')
        reference=source_states[(fold,seed)]
        current=torch.load(run/fold/f'{name}_seed{seed}.pt',map_location='cpu',weights_only=True)
        if current.keys()!=reference.keys():raise AssertionError('Saved state keys differ from warm start')
        # This frozen architecture has only parameters in state_dict, no buffers.
        allowed={key for key in reference if strategy not in prefixes or key.startswith(prefixes[strategy])}
        if set(meta['trainable_parameter_names'])!=allowed:raise AssertionError('Trainable-name metadata violates declared freeze')
        parameters=[];modules={};sum_sq=0.;changed=0;frozen=0;fallback=0
        for key in sorted(reference):
            before=reference[key];after=current[key]
            if not isinstance(before,torch.Tensor) or not isinstance(after,torch.Tensor):raise AssertionError('Tensor state_dict required')
            if after.shape!=before.shape or after.dtype!=before.dtype:raise AssertionError('Saved tensor shape/dtype changed')
            if not torch.isfinite(before).all() or not torch.isfinite(after).all():raise AssertionError('Nonfinite checkpoint tensor')
            exact=bool(torch.equal(after,before));is_frozen=key not in allowed
            if is_frozen:
                frozen+=1
                if not exact:raise AssertionError(f'Frozen tensor changed: {fold}/{name}/seed{seed}/{key}')
            if selected_epoch==-1:
                fallback+=1
                if not exact:raise AssertionError(f'Epoch -1 fallback changed: {fold}/{name}/seed{seed}/{key}')
            delta=(after.to(torch.float64)-before.to(torch.float64)).reshape(-1)
            squared=float(torch.dot(delta,delta));sum_sq+=squared
            changes=int(torch.count_nonzero(after!=before));changed+=changes
            module=key.split('.',1)[0]
            item=modules.setdefault(module,dict(squared_l2_delta=0.,parameter_elements=0,changed_elements=0,tensors=0))
            item['squared_l2_delta']+=squared;item['parameter_elements']+=after.numel()
            item['changed_elements']+=changes;item['tensors']+=1
            parameters.append(dict(name=key,shape=list(after.shape),dtype=str(after.dtype),elements=after.numel(),
                frozen=is_frozen,exactly_equal=exact,changed_elements=changes,l2_delta=float(np.sqrt(squared))))
        for item in modules.values():item['l2_delta']=float(np.sqrt(item.pop('squared_l2_delta')))
        total=sum(p.numel() for p in current.values());trainable=sum(current[key].numel() for key in allowed)
        if total!=meta['total_parameter_count'] or trainable!=meta['trainable_parameter_count']:
            raise AssertionError('Parameter counts disagree with fit metadata')
        frozen_checks+=frozen;fallback_checks+=fallback
        models.append(dict(fold=fold,model=name,seed=seed,strategy=strategy,selected_epoch=selected_epoch,
            source_checkpoint=str((broad/fold/f'soft_symmetry_seed{seed}.pt').relative_to(PROJECT_ROOT)),
            checkpoint=str((run/fold/f'{name}_seed{seed}.pt').relative_to(run)),
            total_parameter_count=total,trainable_parameter_count=trainable,all_weights_finite=True,
            frozen_tensors_checked=frozen,fallback_tensors_checked=fallback,changed_elements=changed,
            total_l2_delta=float(np.sqrt(sum_sq)),modules=modules,parameters=parameters))
    return dict(passed=True,models_checked=len(models),warm_start_integrity=warm_integrity,
                warm_start_provenance_sha256=sha256(broad_path),comparison='torch.equal; L2 drift accumulated in float64',
                frozen_tensor_checks=frozen_checks,fallback_tensor_checks=fallback_checks,
                initial_checkpoint_selections=sum(m['selected_epoch']==-1 for m in models),
                unchanged_models=sum(m['changed_elements']==0 for m in models),
                max_total_l2_delta=max(m['total_l2_delta'] for m in models),models=models)


def replay(run,provenance,selection,report):
    import torch
    from cranepilot.rollout import command_windows
    from scripts.finetune_candidate import FinetuneCandidate
    from scripts.global_research_candidate import CONTROLS
    from scripts.predict_cranepilot import build_packet
    torch.set_num_threads(4)
    cfg=provenance['config'];variants=set(cfg['variants']);seeds=list(cfg['seeds'])
    if len(cfg['folds'])!=3 or len(set(cfg['folds']))!=3 or len(variants)!=10 or len(seeds)!=3 or len(set(seeds))!=3:
        raise ValueError('Expected fixed three folds, three seeds and ten fine-tune variants')
    expected_models=set(CONTROLS)|{'previous_local'}|variants
    if len(expected_models)!=15:raise ValueError('Expected fifteen real ensemble/control forecasts')
    expected_checkpoints={f'{fold}/{name}_seed{seed}.pt' for fold in cfg['folds'] for name in variants for seed in seeds}
    if set(provenance['checkpoint_hashes'])!=expected_checkpoints:raise ValueError('Checkpoint coverage differs from ninety frozen fits')
    source=json.loads((PROJECT_ROOT/cfg['source_run']/'provenance.json').read_text())
    if source['status']!='complete':raise ValueError('Completed split source required')
    data=load_dataset(PROJECT_ROOT/'data/processed')
    report.update(folds=list(cfg['folds']),seeds=seeds,variants=sorted(variants),fitted_models_total=90,
                  real_methods_per_fold=15,history_only_inputs=True,history_samples=41,prediction_steps=30,
                  target_comparison='exact_array_equal',records=[],checks=[])
    for fold in cfg['folds']:
        folder=run/fold;split=source['config']['folds'][fold];groups={}
        for group in ('validation','evaluation','load_stress'):
            for sid in split[group]:
                if sid in groups:raise ValueError(f'Repeated scenario {fold}/{sid}')
                groups[sid]=group
        by_sid={}
        for path in folder.glob('predictions_s*.npz'):
            match=re.fullmatch(r'predictions_s([0-9]+)\.npz',path.name)
            if match is None:raise ValueError(f'Unexpected real prediction filename: {path}')
            sid=int(match.group(1))
            if sid in by_sid:raise ValueError('Repeated retained scenario')
            by_sid[sid]=path
        if set(by_sid)!=set(groups):raise AssertionError(f'Real-record coverage mismatch in {fold}')
        predictor=FinetuneCandidate(cfg,fold,folder)
        if set(predictor.ensembles)!=variants:raise AssertionError('Loaded ensemble set differs from fixed variants')
        for sid in sorted(by_sid):
            path=by_sid[sid];relative=str(path.relative_to(run));scenario=data[sid]
            if relative not in provenance['result_hashes']:raise AssertionError('Prediction file lacks result hash')
            with np.load(path,allow_pickle=False) as saved:
                k=saved['k']
                if k.ndim!=1 or not len(k) or not np.issubdtype(k.dtype,np.integer):raise AssertionError('Integer origins required')
                expected_k=np.arange(40,len(scenario.time)-150,cfg['evaluation_stride'])
                np.testing.assert_array_equal(k,expected_k,err_msg=relative)
                n=len(k);raw=command_windows(scenario,predictor.calibration,k)
                target=finite(saved['target'],(n,30),relative+'/target')
                np.testing.assert_array_equal(target,raw['target_q'],err_msg=relative)
                saved_acc=finite(saved['acceleration'],(n,30),relative+'/acceleration')
                past=k[:,None]-np.arange(41);future=k[:,None]+np.arange(30)*5
                packet=build_packet(scenario.signals['theta'][past],scenario.signals['x'][past],
                                    scenario.signals['xr'][past],scenario.signals['L'][k],
                                    scenario.signals['xr'][future],predictor.calibration)
                actual,seed_predictions,acceleration=predictor.forecast(packet)
                if set(actual)!=expected_models or set(seed_predictions)!=variants:raise AssertionError('Forecast model sets changed')
                acceleration=finite(acceleration,(n,30),'replayed acceleration')
                acc_error=float(np.max(np.abs(acceleration-saved_acc)))
                if acc_error>ACCELERATION_TOLERANCE:raise AssertionError(f'Actuator replay differs: {acc_error}')
                aliases=set() if groups[sid]=='validation' else {'selected','common'}
                expected_fields={'k','target','acceleration'}|expected_models|aliases|{name+'__seeds' for name in variants}
                if set(saved.files)!=expected_fields:raise AssertionError(f'Saved field coverage differs: {relative}')
                resolved={'selected':selection['folds'][fold]['model'],'common':selection['common']['model']}
                for alias in aliases:
                    if resolved[alias] not in actual:raise AssertionError('Selected method missing from fixed predictions')
                    actual[alias]=actual[resolved[alias]]
                maximum=0.;ensemble_count=0;seed_count=0
                for name in sorted(expected_models|aliases):
                    expected=finite(saved[name],(n,30),relative+'/'+name)
                    predicted=finite(actual[name],(n,30),'replayed '+name)
                    error=float(np.max(np.abs(np.rad2deg(predicted-expected))))
                    if error>TOLERANCE_DEG:raise AssertionError(f'{relative}/{name}: {error} degrees')
                    maximum=max(maximum,error);ensemble_count+=1
                    report['checks'].append(dict(kind='ensemble_or_control',fold=fold,scenario=sid,group=groups[sid],
                        model=name,resolved_model=resolved.get(name,name),packets=n,max_error_deg=error))
                for name in sorted(variants):
                    expected=finite(saved[name+'__seeds'],(len(seeds),n,30),relative+'/'+name+'__seeds')
                    predicted=finite(seed_predictions[name],(len(seeds),n,30),'replayed '+name+' seeds')
                    np.testing.assert_array_equal(actual[name],predicted.mean(0),err_msg=name+' ensemble mean')
                    np.testing.assert_array_equal(saved[name],expected.mean(0),err_msg=name+' retained ensemble mean')
                    for index,seed in enumerate(seeds):
                        error=float(np.max(np.abs(np.rad2deg(predicted[index]-expected[index]))))
                        if error>TOLERANCE_DEG:raise AssertionError(f'{relative}/{name}/seed{seed}: {error} degrees')
                        maximum=max(maximum,error);seed_count+=1
                        report['checks'].append(dict(kind='individual_seed',fold=fold,scenario=sid,group=groups[sid],
                            model=name,seed=seed,packets=n,max_error_deg=error))
                report['records'].append(dict(fold=fold,scenario=sid,group=groups[sid],file=relative,packets=n,
                    ensemble_comparisons=ensemble_count,seed_comparisons=seed_count,targets_exact=True,
                    ensemble_means_exact=True,all_shapes_finite=True,max_error_deg=maximum,
                    acceleration_max_abs_difference_m_s2=acc_error))
            print(f'REPLAY {fold} scenario={sid} packets={n} ensembles={ensemble_count} seeds={seed_count} max_deg={maximum:.9g}',flush=True)
    report.update(real_record_files=len(report['records']),packet_count=sum(r['packets'] for r in report['records']),
                  prediction_comparisons=len(report['checks']),
                  ensemble_prediction_comparisons=sum(r['ensemble_comparisons'] for r in report['records']),
                  individual_seed_comparisons=sum(r['seed_comparisons'] for r in report['records']),
                  max_error_deg=max(r['max_error_deg'] for r in report['checks']))


def main():
    os.chdir(PROJECT_ROOT)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('artifacts/runs/finetune_campaign_20261107'))
    parser.add_argument('--output',type=Path,default=Path('results/finetune_campaign_20261107/replay_verification.json'))
    args=parser.parse_args();run=args.run.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError('Preserve previous verification; choose a fresh output')
    provenance_path=run/'provenance.json';provenance=json.loads(provenance_path.read_text())
    if provenance['status']!='complete':raise ValueError('Wait for completed training and evaluation before replay')
    report=dict(passed=False,run=str(run),started_utc=utc_now(),tolerance_deg=TOLERANCE_DEG,relative_tolerance=0.,
                acceleration_tolerance_m_s2=ACCELERATION_TOLERANCE,fitting_performed=False,
                verifier_sha256=sha256(Path(__file__)),provenance_sha256=sha256(provenance_path))
    try:
        integrity={key:verify_mapping(provenance[key],base) for key,base in
                   [('guarded_hashes',PROJECT_ROOT),('checkpoint_hashes',run),('result_hashes',run)]}
        integrity['selection']=verify_mapping({'selection.json':provenance['selection_sha256']},run)
        parent=PROJECT_ROOT/provenance['config']['parent_run']
        parent_provenance=json.loads((parent/'provenance.json').read_text())
        integrity['parent_checkpoints']=verify_mapping(parent_provenance['checkpoint_hashes'],parent)
        integrity['parent_selection']=verify_mapping({'selection.json':parent_provenance['selection_sha256']},parent)
        report['integrity']=integrity;report['original_files']=original_files()
        if not all(v['passed'] for v in integrity.values()):raise AssertionError('Retained evidence hash verification failed')
        if not report['original_files']['passed']:raise AssertionError('Original main/APHYNITY files changed')
        print('HASHES verified; original files main=1111 aphynity=170',flush=True)
        report['parameter_audit']=audit_saved_parameters(run,provenance)
        print('PARAMETERS verified; ninety fits, frozen tensors and epoch -1 fallbacks exact',flush=True)
        selection=json.loads((run/'selection.json').read_text());replay(run,provenance,selection,report)
        report['passed']=True
    except Exception as exc:
        report['failure']={'type':type(exc).__name__,'message':str(exc)};raise
    finally:
        report['finished_utc']=utc_now();output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x') as handle:json.dump(report,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps({key:report[key] for key in ['passed','real_record_files','packet_count','prediction_comparisons',
          'ensemble_prediction_comparisons','individual_seed_comparisons','max_error_deg']},indent=2),flush=True)


if __name__=='__main__':main()
